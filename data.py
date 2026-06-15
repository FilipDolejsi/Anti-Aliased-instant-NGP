import os
import time
import json
import numpy as np
import torch
import imageio

class NeRFDataset:
    def __init__(self, root_dir, split='train', device='cpu'):
        self.root_dir = root_dir
        self.split = split
        self.device = device
        
        meta_path = os.path.join(root_dir, f'transforms_{split}.json')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Dataset metadata not found: {meta_path}")
            
        print(f"Loading {split} dataset from {meta_path}...")
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
            
        self.camera_angle_x = self.meta.get('camera_angle_x', 0.6194058656692505)
        
        self.images = [] 
        self.poses = []
        
        frames = self.meta['frames']
        
        print(f"  Processing {len(frames)} images...")
        start_t = time.time()
        
        for frame in frames:
            fname = os.path.join(root_dir, frame['file_path'] + '.png')
            try:
                im_data = imageio.imread(fname)
            except Exception:
                fname = os.path.join(root_dir, frame['file_path'].strip('./') + '.png')
                im_data = imageio.imread(fname)

            im_data = im_data.astype(np.float32) / 255.0
            
            if im_data.shape[-1] == 3:
                im_data = np.concatenate([im_data, np.ones_like(im_data[..., :1])], axis=-1)
            
            self.images.append(torch.from_numpy(im_data))
            self.poses.append(torch.from_numpy(np.array(frame['transform_matrix'], dtype=np.float32)))
            
        self.H, self.W = self.images[0].shape[:2]
        self.focal = 0.5 * self.W / np.tan(0.5 * self.camera_angle_x)
        
        if split == 'train':
            print("  Generating rays for training...")
            self.rays_o, self.rays_d, self.target_rgba = self.generate_all_rays()
            self.rays_o = self.rays_o.to(device)
            self.rays_d = self.rays_d.to(device)
            self.target_rgba = self.target_rgba.to(device)
            
        print(f"  Loaded {split} set in {time.time()-start_t:.2f}s")

    def generate_all_rays(self):
        i, j = torch.meshgrid(
            torch.arange(self.W, dtype=torch.float32), 
            torch.arange(self.H, dtype=torch.float32), 
            indexing='xy'
        )
        dirs = torch.stack([
            (i - self.W * 0.5) / self.focal, 
            -(j - self.H * 0.5) / self.focal, 
            -torch.ones_like(i)
        ], -1)
        
        rays_o_list = []
        rays_d_list = []
        rgba_list = []
        
        for idx in range(len(self.poses)):
            pose = self.poses[idx]
            ray_d = dirs @ pose[:3, :3].T
            ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
            ray_o = pose[:3, 3].expand_as(ray_d)
            
            rays_o_list.append(ray_o.reshape(-1, 3))
            rays_d_list.append(ray_d.reshape(-1, 3))
            rgba_list.append(self.images[idx].reshape(-1, 4))
            
        return torch.cat(rays_o_list, 0), torch.cat(rays_d_list, 0), torch.cat(rgba_list, 0)
        
    def get_rays_for_image(self, idx):
        pose = self.poses[idx].to(self.device)
        i, j = torch.meshgrid(
            torch.arange(self.W, dtype=torch.float32, device=self.device), 
            torch.arange(self.H, dtype=torch.float32, device=self.device), 
            indexing='xy'
        )
        dirs = torch.stack([
            (i - self.W * 0.5) / self.focal, 
            -(j - self.H * 0.5) / self.focal, 
            -torch.ones_like(i)
        ], -1)
        ray_d = dirs @ pose[:3, :3].T
        ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
        ray_o = pose[:3, 3].expand_as(ray_d)
        
        return ray_o, ray_d, self.images[idx].to(self.device)

    def shuffle(self):
        if self.split == 'train':
            perm = torch.randperm(self.rays_o.shape[0], device=self.device)
            self.rays_o = self.rays_o[perm]
            self.rays_d = self.rays_d[perm]
            self.target_rgba = self.target_rgba[perm]
            
    def __len__(self):
        return self.rays_o.shape[0] if self.split == 'train' else len(self.images)


class NeRFDataset360:
    """
    Loader for mip-NeRF 360 / LLFF format scenes (poses_bounds.npy + images_N/).
    Every 8th frame is held out as test/val (standard LLFF practice).
    Images are RGB only (no alpha); training target is raw pixel colour.
    """
    has_alpha = False

    def __init__(self, root_dir, split='train', device='cpu', downsample=4):
        self.root_dir = root_dir
        self.split = split
        self.device = device
        self.downsample = downsample

        start_t = time.time()
        print(f"Loading 360 {split} dataset from {root_dir} (downsample={downsample})...")

        # Load raw LLFF poses
        poses_arr = np.load(os.path.join(root_dir, 'poses_bounds.npy'))
        N = poses_arr.shape[0]

        # (N, 3, 5): rotation+translation in first 4 cols, [H,W,focal] in col 4
        poses = poses_arr[:, :15].reshape(N, 3, 5)
        bds   = poses_arr[:, 15:17]   # (N, 2): near, far per image

        # Extract intrinsics before convention swap (row 0=H, row 1=W, row 2=focal)
        H_raw    = int(np.round(poses[0, 0, 4]))
        W_raw    = int(np.round(poses[0, 1, 4]))
        focal_raw = float(poses[0, 2, 4])

        # LLFF -> NeRF convention: swap c2w cols 0<->1 with sign flip ([right,down,back] -> [right,up,back])
        poses = np.concatenate([
            poses[:, :, 1:2],    # new col 0 = old col 1
            -poses[:, :, 0:1],   # new col 1 = -old col 0
            poses[:, :, 2:]      # cols 2-4 unchanged
        ], axis=2)

        # c2w (N, 3, 4)
        c2w = poses[:, :3, :4].copy()

        # Recenter, then scale so the 90th-percentile camera distance = 2 units
        # (bds.min() alone can give extreme scales when near bounds are tiny).
        c2w[:, :3, 3] -= c2w[:, :3, 3].mean(axis=0)
        cam_dists = np.linalg.norm(c2w[:, :3, 3], axis=-1)
        scale = 2.0 / (np.percentile(cam_dists, 90) + 1e-8)
        c2w[:, :3, 3] *= scale

        print(f"  Camera distances after normalization: "
              f"min={cam_dists.min()*scale:.2f}  "
              f"mean={cam_dists.mean()*scale:.2f}  "
              f"max={cam_dists.max()*scale:.2f} units")

        # Adjusted intrinsics — set after loading images so H/W match actual file dims
        self._focal_raw = focal_raw
        self._W_raw     = W_raw

        # Train / test split: every 8th frame is test
        all_idx   = np.arange(N)
        test_idx  = all_idx[all_idx % 8 == 0]
        train_idx = all_idx[all_idx % 8 != 0]
        sel = train_idx if split == 'train' else test_idx

        self.poses = [torch.from_numpy(c2w[i].astype(np.float32)) for i in sel]

        # Load images
        imgdir = os.path.join(root_dir, f'images_{downsample}' if downsample > 1 else 'images')
        if not os.path.exists(imgdir):
            raise FileNotFoundError(f"Image directory not found: {imgdir}")

        imgfiles = sorted([
            os.path.join(imgdir, f) for f in os.listdir(imgdir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if len(imgfiles) != N:
            raise ValueError(f"Found {len(imgfiles)} images but {N} poses in poses_bounds.npy")

        self.images = []
        for i in sel:
            img = imageio.imread(imgfiles[i]).astype(np.float32) / 255.0
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            elif img.shape[-1] == 4:
                img = img[..., :3]
            self.images.append(torch.from_numpy(img))

        # Use actual image dims (images_N/ may differ by ±1px from H_raw//N due to rounding)
        actual_H, actual_W = self.images[0].shape[:2]
        self.H     = actual_H
        self.W     = actual_W
        self.focal = self._focal_raw * (actual_W / self._W_raw)

        # Generate flat ray tensors for training
        if split == 'train':
            self.rays_o, self.rays_d, self.target_rgb = self._gen_rays()
            # Keep on CPU — 360° scenes have too many rays to fit on GPU
            self.target_rgba = self.target_rgb   # no alpha — alias for train.py

        print(f"  Loaded {split} set: {len(self.images)} images "
              f"({self.H}×{self.W}, focal={self.focal:.1f}) in {time.time()-start_t:.2f}s")

    def _gen_rays(self):
        i, j = torch.meshgrid(
            torch.arange(self.W, dtype=torch.float32),
            torch.arange(self.H, dtype=torch.float32),
            indexing='xy'
        )
        dirs = torch.stack([
            (i - self.W * 0.5) / self.focal,
            -(j - self.H * 0.5) / self.focal,
            -torch.ones_like(i)
        ], dim=-1)

        rays_o_list, rays_d_list, rgb_list = [], [], []
        for pose, img in zip(self.poses, self.images):
            ray_d = dirs @ pose[:3, :3].T
            ray_d = ray_d / ray_d.norm(dim=-1, keepdim=True)
            ray_o = pose[:3, 3].expand_as(ray_d)
            rays_o_list.append(ray_o.reshape(-1, 3))
            rays_d_list.append(ray_d.reshape(-1, 3))
            rgb_list.append(img.reshape(-1, 3))

        return torch.cat(rays_o_list), torch.cat(rays_d_list), torch.cat(rgb_list)

    def get_rays_for_image(self, idx):
        pose = self.poses[idx].to(self.device)
        i, j = torch.meshgrid(
            torch.arange(self.W, dtype=torch.float32, device=self.device),
            torch.arange(self.H, dtype=torch.float32, device=self.device),
            indexing='xy'
        )
        dirs = torch.stack([
            (i - self.W * 0.5) / self.focal,
            -(j - self.H * 0.5) / self.focal,
            -torch.ones_like(i)
        ], dim=-1)
        ray_d = dirs @ pose[:3, :3].T
        ray_d = ray_d / ray_d.norm(dim=-1, keepdim=True)
        ray_o = pose[:3, 3].expand_as(ray_d)
        # Return rgb image with dummy alpha=1 so evaluation code stays uniform
        img = self.images[idx].to(self.device)
        rgba = torch.cat([img, torch.ones(*img.shape[:2], 1, device=self.device)], dim=-1)
        return ray_o, ray_d, rgba

    def shuffle(self):
        if self.split == 'train':
            perm = torch.randperm(self.rays_o.shape[0])  # CPU — rays live on CPU
            self.rays_o     = self.rays_o[perm]
            self.rays_d     = self.rays_d[perm]
            self.target_rgb = self.target_rgb[perm]
            self.target_rgba = self.target_rgb

    def __len__(self):
        return self.rays_o.shape[0] if self.split == 'train' else len(self.images)
