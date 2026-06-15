import math
import torch


class OccupancyGrid:
    THRESHOLD: float = 0.1

    def __init__(self, resolution: int = 128, device: str = "cuda", threshold: float = 0.1):
        self.resolution = resolution
        self.device = device
        R = resolution
        N = R ** 3
        self.THRESHOLD = threshold
        self.density = torch.full((N,), 100.0, dtype=torch.float32, device=device)
        self.grid = torch.ones(R, R, R, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, model, step: int, chunk_size: int = 262144):
        R = self.resolution
        N = R ** 3
        self.density.mul_(0.95)
        if step < 256:
            indices = torch.randperm(N, device=self.device)
        else:
            half = N // 4          # M/2 total split as M/4 + M/4
            uniform_idx = torch.randint(0, N, (half,), device=self.device)

            occ_idx = (self.density > self.THRESHOLD).nonzero(as_tuple=False).squeeze(1)
            if occ_idx.numel() > 0:
                perm = torch.randperm(occ_idx.numel(), device=self.device)[:half]
                occ_sample = occ_idx[perm]
                indices = torch.cat([uniform_idx, occ_sample])
            else:
                indices = uniform_idx

        iz = indices % R
        iy = (indices // R) % R
        ix = indices // (R * R)

        coords = (torch.stack([ix, iy, iz], dim=-1).float()
                  + torch.rand(len(indices), 3, device=self.device)) / R
        coords.clamp_(0.0, 1.0)

        dummy_d = torch.zeros(chunk_size, 3, device=self.device)
        dummy_d[:, 2] = -1.0

        sigma_list = []
        for i in range(0, len(coords), chunk_size):
            chunk = coords[i : i + chunk_size]
            n = chunk.shape[0]
            if hasattr(model, 'forward_single'):
                _, sigma = model.forward_single(chunk, dummy_d[:n])
            else:
                _, sigma = model(chunk, dummy_d[:n])
            sigma_list.append(sigma)
        sigmas = torch.cat(sigma_list)

        self.density[indices] = torch.maximum(self.density[indices], sigmas)

        self.grid = (self.density > self.THRESHOLD).reshape(R, R, R)

        occupied = self.grid.sum().item()
        if (step // 16) % 31 == 0:
            print(f"  [OccGrid step {step}] {occupied}/{N} voxels occupied "
                  f"({100.0 * occupied / N:.1f}%)")

    def query(self, pts_normalized: torch.Tensor) -> torch.Tensor:
        """Return bool mask (N,) — True for occupied voxels. Input in [0,1]^3."""
        R = self.resolution
        idx = (pts_normalized * R).long().clamp(0, R - 1)
        return self.grid[idx[:, 0], idx[:, 1], idx[:, 2]]
