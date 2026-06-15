import math
import torch
import torch.nn.functional as F
from config import Config
from profiler import profiler

try:
    import warp as wp
    HAS_WARP = True
except Exception:
    wp = None
    HAS_WARP = False

rendering_stats = {"avg_samples_per_ray": 0.0, "n_valid_rays": 0, "n_mlp_evals": 0}

# Step size in normalized [0,1]^3 space (paper Appendix E.1), scaled to world space in render_rays
_DDA_DT: float = math.sqrt(3.0) / 1024.0

def set_warp_dt(dt: float) -> None:
    global _DDA_DT
    _DDA_DT = dt


def contract_to_unit_cube(pts, aabb_min, aabb_max):
    return (pts - aabb_min) / (aabb_max - aabb_min)


def contract_mip360(pts):
    """
    mip-NeRF 360 spatial contraction: maps R³ → [-2,2]³.
    Points inside the unit sphere are unchanged; outside are contracted so
    the entire scene (including background) fits in a bounded domain.
    """
    norm = pts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.where(norm <= 1.0, pts, (2.0 - 1.0 / norm) * (pts / norm))


def normalize_contracted(pts):
    """Map contracted coords [-2,2]³ to [0,1]³ for the hash grid."""
    return (pts + 2.0) / 4.0


def _orthonormal_basis(d):
    """Two unit vectors perpendicular to d via stable Gram-Schmidt. d: (n,3)."""
    ref = torch.zeros_like(d)
    ref[:, 0] = 1.0
    parallel = d[:, 0].abs() > 0.99
    if parallel.any():
        ref[parallel, 0] = 0.0
        ref[parallel, 1] = 1.0
    u = ref - (ref * d).sum(-1, keepdim=True) * d
    u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    v = torch.cross(d, u, dim=-1)
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return u, v


def compute_hexagonal_multisamples(o, d, t0, t1, cone_radius, perturb,
                                   n_samples=6, collapse=False, sigma_scale=1.0):
    """
    Zip-NeRF Section 2, Equations 1-3: multisample pattern per conical frustum.

    o:           (n, 3)  ray origins
    d:           (n, 3)  ray directions (unit)
    t0:          (n, k)  frustum near distance
    t1:          (n, k)  frustum far distance
    cone_radius: float   pixel half-width per unit distance (= 1/focal_length)
    perturb:     bool    random rotation during training
    n_samples:   int     number of multisample points (6 = official Zip-NeRF hex)
    collapse:    bool    if True, zero lateral offsets (ablation, keeps sigma_j unchanged)

    Returns:
        x_world: (n, k, K, 3)  world-space multisample positions
        sigma_j: (n, k, K)     isotropic Gaussian scale per multisample
    """
    device = o.device
    n, k = t0.shape
    K = n_samples
    u, v = _orthonormal_basis(d)  # (n, 3)

    tc = (t0 + t1) * 0.5   # (n, k) frustum midpoint along ray
    td = (t1 - t0) * 0.5   # (n, k) frustum half-width
    sq2 = math.sqrt(2.0)

    if K == 6:
        # Official Zip-NeRF hex pattern (Eq. 1): ordering [0,2,4,3,5,1]×(π/3) gives isotropic covariance
        base_fwd = torch.tensor(
            [0.0, 2*math.pi/3, 4*math.pi/3, math.pi, 5*math.pi/3, math.pi/3],
            dtype=torch.float32, device=device,
        )
        base_rev = base_fwd.flip(0)

        if perturb:
            flip = torch.rand(n, k, 1, device=device) > 0.5
            angles_base = torch.where(
                flip.expand(n, k, 6),
                base_rev[None, None, :].expand(n, k, 6),
                base_fwd[None, None, :].expand(n, k, 6),
            )
            phi = torch.rand(n, k, 1, device=device) * 2.0 * math.pi
            angles = angles_base + phi
        else:
            k_idx = torch.arange(k, dtype=torch.float32, device=device)
            is_odd = (k_idx % 2 > 0.5)
            angles_base = torch.where(
                is_odd[None, :, None].expand(n, k, 6),
                base_rev[None, None, :].expand(n, k, 6),
                base_fwd[None, None, :].expand(n, k, 6),
            )
            shift = (k_idx % 2) * (math.pi / 6)
            angles = angles_base + shift[None, :, None]

        # Eq. 2: moment-matched t_j (specific to K=6 hexagonal pattern)
        j_vals = torch.arange(6, dtype=torch.float32, device=device)
        j_factor = 2.0 * j_vals / 5.0 - 1.0    # [-1, -0.6, -0.2, 0.2, 0.6, 1]
        inner = torch.sqrt((td**2 - tc**2)**2 + 4.0 * tc**4 + 1e-10)
        numer = (t1**2 + 2.0 * tc**2).unsqueeze(2) + \
                (3.0 / math.sqrt(7.0)) * j_factor[None, None, :] * inner.unsqueeze(2)
        denom = (td**2 + 3.0 * tc**2).unsqueeze(2).clamp(min=1e-10)
        t_j = (t0.unsqueeze(2) + td.unsqueeze(2) * numer / denom).clamp(min=1e-6)

    else:
        # General K: uniformly-spaced angles with uniform depth spacing (moment-matching Eq. 2 only holds for K=6)
        if K == 1:
            base  = torch.zeros(1, dtype=torch.float32, device=device)
            alpha = torch.tensor([0.5], dtype=torch.float32, device=device)
        else:
            base  = torch.linspace(0.0, 2.0 * math.pi * (1.0 - 1.0 / K), K,
                                   dtype=torch.float32, device=device)
            alpha = torch.linspace(0.0, 1.0, K, dtype=torch.float32, device=device)

        if perturb:
            phi    = torch.rand(n, k, 1, device=device) * 2.0 * math.pi
            angles = base[None, None, :].expand(n, k, K) + phi
        else:
            k_idx  = torch.arange(k, dtype=torch.float32, device=device)
            shift  = (k_idx % 2) * (math.pi / max(K, 1))
            angles = base[None, None, :].expand(n, k, K) + shift[None, :, None]

        t_j = (t0.unsqueeze(2) + alpha[None, None, :] *
               (t1 - t0).unsqueeze(2)).clamp(min=1e-6)

    # 3D transverse offsets
    cos_a  = torch.cos(angles)
    sin_a  = torch.sin(angles)
    x_perp = cone_radius * t_j * cos_a / sq2
    y_perp = cone_radius * t_j * sin_a / sq2

    if collapse:
        # Ablation: zero lateral offsets, removing the 2D spatial footprint while keeping sigma_j unchanged
        x_perp = torch.zeros_like(x_perp)
        y_perp = torch.zeros_like(y_perp)

    x_world = (o[:, None, None, :] +
               t_j[..., None] * d[:, None, None, :] +
               x_perp[..., None] * u[:, None, None, :] +
               y_perp[..., None] * v[:, None, None, :])   # (n, k, K, 3)

    sigma_j = cone_radius * t_j / sq2 * sigma_scale   # (n, k, K)
    return x_world, sigma_j


if HAS_WARP:
    # AABB intersection
    @wp.kernel
    def ray_aabb_intersection_kernel(
        origins: wp.array(dtype=wp.float32, ndim=2),
        dirs:    wp.array(dtype=wp.float32, ndim=2),
        aabb_min_x: float, aabb_min_y: float, aabb_min_z: float,
        aabb_max_x: float, aabb_max_y: float, aabb_max_z: float,
        t_near:      wp.array(dtype=wp.float32, ndim=1),
        t_far:       wp.array(dtype=wp.float32, ndim=1),
        valid_mask:  wp.array(dtype=wp.int8,    ndim=1),
    ):
        tid = wp.tid()
        ox = origins[tid, 0]; oy = origins[tid, 1]; oz = origins[tid, 2]
        dx = dirs[tid, 0] + 1e-8
        dy = dirs[tid, 1] + 1e-8
        dz = dirs[tid, 2] + 1e-8
        inv_dx = 1.0 / dx; inv_dy = 1.0 / dy; inv_dz = 1.0 / dz

        t0_x = (aabb_min_x - ox) * inv_dx; t1_x = (aabb_max_x - ox) * inv_dx
        t0_y = (aabb_min_y - oy) * inv_dy; t1_y = (aabb_max_y - oy) * inv_dy
        t0_z = (aabb_min_z - oz) * inv_dz; t1_z = (aabb_max_z - oz) * inv_dz

        near = wp.max(wp.max(wp.min(t0_x, t1_x), wp.min(t0_y, t1_y)), wp.min(t0_z, t1_z))
        far  = wp.min(wp.min(wp.max(t0_x, t1_x), wp.max(t0_y, t1_y)), wp.max(t0_z, t1_z))
        near = wp.max(near, 0.0)

        t_near[tid] = near
        t_far[tid]  = far
        if (near < far) and (far > 0.0):
            valid_mask[tid] = wp.int8(1)
        else:
            valid_mask[tid] = wp.int8(0)

    # DDA pass 1: count occupied steps per ray
    @wp.kernel
    def dda_count_kernel(
        origins:    wp.array(dtype=wp.float32, ndim=2),
        dirs:       wp.array(dtype=wp.float32, ndim=2),
        t_near:     wp.array(dtype=wp.float32, ndim=1),
        t_far:      wp.array(dtype=wp.float32, ndim=1),
        occ:        wp.array(dtype=wp.int8,    ndim=1),  # R^3 flattened
        grid_res:   int,
        aabb_min_x: float, aabb_min_y: float, aabb_min_z: float,
        aabb_max_x: float, aabb_max_y: float, aabb_max_z: float,
        dt:         float,
        jitter:     wp.array(dtype=wp.float32, ndim=1),  # per-ray in [0, dt)
        out_counts: wp.array(dtype=wp.int32,   ndim=1),
    ):
        tid = wp.tid()
        ox = origins[tid, 0]; oy = origins[tid, 1]; oz = origins[tid, 2]
        dx = dirs[tid, 0];    dy = dirs[tid, 1];    dz = dirs[tid, 2]

        R = grid_res
        inv_rx = float(R) / (aabb_max_x - aabb_min_x)
        inv_ry = float(R) / (aabb_max_y - aabb_min_y)
        inv_rz = float(R) / (aabb_max_z - aabb_min_z)

        t     = t_near[tid] + jitter[tid]
        t_end = t_far[tid]
        count = int(0)

        while t < t_end:
            px = ox + t * dx
            py = oy + t * dy
            pz = oz + t * dz
            ix = int(wp.clamp((px - aabb_min_x) * inv_rx, 0.0, float(R - 1)))
            iy = int(wp.clamp((py - aabb_min_y) * inv_ry, 0.0, float(R - 1)))
            iz = int(wp.clamp((pz - aabb_min_z) * inv_rz, 0.0, float(R - 1)))
            if occ[ix * R * R + iy * R + iz] != wp.int8(0):
                count = count + 1
            t = t + dt

        out_counts[tid] = count

    # DDA pass 2: write occupied t-values into padded output
    @wp.kernel
    def dda_write_kernel(
        origins:    wp.array(dtype=wp.float32, ndim=2),
        dirs:       wp.array(dtype=wp.float32, ndim=2),
        t_near:     wp.array(dtype=wp.float32, ndim=1),
        t_far:      wp.array(dtype=wp.float32, ndim=1),
        occ:        wp.array(dtype=wp.int8,    ndim=1),
        grid_res:   int,
        aabb_min_x: float, aabb_min_y: float, aabb_min_z: float,
        aabb_max_x: float, aabb_max_y: float, aabb_max_z: float,
        dt:         float,
        jitter:     wp.array(dtype=wp.float32, ndim=1),   # per-ray in [0, dt)
        out_z_vals: wp.array(dtype=wp.float32, ndim=2),   # (n_valid, max_k)
        max_k:      int,
    ):
        tid = wp.tid()
        ox = origins[tid, 0]; oy = origins[tid, 1]; oz = origins[tid, 2]
        dx = dirs[tid, 0];    dy = dirs[tid, 1];    dz = dirs[tid, 2]

        R = grid_res
        inv_rx = float(R) / (aabb_max_x - aabb_min_x)
        inv_ry = float(R) / (aabb_max_y - aabb_min_y)
        inv_rz = float(R) / (aabb_max_z - aabb_min_z)

        t     = t_near[tid] + jitter[tid]
        t_end = t_far[tid]
        j     = int(0)

        while t < t_end:
            if j >= max_k:
                break
            px = ox + t * dx
            py = oy + t * dy
            pz = oz + t * dz
            ix = int(wp.clamp((px - aabb_min_x) * inv_rx, 0.0, float(R - 1)))
            iy = int(wp.clamp((py - aabb_min_y) * inv_ry, 0.0, float(R - 1)))
            iz = int(wp.clamp((pz - aabb_min_z) * inv_rz, 0.0, float(R - 1)))
            if occ[ix * R * R + iy * R + iz] != wp.int8(0):
                out_z_vals[tid, j] = t
                j = j + 1
            t = t + dt


def render_rays(model, origins, dirs, aabb_min, aabb_max, background_color,
                num_samples=256, perturb=True, occupancy_grid=None,
                cone_radius=None, zip_mode=False, scene_type='synthetic',
                zip_collapse=False, zip_n_samples=6, zip_sigma_scale=1.0):
    N      = origins.shape[0]
    device = origins.device

    use_warp = HAS_WARP and type(getattr(model, 'encoder', None)).__name__ in (
        "HashEncodingWarp", "ZipHashEncodingWarp")
    use_dda  = use_warp and (occupancy_grid is not None)

    # World-space DDA step: _DDA_DT is in normalized [0,1] space; scale by longest side.
    dt_world = _DDA_DT * float((aabb_max - aabb_min).max())

    # AABB intersection
    profiler.start("ray_aabb")
    if use_warp:
        t_near_all     = torch.empty(N, dtype=torch.float32, device=device)
        t_far_all      = torch.empty(N, dtype=torch.float32, device=device)
        valid_mask_int = torch.empty(N, dtype=torch.int8,    device=device)
        wp.launch(
            kernel=ray_aabb_intersection_kernel, dim=N,
            inputs=[
                wp.from_torch(origins.contiguous()), wp.from_torch(dirs.contiguous()),
                float(aabb_min[0]), float(aabb_min[1]), float(aabb_min[2]),
                float(aabb_max[0]), float(aabb_max[1]), float(aabb_max[2]),
                wp.from_torch(t_near_all), wp.from_torch(t_far_all),
                wp.from_torch(valid_mask_int),
            ],
            device=Config.DEVICE,
        )
        valid_mask   = valid_mask_int.bool()
        t_near_valid = t_near_all[valid_mask]
        t_far_valid  = t_far_all[valid_mask]
    else:
        inv_dir      = 1.0 / (dirs + 1e-8)
        t0           = (aabb_min - origins) * inv_dir
        t1           = (aabb_max - origins) * inv_dir
        t_near_all   = torch.max(torch.minimum(t0, t1), dim=-1)[0].clamp(min=0.0)
        t_far_all    = torch.min(torch.maximum(t0, t1), dim=-1)[0]
        valid_mask   = (t_near_all < t_far_all) & (t_far_all > 0)
        t_near_valid = t_near_all[valid_mask]
        t_far_valid  = t_far_all[valid_mask]
    profiler.stop("ray_aabb")

    rgb_map = background_color.unsqueeze(0).expand(N, 3).clone()
    if valid_mask.sum() == 0:
        return rgb_map

    o_valid  = origins[valid_mask].contiguous()
    d_valid  = dirs[valid_mask].contiguous()
    n_valid  = o_valid.shape[0]

    # DDA ray marching (Warp + occupancy grid)
    if use_dda:
        profiler.start("ray_sampling")

        occ_flat = occupancy_grid.grid.reshape(-1).to(torch.int8).contiguous()
        R        = occupancy_grid.resolution
        ax, ay, az = float(aabb_min[0]), float(aabb_min[1]), float(aabb_min[2])
        bx, by, bz = float(aabb_max[0]), float(aabb_max[1]), float(aabb_max[2])

        # Shared jitter so count and write passes start from the same t (keeps valid_sample_mask consistent)
        jitter = (torch.rand(n_valid, dtype=torch.float32, device=device).mul_(dt_world)
                  if perturb else
                  torch.full((n_valid,), 0.5 * dt_world, dtype=torch.float32, device=device))

        # Pass 1: count occupied steps per ray
        counts = torch.zeros(n_valid, dtype=torch.int32, device=device)
        wp.launch(
            kernel=dda_count_kernel, dim=n_valid,
            inputs=[
                wp.from_torch(o_valid), wp.from_torch(d_valid),
                wp.from_torch(t_near_valid), wp.from_torch(t_far_valid),
                wp.from_torch(occ_flat), R,
                ax, ay, az, bx, by, bz, dt_world,
                wp.from_torch(jitter),
                wp.from_torch(counts),
            ],
            device=Config.DEVICE,
        )

        max_k = min(int(counts.max().item()), num_samples)
        if zip_mode and cone_radius is not None:
            # Cap max_k at 3x the running average so a single outlier ray can't blow up memory
            avg_k = max(int(rendering_stats["avg_samples_per_ray"]) + 1, 1)
            max_k = min(max_k, max(avg_k * 3, 64))
        if max_k == 0:
            profiler.stop("ray_sampling")
            return rgb_map

        z_vals = torch.zeros(n_valid, max_k, dtype=torch.float32, device=device)
        wp.launch(
            kernel=dda_write_kernel, dim=n_valid,
            inputs=[
                wp.from_torch(o_valid), wp.from_torch(d_valid),
                wp.from_torch(t_near_valid), wp.from_torch(t_far_valid),
                wp.from_torch(occ_flat), R,
                ax, ay, az, bx, by, bz, dt_world,
                wp.from_torch(jitter), wp.from_torch(z_vals), max_k,
            ],
            device=Config.DEVICE,
        )
        profiler.stop("ray_sampling")

        # Mask: True only for slots that hold an actual DDA sample
        k_idx            = torch.arange(max_k, device=device).unsqueeze(0)
        valid_sample_mask = k_idx < counts.clamp(max=max_k).unsqueeze(1)  # (n_valid, max_k)

        # MLP forward (all DDA samples are occupied by construction)
        profiler.start("occ_query")   # stage kept for profiler symmetry; no actual query
        profiler.stop("occ_query")

        dirs_expand = d_valid.unsqueeze(1).expand(-1, max_k, -1)
        dirs_flat   = dirs_expand.reshape(-1, 3)
        valid_flat  = valid_sample_mask.reshape(-1)
        n_mlp       = int(valid_flat.sum().item())

        rendering_stats["n_mlp_evals"]        = n_mlp
        rendering_stats["n_valid_rays"]        = n_valid
        rendering_stats["avg_samples_per_ray"] = n_mlp / max(n_valid, 1)

        if n_mlp == 0:
            return rgb_map

        profiler.start("mlp_forward")
        if zip_mode and cone_radius is not None:
            # Zip-NeRF: replace each DDA point with zip_n_samples frustum samples
            t0 = (z_vals - 0.5 * dt_world).clamp(min=t_near_valid.unsqueeze(1))
            t1 = (z_vals + 0.5 * dt_world).clamp(max=t_far_valid.unsqueeze(1))
            x_multi_world, sigma_j_enc = compute_hexagonal_multisamples(
                o_valid, d_valid, t0, t1, cone_radius, perturb,
                n_samples=zip_n_samples, collapse=zip_collapse,
                sigma_scale=zip_sigma_scale,
            )   # (n_valid, max_k, zip_n_samples, 3), (n_valid, max_k, zip_n_samples)

            if scene_type == '360':
                # Jacobian scaling for spatial contraction R³ → [-2,2]³: local Gaussian scale changes by |det(J_C)|^(1/3)
                norm_world = torch.norm(x_multi_world, dim=-1, keepdim=False)  # (n_valid, max_k, 6)
                norm_clamped = torch.clamp(norm_world, min=1.0)
                # |det(J_C)|^(1/3) = (∛(2‖x‖-1) / ‖x‖)² — paper supplement Eq. 14
                jacobian_factor = torch.exp(
                    (2.0 / 3.0) * torch.log(2.0 * norm_clamped - 1.0 + 1e-8)
                    - 2.0 * torch.log(norm_clamped)
                )
                # Maps world-space σ to contracted [-2,2]³ σ, then /4 for normalize_contracted's [-2,2]³ → [0,1]³
                sigma_j_enc = sigma_j_enc * jacobian_factor / 4.0

                x_multi_norm = normalize_contracted(
                    contract_mip360(x_multi_world.reshape(-1, 3))
                ).reshape(n_valid, max_k, zip_n_samples, 3).clamp(0.0, 1.0)
            else:
                # Synthetic (Blender): no contraction, direct unit cube normalization.
                x_multi_norm = contract_to_unit_cube(
                    x_multi_world.reshape(-1, 3), aabb_min, aabb_max
                ).reshape(n_valid, max_k, zip_n_samples, 3).clamp(0.0, 1.0)
                # Normalize σ to [0,1] grid space (hash grid resolution is cells-per-unit);
                # without this world-space σ collapses ω at fine levels and Zip drops to ~25 dB.
                world_scale = (aabb_max - aabb_min).max()
                sigma_j_enc = sigma_j_enc / world_scale

            x_multi_flat  = x_multi_norm.reshape(n_valid * max_k, zip_n_samples, 3)
            sigma_flat_enc = sigma_j_enc.reshape(n_valid * max_k, zip_n_samples)
            rgb_occ, sigma_occ = model(
                x_multi_flat[valid_flat], sigma_flat_enc[valid_flat], dirs_flat[valid_flat]
            )
        else:
            pts_world = o_valid.unsqueeze(1) + z_vals.unsqueeze(-1) * d_valid.unsqueeze(1)
            if scene_type == '360':
                pts_norm = normalize_contracted(contract_mip360(pts_world.reshape(-1, 3)))
            else:
                pts_norm = contract_to_unit_cube(pts_world.reshape(-1, 3), aabb_min, aabb_max)
            pts_query = pts_norm.clamp(0.0, 1.0)
            rgb_occ, sigma_occ = model(pts_query[valid_flat], dirs_flat[valid_flat])
        profiler.stop("mlp_forward")

        rgb_flat            = torch.zeros(n_valid * max_k, 3, device=device)
        sigma_flat          = torch.zeros(n_valid * max_k,    device=device)
        rgb_flat[valid_flat]   = rgb_occ.float()
        sigma_flat[valid_flat] = sigma_occ.float()

        rgb   = rgb_flat.reshape(n_valid, max_k, 3)
        sigma = sigma_flat.reshape(n_valid, max_k)

        # Volume rendering with constant-dt deltas (paper E.1: fixed step size)
        profiler.start("volume_rendering")
        sigma = sigma * valid_sample_mask.float()
        dists = valid_sample_mask.float() * dt_world

    else:
        # Stratified linspace sampling (PyTorch fallback)
        profiler.start("ray_sampling")
        t_vals = torch.linspace(0.0, 1.0, num_samples, device=device)
        z_vals = t_near_valid.unsqueeze(-1) * (1.0 - t_vals) + t_far_valid.unsqueeze(-1) * t_vals
        mids   = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper  = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower  = torch.cat([z_vals[..., :1], mids],  dim=-1)
        t_rand = torch.rand_like(z_vals) if perturb else torch.full_like(z_vals, 0.5)
        z_vals = lower + (upper - lower) * t_rand

        pts_world   = o_valid.unsqueeze(1) + z_vals.unsqueeze(-1) * d_valid.unsqueeze(1)
        dirs_expand = d_valid.unsqueeze(1).expand(-1, num_samples, -1)
        if scene_type == '360':
            pts_norm = normalize_contracted(contract_mip360(pts_world.reshape(-1, 3)))
        else:
            pts_norm = contract_to_unit_cube(pts_world.reshape(-1, 3), aabb_min, aabb_max)
        valid_pts_mask = torch.all((pts_norm >= 0.0) & (pts_norm <= 1.0), dim=-1).reshape(n_valid, num_samples)
        pts_query   = pts_norm.clamp(0.0, 1.0)
        dirs_flat   = dirs_expand.reshape(-1, 3)
        profiler.stop("ray_sampling")

        N_pts = pts_query.shape[0]
        if occupancy_grid is not None:
            profiler.start("occ_query")
            occ_mask = occupancy_grid.query(pts_query)
            if not occ_mask.any():
                occ_mask = torch.ones(N_pts, dtype=torch.bool, device=device)
            profiler.stop("occ_query")

            n_mlp = int(occ_mask.sum().item())
            rendering_stats["n_mlp_evals"]        = n_mlp
            rendering_stats["n_valid_rays"]        = n_valid
            rendering_stats["avg_samples_per_ray"] = n_mlp / max(n_valid, 1)

            profiler.start("mlp_forward")
            rgb_occ, sigma_occ = model(pts_query[occ_mask], dirs_flat[occ_mask])
            profiler.stop("mlp_forward")

            rgb_flat            = torch.zeros(N_pts, 3, device=device)
            sigma_flat          = torch.zeros(N_pts,    device=device)
            rgb_flat[occ_mask]   = rgb_occ.float()
            sigma_flat[occ_mask] = sigma_occ.float()
        else:
            rendering_stats["n_mlp_evals"]        = N_pts
            rendering_stats["n_valid_rays"]        = n_valid
            rendering_stats["avg_samples_per_ray"] = num_samples

            profiler.start("mlp_forward")
            rgb_flat, sigma_flat = model(pts_query, dirs_flat)
            rgb_flat   = rgb_flat.float()
            sigma_flat = sigma_flat.float()
            profiler.stop("mlp_forward")

        rgb   = rgb_flat.reshape(n_valid, num_samples, 3)
        sigma = sigma_flat.reshape(n_valid, num_samples)
        valid_sample_mask = valid_pts_mask
        max_k = num_samples

        profiler.start("volume_rendering")
        sigma = sigma * valid_sample_mask.float()
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, (t_far_valid - z_vals[..., -1]).unsqueeze(-1)], dim=-1)

    # Volume rendering (shared)
    # Explicit float32 for exp/transmittance to prevent underflow
    sigma_f32 = sigma.to(torch.float32) if sigma.dtype != torch.float32 else sigma
    dists_f32 = dists.to(torch.float32) if dists.dtype != torch.float32 else dists

    alpha = 1.0 - torch.exp(-sigma_f32 * dists_f32)

    # Transmittance in float32 (cumprod precision critical)
    transmittance = torch.cumprod(
        torch.cat([torch.ones(n_valid, 1, device=device, dtype=torch.float32),
                   1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[:, :-1]

    # Weights stay in float32 through final composition
    weights = alpha * transmittance * (transmittance > 1e-4).float()
    rgb_rendered = torch.sum(weights.unsqueeze(-1) * rgb.to(torch.float32), dim=1)
    acc_map = torch.sum(weights, dim=-1)
    rgb_map[valid_mask] = rgb_rendered + background_color * (1.0 - acc_map.unsqueeze(-1))
    profiler.stop("volume_rendering")

    return rgb_map
