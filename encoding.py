import math
import torch
import torch.nn as nn
import numpy as np
from config import Config
try:
    import warp as wp
    HAS_WARP = True
except Exception:
    wp = None
    HAS_WARP = False

class HashEncodingPyTorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.L = config.L
        self.F = config.F
        self.T = config.T
        b = math.exp((math.log(config.N_MAX) - math.log(config.N_MIN)) / (config.L - 1))
        self.resolutions = [math.floor(config.N_MIN * (b ** i)) for i in range(config.L)]
        self.embeddings = nn.Parameter(torch.zeros(self.L, self.T, self.F))
        nn.init.uniform_(self.embeddings, -1e-4, 1e-4)

        # int32 primes match the Warp kernel so both backends hash identically
        self.register_buffer('primes', torch.tensor(
            [config.PRIME_1, config.PRIME_2, config.PRIME_3], dtype=torch.int32
        ))
        self.register_buffer('offsets', torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ], dtype=torch.long))

    def forward(self, x):
        # Hash table stored at fp16 for L2 cache efficiency (paper Sec 4); fp32 master kept for Adam
        emb_half = self.embeddings.half()
        encoded_features = []

        for i, res in enumerate(self.resolutions):
            x_scaled = x * res
            x_floor = x_scaled.long()
            weights = x_scaled - x_floor.float()
            fx, fy, fz = weights[:, 0], weights[:, 1], weights[:, 2]

            c = x_floor.unsqueeze(1) + self.offsets.unsqueeze(0)

            # int32 multiplication wraps identically to the Warp kernel (paper Eq. 4).
            c_i32 = c.to(torch.int32)
            # Coarse levels where (N_l+1)^3 <= T get a direct 1:1 mapping, avoiding hash collisions (paper Sec 3)
            if (res + 1) ** 3 <= self.T:
                r1 = res + 1
                cx = c_i32[..., 0].to(torch.int64).clamp(0, res)
                cy = c_i32[..., 1].to(torch.int64).clamp(0, res)
                cz = c_i32[..., 2].to(torch.int64).clamp(0, res)
                h = cx * r1 * r1 + cy * r1 + cz
            else:
                h = ((c_i32[..., 0] * self.primes[0]) ^
                     (c_i32[..., 1] * self.primes[1]) ^
                     (c_i32[..., 2] * self.primes[2])).to(torch.int64) & (self.T - 1)

            # Lookup in fp16, cast to fp32 for interpolation computation.
            corners = emb_half[i][h].float()

            c00 = corners[:, 0] * (1-fx).unsqueeze(-1) + corners[:, 4] * fx.unsqueeze(-1)
            c01 = corners[:, 1] * (1-fx).unsqueeze(-1) + corners[:, 5] * fx.unsqueeze(-1)
            c10 = corners[:, 2] * (1-fx).unsqueeze(-1) + corners[:, 6] * fx.unsqueeze(-1)
            c11 = corners[:, 3] * (1-fx).unsqueeze(-1) + corners[:, 7] * fx.unsqueeze(-1)
            c0 = c00 * (1-fy).unsqueeze(-1) + c10 * fy.unsqueeze(-1)
            c1 = c01 * (1-fy).unsqueeze(-1) + c11 * fy.unsqueeze(-1)
            c = c0 * (1-fz).unsqueeze(-1) + c1 * fz.unsqueeze(-1)

            encoded_features.append(c)

        return torch.cat(encoded_features, dim=-1)


# Zip-NeRF helpers

def erf_approx(x):
    """Fast approximation of erf — Zip-NeRF supplement Eq. 11."""
    return torch.sign(x) * torch.sqrt(
        torch.clamp(1.0 - torch.exp(-(4.0 / math.pi) * x * x), min=0.0)
    )


class ZipHashEncodingPyTorch(nn.Module):
    """Hash encoding with Zip-NeRF multisampling + downweighting (Eq. 4)."""

    def __init__(self, config):
        super().__init__()
        self.L = config.L
        self.F = config.F
        self.T = config.T
        b = math.exp((math.log(config.N_MAX) - math.log(config.N_MIN)) / (config.L - 1))
        self.resolutions = [math.floor(config.N_MIN * (b ** i)) for i in range(config.L)]
        self.embeddings = nn.Parameter(torch.zeros(self.L, self.T, self.F))
        nn.init.uniform_(self.embeddings, -1e-4, 1e-4)
        self._mean_v2_cache = None
        # Ablation: set True by train.py to fix omega=1 (no scale-aware attenuation)
        self.no_downweighting = False
        self.register_buffer('primes', torch.tensor(
            [config.PRIME_1, config.PRIME_2, config.PRIME_3], dtype=torch.int32
        ))
        self.register_buffer('offsets', torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ], dtype=torch.long))

    def _trilerp(self, x_flat, res):
        """Trilinear interpolation for a batch of points at one resolution level."""
        emb_half = self.embeddings.half()
        i = self.resolutions.index(res)
        x_scaled = x_flat * float(res)
        x_floor = x_scaled.long()
        w = x_scaled - x_floor.float()
        fx, fy, fz = w[:, 0], w[:, 1], w[:, 2]
        c = x_floor.unsqueeze(1) + self.offsets.unsqueeze(0)
        c_i32 = c.to(torch.int32)
        if (res + 1) ** 3 <= self.T:
            r1 = res + 1
            cx = c_i32[..., 0].to(torch.int64).clamp(0, res)
            cy = c_i32[..., 1].to(torch.int64).clamp(0, res)
            cz = c_i32[..., 2].to(torch.int64).clamp(0, res)
            h = cx * r1 * r1 + cy * r1 + cz
        else:
            h = ((c_i32[..., 0] * self.primes[0]) ^
                 (c_i32[..., 1] * self.primes[1]) ^
                 (c_i32[..., 2] * self.primes[2])).to(torch.int64) & (self.T - 1)
        corners = emb_half[i][h].float()
        c00 = corners[:, 0] * (1-fx).unsqueeze(-1) + corners[:, 4] * fx.unsqueeze(-1)
        c01 = corners[:, 1] * (1-fx).unsqueeze(-1) + corners[:, 5] * fx.unsqueeze(-1)
        c10 = corners[:, 2] * (1-fx).unsqueeze(-1) + corners[:, 6] * fx.unsqueeze(-1)
        c11 = corners[:, 3] * (1-fx).unsqueeze(-1) + corners[:, 7] * fx.unsqueeze(-1)
        c0 = c00 * (1-fy).unsqueeze(-1) + c10 * fy.unsqueeze(-1)
        c1 = c01 * (1-fy).unsqueeze(-1) + c11 * fy.unsqueeze(-1)
        return c0 * (1-fz).unsqueeze(-1) + c1 * fz.unsqueeze(-1)

    def forward_single(self, x):
        """Single-point lookup — used by occupancy grid (no multisampling)."""
        return torch.cat([self._trilerp(x, res) for res in self.resolutions], dim=-1)

    def _get_mean_v2(self):
        """Cached mean(V_l^2) per level. Recomputed each training step, frozen during eval."""
        if torch.is_grad_enabled() or self._mean_v2_cache is None:
            self._mean_v2_cache = self.embeddings.pow(2).mean(dim=(1, 2)).detach()
        return self._mean_v2_cache

    def forward(self, x_multi, sigma_j):
        """
        x_multi: (N, 6, 3)  multisample positions in [0,1]
        sigma_j: (N, 6)     isotropic Gaussian scale per sample
        Returns: (N, L*F + L)  spatial features + scale featurization (Appendix C)
        """
        N, K, _ = x_multi.shape
        x_flat = x_multi.reshape(N * K, 3)
        encoded_features = []
        scale_features = []

        mean_v2 = self._get_mean_v2()  # (L,) — cached during eval, fresh each train step

        for i, res in enumerate(self.resolutions):
            n_l = float(res)
            feat = self._trilerp(x_flat, res).reshape(N, K, self.F)

            # Downweighting ω_{j,l} = erf(1/sqrt(8 σ_j² n_l²))
            arg = 1.0 / torch.sqrt(8.0 * sigma_j.pow(2) * (n_l * n_l) + 1e-10)
            omega = erf_approx(arg).clamp(0.0, 1.0)  # (N, K)
            if self.no_downweighting:
                omega = torch.ones_like(omega)
            encoded_features.append((omega.unsqueeze(-1) * feat).mean(dim=1))  # (N, F)

            # Scale featurization (Appendix C Eq. 10):
            # (2·mean_j(ω_{j,l}) - 1) · sqrt(V_init² + stop_grad(mean(V_l²)))
            mean_omega = omega.mean(dim=1)  # (N,)
            s = (2.0 * mean_omega - 1.0) * torch.sqrt(1e-8 + mean_v2[i])
            scale_features.append(s.unsqueeze(-1))  # (N, 1)

        spatial = torch.cat(encoded_features, dim=-1)   # (N, L*F)
        scales = torch.cat(scale_features, dim=-1)      # (N, L)
        return torch.cat([spatial, scales], dim=-1)     # (N, L*F + L)


if HAS_WARP:
    wp.init()

    # Embeddings stored at fp16 (paper Sec 4); forward accumulates in fp32, backward writes fp32 gradients
    @wp.kernel
    def hash_grid_forward_kernel(
        inputs: wp.array(dtype=wp.float32, ndim=2),
        embeddings: wp.array(dtype=wp.float16, ndim=3),  # fp16 storage
        resolutions: wp.array(dtype=wp.int32),
        primes: wp.array(dtype=wp.int32),
        L: int, T: int, F: int, max_r1: int,
        output: wp.array(dtype=wp.float32, ndim=2)
    ):
        tid = wp.tid()
        x_in, y_in, z_in = inputs[tid, 0], inputs[tid, 1], inputs[tid, 2]

        for l in range(L):
            res = float(resolutions[l])
            x_s, y_s, z_s = x_in * res, y_in * res, z_in * res
            x0, y0, z0 = int(wp.floor(x_s)), int(wp.floor(y_s)), int(wp.floor(z_s))
            wx, wy, wz = x_s - float(x0), y_s - float(y0), z_s - float(z0)

            feat_val = wp.vec2(0.0, 0.0)

            for i in range(8):
                dx = (i >> 2) & 1
                dy = (i >> 1) & 1
                dz = i & 1

                cx = x0 + dx
                cy = y0 + dy
                cz = z0 + dz

                # Direct mapping for coarse levels (paper Sec 3); max_r1 avoids int32 overflow of r1^3 at fine levels
                r1 = resolutions[l] + 1
                if r1 <= max_r1:
                    h_idx = cx * r1 * r1 + cy * r1 + cz
                else:
                    h_idx = ((cx * primes[0]) ^ (cy * primes[1]) ^ (cz * primes[2])) & (T - 1)

                weight = (1.0 - wx if dx == 0 else wx) * \
                         (1.0 - wy if dy == 0 else wy) * \
                         (1.0 - wz if dz == 0 else wz)

                # Cast fp16 → fp32 before accumulation to avoid precision loss.
                feat_val[0] += wp.float32(embeddings[l, h_idx, 0]) * weight
                feat_val[1] += wp.float32(embeddings[l, h_idx, 1]) * weight

            output[tid, l * F + 0] = feat_val[0]
            output[tid, l * F + 1] = feat_val[1]

    @wp.kernel
    def hash_grid_backward_kernel(
        grad_output: wp.array(dtype=wp.float32, ndim=2),
        inputs: wp.array(dtype=wp.float32, ndim=2),
        resolutions: wp.array(dtype=wp.int32),
        primes: wp.array(dtype=wp.int32),
        L: int, T: int, F: int, max_r1: int,
        grad_embeddings: wp.array(dtype=wp.float32, ndim=3)
    ):
        tid = wp.tid()
        x_in, y_in, z_in = inputs[tid, 0], inputs[tid, 1], inputs[tid, 2]
        
        for l in range(L):
            res = float(resolutions[l])
            x_s, y_s, z_s = x_in * res, y_in * res, z_in * res
            x0, y0, z0 = int(wp.floor(x_s)), int(wp.floor(y_s)), int(wp.floor(z_s))
            wx, wy, wz = x_s - float(x0), y_s - float(y0), z_s - float(z0)
            
            g0 = grad_output[tid, l*F + 0]
            g1 = grad_output[tid, l*F + 1]
            
            for i in range(8):
                dx = (i >> 2) & 1
                dy = (i >> 1) & 1
                dz = i & 1
                
                cx = x0 + dx
                cy = y0 + dy
                cz = z0 + dz
                
                r1 = resolutions[l] + 1
                if r1 <= max_r1:
                    h_idx = cx * r1 * r1 + cy * r1 + cz
                else:
                    h_idx = ((cx * primes[0]) ^ (cy * primes[1]) ^ (cz * primes[2])) & (T - 1)

                weight = (1.0 - wx if dx == 0 else wx) * \
                         (1.0 - wy if dy == 0 else wy) * \
                         (1.0 - wz if dz == 0 else wz)

                wp.atomic_add(grad_embeddings, l, h_idx, 0, g0 * weight)
                wp.atomic_add(grad_embeddings, l, h_idx, 1, g1 * weight)

    class WarpHashFunc(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs, embeddings, resolutions, primes, config):
            ctx.config = config
            # Save fp32 master for backward (gradient stays fp32 for Adam).
            ctx.save_for_backward(inputs, embeddings, resolutions, primes)
            N = inputs.shape[0]
            L, T, F = embeddings.shape
            encoded = torch.empty((N, L * F), device=inputs.device, dtype=torch.float32)

            # max_r1 = floor(T^(1/3)), the largest r1=(N_l+1) with (N_l+1)^3<=T; avoids int32 overflow of r1^3
            max_r1 = int(T ** (1.0 / 3.0))
            while (max_r1 + 1) ** 3 <= T:
                max_r1 += 1
            wp.launch(kernel=hash_grid_forward_kernel, dim=N, inputs=[
                wp.from_torch(inputs),
                wp.from_torch(embeddings.half()),  # fp16 for L2 cache efficiency
                wp.from_torch(resolutions),
                wp.from_torch(primes), L, T, F, max_r1,
                wp.from_torch(encoded),
            ], device=config.DEVICE)
            return encoded

        @staticmethod
        def backward(ctx, grad_output):
            inputs, embeddings, resolutions, primes = ctx.saved_tensors
            L, T, F = embeddings.shape
            N = inputs.shape[0]
            # Gradient accumulates into fp32 buffer → Adam sees fp32 gradients.
            grad_embeddings = torch.zeros_like(embeddings)

            max_r1 = int(T ** (1.0 / 3.0))
            while (max_r1 + 1) ** 3 <= T:
                max_r1 += 1
            wp.launch(kernel=hash_grid_backward_kernel, dim=N, inputs=[
                wp.from_torch(grad_output.contiguous()), wp.from_torch(inputs),
                wp.from_torch(resolutions), wp.from_torch(primes),
                L, T, F, max_r1, wp.from_torch(grad_embeddings),
            ], device=Config.DEVICE)
            return None, grad_embeddings, None, None, None

    class HashEncodingWarp(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.L, self.T, self.F = config.L, config.T, config.F
            b = math.exp((math.log(config.N_MAX) - math.log(config.N_MIN)) / (config.L - 1))
            resolutions_list = [math.floor(config.N_MIN * (b ** i)) for i in range(config.L)]
            self.register_buffer('resolutions', torch.tensor(resolutions_list, dtype=torch.int32))
            self.register_buffer('primes', torch.from_numpy(
                np.array([config.PRIME_1, config.PRIME_2, config.PRIME_3]).astype(np.int32)
            ))
            self.embeddings = nn.Parameter(torch.zeros(self.L, self.T, self.F))
            nn.init.uniform_(self.embeddings, -1e-4, 1e-4)

        def forward(self, x):
            return WarpHashFunc.apply(x, self.embeddings, self.resolutions, self.primes, self.config)

    class ZipHashEncodingWarp(nn.Module):
        """Warp-accelerated hash encoding with Zip-NeRF multisampling + downweighting."""

        def __init__(self, config):
            super().__init__()
            self.config = config
            self.L, self.T, self.F = config.L, config.T, config.F
            b = math.exp((math.log(config.N_MAX) - math.log(config.N_MIN)) / (config.L - 1))
            resolutions_list = [math.floor(config.N_MIN * (b ** i)) for i in range(config.L)]
            self.resolutions_list = resolutions_list
            self.register_buffer('resolutions', torch.tensor(resolutions_list, dtype=torch.int32))
            self.register_buffer('primes', torch.from_numpy(
                np.array([config.PRIME_1, config.PRIME_2, config.PRIME_3]).astype(np.int32)
            ))
            self.embeddings = nn.Parameter(torch.zeros(self.L, self.T, self.F))
            nn.init.uniform_(self.embeddings, -1e-4, 1e-4)
            self._mean_v2_cache = None
            # Ablation: set True by train.py to fix omega=1 (no scale-aware attenuation)
            self.no_downweighting = False

        def _get_mean_v2(self):
            """Cached mean(V_l^2) per level. Recomputed each training step, frozen during eval."""
            if torch.is_grad_enabled() or self._mean_v2_cache is None:
                self._mean_v2_cache = self.embeddings.pow(2).mean(dim=(1, 2)).detach()
            return self._mean_v2_cache

        def forward_single(self, x):
            """Single-point lookup for occupancy grid queries (no multisampling)."""
            return WarpHashFunc.apply(x, self.embeddings, self.resolutions, self.primes, self.config)

        def forward(self, x_multi, sigma_j):
            """
            x_multi: (N, 6, 3)  multisample positions in [0,1]
            sigma_j: (N, 6)     isotropic Gaussian scale per sample
            Returns: (N, L*F + L)  spatial features + scale featurization (Appendix C)
            """
            N, K, _ = x_multi.shape
            x_flat = x_multi.reshape(N * K, 3).contiguous()

            # One kernel call for all N*K sample positions
            feats_flat = WarpHashFunc.apply(
                x_flat, self.embeddings, self.resolutions, self.primes, self.config
            )
            feats = feats_flat.reshape(N, K, self.L, self.F)   # (N, K, L, F)

            # erf_approx in-place: omega = sqrt(max(0, 1 - exp(-4/π · arg²)))
            n_l_t  = torch.tensor(self.resolutions_list, dtype=torch.float32, device=x_multi.device)
            arg_sq = 1.0 / (8.0 * sigma_j.unsqueeze(-1).pow(2)
                            * n_l_t[None, None, :].pow(2) + 1e-10)   # (N, K, L)
            arg_sq.mul_(-4.0 / math.pi)
            torch.exp_(arg_sq)
            omega = (1.0 - arg_sq).clamp_(min=0.0)
            del arg_sq
            omega.sqrt_()
            if self.no_downweighting:
                omega = torch.ones_like(omega)

            feats_weighted = (omega.unsqueeze(-1) * feats).mean(dim=1)
            spatial = feats_weighted.reshape(N, self.L * self.F)

            mean_omega = omega.mean(dim=1)           # (N, L)
            mean_v2    = self._get_mean_v2()
            scales = (2.0 * mean_omega - 1.0) * torch.sqrt(1e-8 + mean_v2[None, :])

            return torch.cat([spatial, scales], dim=-1)
