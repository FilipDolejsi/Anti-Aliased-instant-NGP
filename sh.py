import torch
import torch.nn as nn
from config import Config
try:
    import warp as wp
    HAS_WARP = True
except Exception:
    wp = None
    HAS_WARP = False

class SphericalHarmonics(nn.Module):
    def __init__(self, degree=4):
        super().__init__()
        self.degree = degree
        self.dim = degree ** 2

    def forward(self, d):
        x, y, z = d[..., 0], d[..., 1], d[..., 2]
        
        # Level 0 (1 coeff)
        sh = [torch.zeros_like(x) + 0.28209479177387814]
        
        if self.degree > 1:
            # Level 1 (3 coeffs)
            sh.extend([
                -0.4886025119029199 * y, 
                0.4886025119029199 * z, 
                -0.4886025119029199 * x
            ])
            
        if self.degree > 2:
            # Level 2 (5 coeffs)
            xx, yy, zz = x*x, y*y, z*z
            xy, yz, xz = x*y, y*z, x*z
            sh.extend([
                1.0925484305920792 * xy, 
                -1.0925484305920792 * yz,
                0.31539156525252005 * (2.0 * zz - xx - yy), 
                -1.0925484305920792 * xz,
                0.5462742152960396 * (xx - yy)
            ])
            
        if self.degree > 3:
            # Level 3 (7 coeffs)
            C0 = 0.5900435899266435
            C1 = 2.890611442640554
            C2 = 0.4570457994644658
            C3 = 0.3731763325901154
            C4 = 1.445305721320277
            
            xx, yy, zz = x*x, y*y, z*z
            
            sh.extend([
                -C0 * y * (3 * xx - yy),
                C1 * x * y * z,
                -C2 * y * (4 * zz - xx - yy),
                C3 * z * (2 * zz - 3 * xx - 3 * yy),
                C2 * x * (4 * zz - xx - yy),
                C4 * z * (xx - yy),
                -C0 * x * (xx - 3 * yy)
            ])

        return torch.stack(sh, dim=-1)


if HAS_WARP:
    wp.init()

    @wp.kernel
    def sh_forward_kernel(
        dirs: wp.array(dtype=wp.float32, ndim=2),
        out: wp.array(dtype=wp.float32, ndim=2)
    ):
        tid = wp.tid()
        x = dirs[tid, 0]
        y = dirs[tid, 1]
        z = dirs[tid, 2]
        
        xx = x*x
        yy = y*y
        zz = z*z
        xy = x*y
        yz = y*z
        xz = x*z

        # Level 0
        out[tid, 0] = 0.28209479177387814
        
        # Level 1
        out[tid, 1] = -0.4886025119029199 * y
        out[tid, 2] = 0.4886025119029199 * z
        out[tid, 3] = -0.4886025119029199 * x
        
        # Level 2
        out[tid, 4] = 1.0925484305920792 * xy
        out[tid, 5] = -1.0925484305920792 * yz
        out[tid, 6] = 0.31539156525252005 * (2.0 * zz - xx - yy)
        out[tid, 7] = -1.0925484305920792 * xz
        out[tid, 8] = 0.5462742152960396 * (xx - yy)
        
        # Level 3
        C0 = 0.5900435899266435
        C1 = 2.890611442640554
        C2 = 0.4570457994644658
        C3 = 0.3731763325901154
        C4 = 1.445305721320277
        
        out[tid, 9] = -C0 * y * (3.0 * xx - yy)
        out[tid, 10] = C1 * xy * z
        out[tid, 11] = -C2 * y * (4.0 * zz - xx - yy)
        out[tid, 12] = C3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
        out[tid, 13] = C2 * x * (4.0 * zz - xx - yy)
        out[tid, 14] = C4 * z * (xx - yy)
        out[tid, 15] = -C0 * x * (xx - 3.0 * yy)

    class WarpSHFunc(torch.autograd.Function):
        @staticmethod
        def forward(ctx, dirs):
            N = dirs.shape[0]
            out = torch.empty((N, 16), device=dirs.device, dtype=torch.float32)
            wp.launch(kernel=sh_forward_kernel, dim=N, inputs=[wp.from_torch(dirs), wp.from_torch(out)], device=Config.DEVICE)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            return None

    class SphericalHarmonicsWarp(nn.Module):
        def __init__(self, degree=4):
            super().__init__()
            self.degree = degree
            self.dim = degree ** 2

        def forward(self, d):
            return WarpSHFunc.apply(d)
