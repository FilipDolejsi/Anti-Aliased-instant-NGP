"""Optional imports (Warp, matplotlib, scikit-image, imageio) with feature flags."""
import importlib
import sys

try:
    import warp as wp
    HAS_WARP = True
except Exception:
    wp = None
    HAS_WARP = False
    print("Warning: NVIDIA Warp not found. 'HashEncodingWarp' and other Warp kernels will be unavailable.")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False
    print("Warning: 'matplotlib' not found. Graphs will not be generated. Install via `pip install matplotlib`.")

try:
    from skimage.metrics import structural_similarity as compute_ssim
    HAS_SSIM = True
except Exception:
    compute_ssim = None
    HAS_SSIM = False
    print("Warning: 'scikit-image' not found. SSIM will not be calculated. Install via `pip install scikit-image`.")

try:
    import imageio
    HAS_IMAGEIO = True
except Exception:
    imageio = None
    HAS_IMAGEIO = False
    print("Warning: 'imageio' not found. Image reading/writing will fail without it. Install via `pip install imageio`.")

__all__ = [
    "wp", "HAS_WARP",
    "plt", "HAS_MATPLOTLIB",
    "compute_ssim", "HAS_SSIM",
    "imageio", "HAS_IMAGEIO"
]
