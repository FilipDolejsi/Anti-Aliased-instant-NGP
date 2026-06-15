import math

class Config:
    # Data Paths
    DATA_ROOT = "../instant-ngp/data/lego"
    
    # Hash Encoding (Paper Section 3)
    L = 16              # Number of levels
    F = 2               # Feature dimension per level
    
    T = 2**19           # Hash table size, paper default (sized for GPU L2 cache)
    
    N_MIN = 16          # Coarsest resolution
    N_MAX = 2048        # Finest resolution (Sec 5.4: "set to 2048 ... for NeRF")
    
    # Architecture
    HIDDEN_DIM_DENSITY = 64
    HIDDEN_DIM_COLOR = 64
    
    # Hashing Primes (Eq. 4)
    PRIME_1 = 1
    PRIME_2 = 2654435761
    PRIME_3 = 805459861

    # Training (Paper Section 4 & 5.4 & Appendix E.3)
    SAMPLE_BUDGET = 2**18   # fixed MLP evals/step (paper's "256Ki batch"); ray count scales with occupancy in Warp+DDA mode
    BATCH_SIZE = 4096       # fallback ray count for PyTorch/linspace mode
    LR = 1e-2
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.99
    ADAM_EPS = 1e-15

    ITERATIONS = 35000
    VAL_INTERVAL = 2000

    # Rendering & Transparency Handling
    N_SAMPLES = 1024    # samples per ray (paper Appendix E.1: step size = sqrt(3)/1024)
    
    RANDOM_BG_TRAIN = True   # random bg prevents floaters (iNGP default; see NVlabs/instant-ngp discussion #192)
    
    # Scene Bounds
    AABB_MIN = [-1.3, -1.3, -1.3]
    AABB_MAX = [ 1.3,  1.3,  1.3]
    
    # Toggle Backend
    USE_WARP = False    

    DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
