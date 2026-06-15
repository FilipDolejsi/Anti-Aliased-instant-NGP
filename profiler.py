import torch
from collections import defaultdict

_STAGE_ORDER = [
    "data_loading",
    "ray_aabb",
    "ray_sampling",   # Warp path: march+occ merged here; PyTorch path: sampling only
    "occ_query",      # PyTorch path only — absent in Warp march path
    "mlp_forward",
    "volume_rendering",
    "loss",
    "backward",
    "optimizer",
]


class CUDAProfiler:
    def __init__(self):
        self.enabled = False
        self.print_interval = 500
        self._pending: dict[str, list] = defaultdict(list)
        self._accumulated: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._active: dict[str, torch.cuda.Event] = {}

    def enable(self, print_interval: int = 500):
        self.enabled = True
        self.print_interval = print_interval

    def start(self, name: str):
        if not self.enabled:
            return
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._active[name] = e

    def stop(self, name: str):
        if not self.enabled or name not in self._active:
            return
        e_end = torch.cuda.Event(enable_timing=True)
        e_end.record()
        self._pending[name].append((self._active.pop(name), e_end))

    def tick(self, step: int):
        """Call once per training step. Syncs and reports every print_interval steps."""
        if not self.enabled:
            return
        # Discard any start events that never got a matching stop (e.g. skipped steps)
        self._active.clear()
        if step > 0 and step % self.print_interval == 0:
            torch.cuda.synchronize()
            for name, pairs in self._pending.items():
                for e_start, e_end in pairs:
                    try:
                        self._accumulated[name] += e_start.elapsed_time(e_end)
                        self._counts[name] += 1
                    except Exception:
                        pass
            self._pending.clear()
            self._report(step)
            self._accumulated.clear()
            self._counts.clear()

    def _report(self, step: int):
        n = self.print_interval
        print(f"\n{'='*60}")
        print(f"  CUDA Stage Profiling  |  Step {step}  |  avg over {n} steps")
        print(f"{'='*60}")
        total = 0.0
        printed = set()
        for name in _STAGE_ORDER:
            if name in self._accumulated and self._counts[name] > 0:
                avg = self._accumulated[name] / self._counts[name]
                total += avg
                print(f"  {name:<22} {avg:8.3f} ms/step")
                printed.add(name)
        for name in self._accumulated:
            if name not in printed and self._counts[name] > 0:
                avg = self._accumulated[name] / self._counts[name]
                total += avg
                print(f"  {name:<22} {avg:8.3f} ms/step")
        print(f"  {'-'*34}")
        print(f"  {'TOTAL (sum)':<22} {total:8.3f} ms/step")
        print(f"{'='*60}\n")


profiler = CUDAProfiler()
