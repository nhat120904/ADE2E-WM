"""DrivoR Nav2 model forward latency / memory on one GPU (batch 1, 4 cams 672x1148), fp32 and bf16 autocast.

Measures the model forward only (image backbone + decoders + scorer), not feature building / JPEG decode.
"""
import json
import os
import time

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

root = os.environ["ADE2E_ROOT"]
with initialize_config_dir(config_dir=f"{root}/configs/navsim/agent", version_base=None):
    cfg = compose(config_name="drivor_nav2", overrides=[f"checkpoint_path={root}/checkpoints/drivor/drivor_Nav2_10epochs.pth"])
agent = instantiate(cfg)
agent.initialize()
model = agent._drivor_model.cuda().eval()
H, W = cfg.config.image_size[1], cfg.config.image_size[0]
res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda}
for bs in [1, 8]:
    feats = {"image": torch.randn(bs, 4, 3, H, W, device="cuda"), "ego_status": torch.randn(bs, 4, 11, device="cuda")}
    for mode in ["fp32", "bf16"]:
        torch.cuda.reset_peak_memory_stats()
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if mode == "bf16" else torch.autocast("cuda", enabled=False)
        with torch.no_grad(), ctx:
            for _ in range(5):
                model(feats)
            torch.cuda.synchronize()
            ts = []
            for _ in range(30):
                t0 = time.perf_counter(); model(feats); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        ts = sorted(ts)
        res[f"bs{bs}_{mode}"] = {"median_ms": 1e3 * ts[len(ts) // 2], "p90_ms": 1e3 * ts[int(0.9 * len(ts))],
                                 "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
print(json.dumps(res, indent=1))
json.dump(res, open(f"{root}/docs/phase0/drivor_gpu_latency.json", "w"), indent=1)
