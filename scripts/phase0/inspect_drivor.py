"""Inspect the released DrivoR checkpoints and the exact token shapes of the Nav2 model.

Loads the checkpoint, instantiates the agent from configs/navsim/agent/drivor_nav2.yaml, checks the
state dict loads with no missing/unexpected keys, and runs one forward pass on a random batch with
hooks on the image encoder and decoders to record tensor shapes. CPU only.
"""
import json
import os
import sys
import time
from collections import Counter

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

root = os.environ["ADE2E_ROOT"]
out = {}
for name in ["drivor_Nav2_10epochs.pth", "drivor_Nav1_train_only_85k_25epochs.ckpt"]:
    ck = torch.load(f"{root}/checkpoints/drivor/{name}", map_location="cpu")
    sd = ck["state_dict"] if "state_dict" in ck else ck
    prefixes = Counter(".".join(k.split(".")[:3]) for k in sd)
    out[name] = {
        "top_level_keys": [k for k in ck.keys()][:20],
        "n_tensors": len(sd),
        "n_params_total": int(sum(v.numel() for v in sd.values() if torch.is_tensor(v))),
        "module_param_counts": {p: int(sum(v.numel() for k, v in sd.items() if k.startswith(p))) for p in prefixes},
        "scene_embeds": list(sd.get("agent._drivor_model.scene_embeds", torch.empty(0)).shape),
        "hparams": {k: str(v)[:300] for k, v in ck.items() if k not in ("state_dict", "optimizer_states")}
        if isinstance(ck, dict) else {},
    }

with initialize_config_dir(config_dir=f"{root}/configs/navsim/agent", version_base=None):
    cfg = compose(config_name="drivor_nav2", overrides=[f"checkpoint_path={root}/checkpoints/drivor/drivor_Nav2_10epochs.pth"])
agent = instantiate(cfg)
sd = torch.load(cfg.checkpoint_path, map_location="cpu")["state_dict"]
res = agent.load_state_dict({k.replace("agent._drivor_model", "_drivor_model"): v for k, v in sd.items()}, strict=False)
out["load_state_dict"] = {"missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys)}
model = agent._drivor_model.eval()
out["trainable_vs_frozen_params_in_model"] = {
    "total": sum(p.numel() for p in model.parameters()),
    "requires_grad": sum(p.numel() for p in model.parameters() if p.requires_grad),
}

shapes = {}
def hook(name):
    def fn(m, inp, o):
        def sh(x):
            if torch.is_tensor(x):
                return list(x.shape)
            if isinstance(x, (list, tuple)):
                return [sh(y) for y in x]
            return type(x).__name__
        shapes[name] = {"in": [sh(i) for i in inp], "out": sh(o)}
    return fn
model.image_backbone.register_forward_hook(hook("image_backbone(ImgEncoder)"))
model.trajectory_decoder.register_forward_hook(hook("trajectory_decoder"))
model.scorer_attention.register_forward_hook(hook("scorer_attention"))
model.scorer.register_forward_hook(hook("scorer"))

H, W = cfg.config.image_size[1], cfg.config.image_size[0]
feats = {"image": torch.randn(1, 4, 3, H, W), "ego_status": torch.randn(1, 4, 11)}
with torch.no_grad():
    t0 = time.time(); o = model(feats); dt = time.time() - t0
out["forward_shapes"] = shapes
out["outputs"] = {k: (list(v.shape) if torch.is_tensor(v) else type(v).__name__) for k, v in o.items()}
out["cpu_forward_seconds_1thread_x%d" % torch.get_num_threads()] = dt
vit = model.image_backbone.model.lora_vit
out["vit"] = {"patch": vit.patch_embed.patch_size, "grid": vit.patch_embed.grid_size, "embed_dim": vit.embed_dim,
              "num_prefix_tokens(cls+reg)": vit.num_prefix_tokens, "depth": len(vit.blocks)}
print(json.dumps(out, indent=1, default=str))
json.dump(out, open(f"{root}/docs/phase0/drivor_checkpoint_inspection.json", "w"), indent=1, default=str)
