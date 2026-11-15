"""Run a trained HGO checkpoint on navtest and write the selected trajectories for the official EPDMS evaluation
(src/ade2e_wm/agents/precomputed_agent.py): <run>/navtest_plans.npz with token[N], plan[N,8,3]; also every proposal
props[N,64,8,3], the selection score sel[N,64] and the scorer's sub-score probabilities sub[N,64,6] (+ slot-risk NC term)
for the oracle / regret analysis (scripts/hgo/oracle_props.py); raw head logits logit[N,64,6], and for slot variants
slot_risk[N,64,S] (logits, -30 = empty slot), slot_cls[N,S], slot_box[N,S,7], to recompute the selection offline
(scripts/hgo/selection_analysis.py).

v2 checkpoints (config wm_kind="v2"): plan / sel / sub follow the selection rule (--select overrides the config's rule
for readout="both"); every available rule is also saved as sel_<rule>, sub_<rule> ("v2" structured, "mono", "pmono");
the structured readout's ctx_logit [N,64,4] (dac, ddc, ep, comfort), pair_fault / pair_ttc [N,64,S] logits (-30 = empty
slot) and area [N,64,8,4] flag probabilities; the monolithic logit_mono and the pair-mono logit_pmono [N,64,6]
(SUB_SCORES order); slot variants
agent_phys [N,S,8,4], slot_cls, slot_box. No future swap (v2 forecasts once per scene on log-replay).

usage: python scripts/hgo/dump_plans.py --run exp/hgo/runs/H32_s0 [--ckpt last.pt]
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ade2e_wm.hgo.data import HGODataset, input_keys
from ade2e_wm.hgo.model import HGO, SUB_SCORES, Config, decode_boxes

ROOT = Path(__file__).resolve().parents[2]


SENS_KEYS = ["spread_reg", "spread_slot", "hchange_reg", "hchange_slot", "motion_spread_m", "motion_disp_m"]


@torch.no_grad()
def wm_sensitivity(out, c):
    """How much the imagined future depends on the proposal, per scene (float32):
    spread_*  = mean over proposals of ||f(p) - mean_p f|| / mean ||f||  (registers / kept slots, all horizons)
    hchange_* = mean ||f(last horizon) - f(first horizon)|| / mean ||f||  (reference: change over time)
    motion_spread_m = std over proposals of the predicted slot position at 4 s (m, mean over kept slots); NAVSIM logs
      are non-reactive, so the true agent future does not depend on the ego action;
    motion_disp_m = mean predicted slot displacement at 4 s (m), as reference."""
    f = out["imagined"].float()  # (B*P, H, N, d)
    B = out["state"].shape[0]
    P = f.shape[0] // B
    f = f.view(B, P, *f.shape[1:])
    nr = c.n_reg
    keep = out["keep"][:, nr:].float()  # (B, S)
    res = {}
    for name, g, m in [("reg", f[:, :, :, :nr], None), ("slot", f[:, :, :, nr:], keep)]:
        if g.shape[3] == 0:
            continue
        norm = g.norm(dim=-1)  # (B, P, H, T)
        dev = (g - g.mean(1, keepdim=True)).norm(dim=-1)
        hch = (g[:, :, -1] - g[:, :, 0]).norm(dim=-1)  # (B, P, T)
        if m is None:
            m = torch.ones(B, g.shape[3], device=g.device)
        w = m[:, None, None].expand_as(norm)

        def mean(v, ww):
            return (v * ww).sum(tuple(range(1, v.dim()))) / ww.sum(tuple(range(1, v.dim()))).clamp(min=1)
        den = mean(norm, w).clamp(min=1e-6)
        empty = m.sum(1) == 0  # no kept slot in this scene
        res[f"spread_{name}"] = (mean(dev, w) / den).masked_fill(empty, float("nan"))
        res[f"hchange_{name}"] = (mean(hch, m[:, None].expand_as(hch)) / den).masked_fill(empty, float("nan"))
    if c.n_slot:
        mot = out["imagined_motion"].float()  # (B*P, S, 8, 2) offsets from the current slot position
        mot = mot.view(B, P, *mot.shape[1:])[..., -1, :]  # (B, P, S, 2) at 4 s
        sd = mot.std(1).norm(dim=-1)  # (B, S)
        res["motion_spread_m"] = (sd * keep).sum(1) / keep.sum(1).clamp(min=1)
        res["motion_disp_m"] = (mot.norm(dim=-1).mean(1) * keep).sum(1) / keep.sum(1).clamp(min=1)
    return {k: v.cpu().numpy().astype(np.float32) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--logs", default="data/_lists/phase1_test_logs.txt")
    ap.add_argument("--tokens", default="data/_lists/navtest_tokens.txt")
    ap.add_argument("--pkl-dir", default="data/navsim_logs/test")
    ap.add_argument("--sensors", default="data/sensor_blobs/test")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-logs", type=int, default=0)
    ap.add_argument("--out", default="navtest_plans.npz")
    ap.add_argument("--swap", type=int, default=1, help="also score every proposal with another proposal's imagined "
                    "future (fixed derangement per batch) and record how much the imagined future depends on the action")
    ap.add_argument("--select", default="", help="v2 readout='both': selection rule for the plan (v2 | mono | pmono)")
    ap.add_argument("--raw-det", type=int, default=0, help="also save the top-K detections BEFORE the score threshold "
                    "(det_box [N,K,7], det_cls [N,K], det_score [N,K]) for the perception diagnosis "
                    "(scripts/hgo/perception_diag.py)")
    a = ap.parse_args()
    os.chdir(ROOT)
    ck = torch.load(a.run / a.ckpt, map_location="cpu")
    cfg = Config(**{**ck["config"], "vit_weights": "", **({"select": a.select} if a.select else {})})
    model = HGO(cfg)
    model.load_state_dict(ck["model"])
    model = model.cuda().eval()
    logs = Path(a.logs).read_text().split()[: a.max_logs or None]
    toks = set(Path(a.tokens).read_text().split())
    ds = HGODataset(logs, a.pkl_dir, a.sensors, wh=cfg.image_wh, horizons=cfg.horizons, tokens=toks, train=False,
                    area=cfg.perc_grid.area, prev=cfg.prev_frame)
    keys = input_keys(cfg.prev_frame)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, pin_memory=True)
    tokens, plans, props, sel, sub, t0 = [], [], [], [], [], time.time()
    raw = {"det_box": [], "det_cls": [], "det_score": []}
    logit, risk, scls, sbox = [], [], [], []  # raw head logits, per-slot risk logits, slot class / box (selection analysis)
    sw = {k: [] for k in ["logit_swap", "risk_swap", "sel_swap", "perm"]}
    sens = {k: [] for k in SENS_KEYS}
    v2 = {k: [] for k in ["pair_fault", "pair_ttc", "area", "agent_phys", "ctx_logit", "logit_mono", "logit_pmono",
                          "sel_v2", "sel_mono", "sel_pmono", "sub_v2", "sub_mono", "sub_pmono"]}  # v2 readout outputs
    is_v2 = cfg.wm_kind == "v2"
    if is_v2 and a.swap:  # v2 forecasts once per scene on log-replay: there is no per-proposal future to swap
        print("v2 model: --swap ignored")
        a.swap = 0
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in dl:
            x = {k: b[k].cuda(non_blocking=True) for k in keys}
            out = model(x, return_imagined=bool(a.swap))
            plans.append(out["trajectory"].float().cpu().numpy())
            if a.raw_det:
                db, _, dc, ds_ = decode_boxes(out["det"], a.raw_det, 0.0, model.pgrid)
                raw["det_box"].append(db.float().cpu().numpy().astype(np.float16))
                raw["det_cls"].append(dc.cpu().numpy().astype(np.int8))
                raw["det_score"].append(ds_.float().cpu().numpy().astype(np.float16))
            props.append(out["proposals"][-1].float().cpu().numpy())
            sel.append(out["pdm_score"].float().cpu().numpy())
            if is_v2:
                np_ = lambda t: t.float().cpu().numpy()
                sub.append(np_(out["sub_probs"]))
                for r in ("v2", "mono", "pmono"):
                    if f"pdm_score_{r}" in out:
                        v2[f"sel_{r}"].append(np_(out[f"pdm_score_{r}"]))
                        v2[f"sub_{r}"].append(np_(out[f"sub_probs_{r}"]))
                for r in ("mono", "pmono"):
                    if f"scores_{r}" in out:
                        v2[f"logit_{r}"].append(np_(torch.stack([out[f"scores_{r}"][k] for k in SUB_SCORES], -1)))
                if "scores" in out:
                    sc = out["scores"]
                    v2["ctx_logit"].append(np_(torch.stack([sc[k] for k in ("dac", "ddc", "ep", "comfort")], -1)))
                    v2["area"].append(np_(sc["area"].sigmoid()).astype(np.float16))
                    if "pair_fault" in sc:
                        v2["pair_fault"].append(np_(sc["pair_fault"]).astype(np.float16))
                        v2["pair_ttc"].append(np_(sc["pair_ttc"]).astype(np.float16))
                if cfg.n_slot:
                    v2["agent_phys"].append(out["agent_phys"].float().cpu().numpy().astype(np.float16))
                    scls.append(out["slot_cls"].cpu().numpy().astype(np.int8))
                    sbox.append(out["slot_box"].float().cpu().numpy().astype(np.float16))
            else:
                sub.append(torch.stack([out["scores"][k].float().sigmoid() for k in SUB_SCORES], -1).cpu().numpy())
                logit.append(torch.stack([out["scores"][k].float() for k in SUB_SCORES], -1).cpu().numpy())
            if "slot_risk" in out["scores"]:  # (B,64,S); empty slots are -30
                risk.append(out["scores"]["slot_risk"].float().cpu().numpy().astype(np.float16))
                scls.append(out["slot_cls"].cpu().numpy().astype(np.int8))
                sbox.append(out["slot_box"].float().cpu().numpy().astype(np.float16))
            if a.swap:
                for k, v in wm_sensitivity(out, cfg).items():
                    sens[k].append(v)
                order = torch.randperm(cfg.n_prop, generator=gen)
                perm = torch.empty_like(order)
                perm[order] = order.roll(-1)  # derangement: no proposal keeps its own future
                o2 = model(x, fut_perm=perm.cuda())
                sw["logit_swap"].append(torch.stack([o2["scores"][k].float() for k in SUB_SCORES], -1).cpu().numpy())
                sw["sel_swap"].append(o2["pdm_score"].float().cpu().numpy())
                if "slot_risk" in o2["scores"]:
                    sw["risk_swap"].append(o2["scores"]["slot_risk"].float().cpu().numpy().astype(np.float16))
                sw["perm"].append(np.tile(perm.numpy()[None], (len(b["token"]), 1)))
            tokens += list(b["token"])
    plan = np.concatenate(plans).astype(np.float32)
    np.savez(a.run / a.out, token=np.array(tokens), plan=plan, props=np.concatenate(props).astype(np.float32),
             sel=np.concatenate(sel).astype(np.float32), sub=np.concatenate(sub).astype(np.float32),
             sub_names=np.array(SUB_SCORES), score_w=json.dumps(cfg.score_w),
             **({"logit": np.concatenate(logit).astype(np.float32)} if logit else {}),
             **({"slot_risk": np.concatenate(risk)} if risk else {}),
             **({"slot_cls": np.concatenate(scls), "slot_box": np.concatenate(sbox)} if scls else {}),
             **{k: np.concatenate(v) for k, v in v2.items() if v}, **{k: np.concatenate(v) for k, v in raw.items() if v},
             **{k: np.concatenate(v) for k, v in sw.items() if v}, **{k: np.concatenate(v) for k, v in sens.items() if v})
    missing = sorted(toks - set(tokens))  # the precomputed agent fails on these and the devkit then stops
    info = {"scenes": len(tokens), "navtest_tokens": len(toks), "missing": len(missing), "missing_first": missing[:20],
            "seconds": round(time.time() - t0, 1), "ckpt": str(a.run / a.ckpt), "step": ck.get("step")}
    (a.run / (Path(a.out).stem + ".json")).write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
