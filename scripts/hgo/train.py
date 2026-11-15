"""Train one HGO variant (docs/METHOD.md). Single GPU, bf16 autocast.

usage: python scripts/hgo/train.py --variant H32 --out exp/hgo/runs/H32_s0 [--max-steps N for a smoke run]
Stage 1 (--warmup-epochs): encoder + detection / BEV + state probe. Stage 2: everything (JEPA world model with the
logged action, proposals, scorer on imagined futures with online PDM labels from the metric cache, as DrivoR).

v3 on the CG split (docs/CG_SPLIT.md, RESEARCH_DESIGN.md §3 Protocol B):
  python scripts/hgo/train.py --variant V3 --fold data/_lists/cg/split_v1/turn_vru --fold-variant withheld \
      [--holdout-cell "turn|vru"  (B-L12)] --warmup-check 0.25 --out exp/hgo/runs/V3_turn_vru_withheld_s0
--fold sets the train logs (the split's pool_logs.txt), the tokens (<fold>/train_<variant>.txt; 'full' = the whole pool,
the shared exposed reference) and the val logs (the split's val_logs.txt: training-session logs with no pool window), and
adds the split's metric cache. --cfg key=value overrides any Config field (ablations, e.g. --cfg prev_frame=false).
Warm-up stop (--warmup-check f > 0): every f epochs of stage 1, slot recall on the val logs; stage 2 starts when it gains
less than --warmup-tol points (after --warmup-min-epochs), at the latest after --warmup-max-epochs.
"""

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def pdm_scores(args):
    """PDM sub-scores of every proposal of one sample (DrivoR's get_sub_score, non-reactive metric cache)."""
    from ade2e_wm.hgo.pdm import proposal_scores
    path, poses = args
    return proposal_scores(path, poses)


def pdm_labels(args):
    """v2 labels of every proposal of one sample (pdm.proposal_labels: per-track at-fault / TTC, raw EP, flags, ...)."""
    from ade2e_wm.hgo.pdm import proposal_labels
    path, poses, tracks = args
    return proposal_labels(path, poses, tracks)


def metric_cache_paths_multi(dirs):
    """token -> metric_cache.pkl over several cache directories (the first directory wins on duplicates)."""
    out = {}
    for d in reversed([d for d in dirs if d]):
        out.update(metric_cache_paths(d))
    return out


def parse_overrides(items):
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def metric_cache_paths(d):
    if not d:
        return {}
    d = Path(d)
    csv = sorted((d / "metadata").glob("*.csv"))
    if not csv:
        return {}
    out = {}
    for line in csv[0].read_text().splitlines()[1:]:
        p = Path(line.strip())
        out[p.parent.name] = str(p)  # .../<log>/<scenario type>/<token>/metric_cache.pkl
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="H32")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-logs", default="", help="default: the split's pool_logs.txt with --fold, else the pilot "
                    "train logs; with --fold a given list must be a subset of the pool logs (smoke runs)")
    ap.add_argument("--val-logs", default="", help="default: the split's val_logs.txt with --fold, else the pilot val logs")
    ap.add_argument("--pkl-dir", default="data/navsim_logs/trainval")
    ap.add_argument("--sensors", default="data/sensor_blobs/trainval")
    ap.add_argument("--tokens", default="", help="optional file of t0 tokens to train on (e.g. a compositional split)")
    ap.add_argument("--metric-cache", nargs="+", default=["exp/metric_cache/train_hgo"])
    ap.add_argument("--fold", default="", help="CG fold directory, e.g. data/_lists/cg/split_v1/turn_vru")
    ap.add_argument("--fold-variant", default="withheld", choices=["withheld", "exposed", "full"])
    ap.add_argument("--holdout-cell", default="", help="L2 holdout (B-L12): mask the scorer labels of candidates that "
                    "perform this cell, e.g. 'turn|vru' (src/ade2e_wm/hgo/holdout.py)")
    ap.add_argument("--cfg", nargs="*", default=[], help="Config overrides key=value (JSON values)")
    ap.add_argument("--warmup-check", type=float, default=0.0, help="epochs between val recall checks in stage 1 (0: "
                    "fixed --warmup-epochs)")
    ap.add_argument("--warmup-tol", type=float, default=1.0, help="stop the warm-up when the macro slot recall gains "
                    "less than this (points) between two checks")
    ap.add_argument("--warmup-min-epochs", type=float, default=0.5)
    ap.add_argument("--warmup-max-epochs", type=float, default=2.0)
    ap.add_argument("--bev-dir", default="exp/hgo/bev")
    ap.add_argument("--wh", type=int, nargs=2, default=[1148, 672])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--pdm-workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-logs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--det-thr", type=float, default=0.3, help="slot detection threshold (0 in smoke runs to exercise slots)")
    ap.add_argument("--pdm-props", type=int, default=64, help="proposals per sample imagined by the world model and scored by PDM for the scorer loss "
                    "(random subset; 64 = all, as DrivoR)")
    ap.add_argument("--ema-sync", type=int, default=1, help="copy the online encoder into the EMA target at the start of "
                    "stage 2 (RESEARCH_DESIGN §8; the H32 pilot ran without it)")
    ap.add_argument("--val-batches", type=int, default=50, help="validation batches after every epoch (0: none)")
    a = ap.parse_args()
    os.chdir(ROOT)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.fold:
        split = Path(a.fold).parent  # a fold directory of a split (also for 'full', which uses the whole pool)
        if a.train_logs:
            extra_logs = set(Path(a.train_logs).read_text().split()) - set((split / "pool_logs.txt").read_text().split())
            assert not extra_logs, f"--train-logs outside the pool of {split}: {sorted(extra_logs)[:3]}"
        a.train_logs = a.train_logs or str(split / "pool_logs.txt")
        a.tokens = str(split / "pool_tokens.txt" if a.fold_variant == "full" else Path(a.fold) / f"train_{a.fold_variant}.txt")
        a.val_logs = a.val_logs or str(split / "val_logs.txt")
        extra = Path("exp/metric_cache") / f"cg_{split.name}_train"
        if extra.exists() and str(extra) not in a.metric_cache:
            a.metric_cache = a.metric_cache + [str(extra)]
        if a.holdout_cell:
            assert a.fold_variant == "withheld", "the L2 holdout belongs to the withheld model only"
    a.val_logs = a.val_logs or "data/_lists/phase1_val_logs.txt"
    a.train_logs = a.train_logs or "data/_lists/phase1_train_logs.txt"
    # the PDM pool forks before CUDA is initialised
    pool = ProcessPoolExecutor(a.pdm_workers, mp_context=mp.get_context("fork")) if a.pdm_workers else None

    import torch
    from torch.utils.data import DataLoader

    from ade2e_wm.hgo import losses as L
    from ade2e_wm.hgo.data import M_MAX, HGODataset, input_keys
    from ade2e_wm.hgo.holdout import cell_candidates, parse_cell
    from ade2e_wm.hgo.model import HGO, PROBE_GRID, count_params, make_config

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    read = lambda f: Path(f).read_text().split()
    logs = read(a.train_logs)[: a.max_logs or None]
    tokens = set(read(a.tokens)) if a.tokens else None
    cfg = make_config(a.variant, image_wh=tuple(a.wh), det_thr=a.det_thr,
                      vit_weights=str(ROOT / "checkpoints/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"),
                      **parse_overrides(a.cfg))
    G = cfg.perc_grid
    KEYS = input_keys(cfg.prev_frame)
    if a.holdout_cell:
        parse_cell(a.holdout_cell)
        assert cfg.wm_kind == "v2", "the L2 holdout masks the v2 / v3 readout losses"
    dkw = dict(wh=a.wh, horizons=cfg.horizons, bev_dir=a.bev_dir, area=G.area, prev=cfg.prev_frame)
    ds = HGODataset(logs, a.pkl_dir, a.sensors, tokens=tokens, train=True, **dkw)
    make_loader = lambda d: DataLoader(d, batch_size=a.bs, shuffle=True, num_workers=a.workers, pin_memory=True,
                                       drop_last=True, persistent_workers=a.workers > 0,
                                       prefetch_factor=2 if a.workers else None)
    ds_warm = copy.copy(ds)  # perception warm-up: no future images (JEPA targets unused)
    ds_warm.load_future_images = False
    dl = make_loader(ds)
    mc = metric_cache_paths_multi(a.metric_cache)
    v2 = cfg.wm_kind == "v2"
    val_dl = None
    if a.val_batches and Path(a.val_logs).exists():  # monitoring only: every window of the val logs, no model selection
        ds_val = HGODataset(read(a.val_logs)[: a.max_logs or None], a.pkl_dir, a.sensors, train=True, **dkw)
        ds_val.load_future_images = False
        val_dl = DataLoader(ds_val, batch_size=a.bs, shuffle=False, num_workers=min(a.workers, 4), drop_last=False)
    model = HGO(cfg).cuda()
    n_params = count_params(model)
    params = [p for p in model.parameters() if p.requires_grad]
    lr = a.lr * a.bs / 64  # DrivoR: base_lr 5e-4 at batch 64
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=a.wd)
    steps_per_epoch = len(dl)
    total = a.max_steps or a.epochs * steps_per_epoch
    warm = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    check_every = int(a.warmup_check * steps_per_epoch) if a.warmup_check > 0 and not a.max_steps else 0
    stage1 = int((a.warmup_max_epochs if check_every else a.warmup_epochs) * steps_per_epoch) if not a.max_steps \
        else a.max_steps // 3
    stage1_min = int(a.warmup_min_epochs * steps_per_epoch)
    win_tokens = [ds.logs[li].token[t] for li, t in ds.index]
    info = {"args": {k: str(v) for k, v in vars(a).items()}, "config": {k: str(v) for k, v in cfg.__dict__.items()},
            "params_trainable": n_params, "state_tokens": cfg.n_state, "n_reg": cfg.n_reg, "n_slot": cfg.n_slot,
            "params_by_module": {n: sum(p.numel() for p in m.parameters() if p.requires_grad)
                                 for n, m in model.named_children()},
            "manifest_tokens": len(tokens) if tokens else None, "train_windows": len(ds),
            "train_windows_with_metric_cache": sum(t in mc for t in win_tokens),
            "steps_per_epoch": steps_per_epoch, "metric_cache_entries": len(mc),
            "stage1_steps": stage1, "warmup_check_steps": check_every, "total_steps": total,
            "val_windows": len(val_dl.dataset) if val_dl else 0}
    (a.out / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    W = dict(hm=1.0, box=1.0, seg=1.0, hm_prev=0.5, box_prev=0.5, probe_hm=0.5, probe_seg=0.5, jepa_reg=1.0,
             jepa_slot=1.0, motion=0.5, traj=1.0,
             nc=1.0, dac=1.0, ep=1.0, ttc=1.0, comfort=1.0, ddc=1.0, risk=0.5,
             slot_probe_hm=0.5, slot_probe_box=0.5, view=1.0, phys=0.5, area=1.0, pair_fault=1.0, pair_ttc=1.0)
    if v2:  # same total weight on the latent prediction terms whatever the state (RESEARCH_DESIGN §8, P1)
        n_lat = int(cfg.n_reg > 0) + int(cfg.n_slot > 0)
        W["jepa_reg"] = W["view"] = 1.0 / max(n_lat, 1)

    @torch.no_grad()
    def validate(perception_only=False):
        """Monitoring on the val logs (no model selection): detection / BEV losses, slot recall per class at the model's
        threshold (class-aware, 2 m) and its macro mean (the warm-up stopping criterion), and without perception_only
        the ADE of the selected and of the closest proposal (no PDM)."""
        if val_dl is None:
            return None
        model.eval()
        s = {"hm": 0.0, "seg": 0.0} if perception_only else {"hm": 0.0, "seg": 0.0, "ade_selected": 0.0, "min_ade": 0.0}
        tp, ng = torch.zeros(4), torch.zeros(4)
        n = 0
        for i, vb in enumerate(val_dl):
            if i >= a.val_batches:
                break
            vb = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in vb.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model({k: vb[k] for k in KEYS}, perception_only=perception_only)
            tv = L.det_targets(vb["gt_box"], vb["gt_vel"], vb["gt_cls"], vb["gt_mask"], G)
            s["hm"] += float(L.det_loss(o["det"], tv, with_reg=False)["hm"])
            s["seg"] += float(L.seg_loss(o["det"]["seg"], L.bev_on_grid(vb["bev"], G), vb["bev_ok"]))
            if cfg.n_slot:
                t_, n_ = L.det_recall(o["slot_box"], o["slot_keep"], o["slot_cls"], vb["gt_box"], vb["gt_mask"], vb["gt_cls"])
                tp, ng = tp + t_, ng + n_
            if not perception_only:
                s["ade_selected"] += float((o["trajectory"][..., :2] - vb["traj"][..., :2]).norm(dim=-1).mean())
                s["min_ade"] += float((o["proposals"][-1][..., :2] - vb["traj"][:, None, :, :2]).norm(dim=-1).mean(-1)
                                      .amin(1).mean())
            n += 1
        model.train()
        res = {f"val_{k}": round(v / max(n, 1), 4) for k, v in s.items()}
        if cfg.n_slot:
            rec = 100 * tp / ng.clamp(min=1)
            res.update({f"val_recall_{c}": round(float(r), 2) for c, r in zip(["veh", "ped", "bike", "static"], rec)})
            res["val_recall_macro"] = round(float(rec[ng > 0].mean()), 2) if (ng > 0).any() else 0.0
        return res
    log = open(a.out / "train_log.jsonl", "a")
    step, start_ep = 0, 0
    if (a.out / "last.pt").exists() and not a.max_steps:  # resume at the next epoch boundary
        ck = torch.load(a.out / "last.pt", map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, start_ep = ck["step"], ck["epoch"] + 1
        stage1 = ck.get("stage1", stage1)
        print(json.dumps({"resumed_from_epoch": ck["epoch"], "step": step, "stage1": stage1}), flush=True)
    model.train()
    done = False
    synced = step > stage1  # a run resumed after the stage boundary keeps its EMA target
    n_bad = n_bad_run = n_pdm_fail = 0
    diag = {}
    acc = {"t_data": 0.0, "t_step": 0.0, "n": 0, "wall": time.time()}
    cur_future, cur_dl = None, None
    t_last = time.time()
    last_recall = None
    ep = start_ep
    # one pass = one epoch over the shuffled windows, except that a stage-1 pass ends at the stage boundary (fixed, or set
    # by the warm-up stopping criterion) so that stage 2 starts with the loader of future images
    while step < total and not done:
        t_ep = time.time()
        future = bool(a.max_steps) or step >= stage1
        if future != cur_future:
            cur_dl = None  # shuts down the previous loader's workers
            cur_dl, cur_future = (dl if future else make_loader(ds_warm)), future
        for batch in cur_dl:
            t_data = time.time() - t_last
            t0 = time.time()
            cuda_rng = torch.cuda.get_rng_state()
            b = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            x = {k: b[k] for k in KEYS}
            full = step >= stage1
            if full and a.ema_sync and not synced:
                model.sync_targets()
                synced = True
            sub = np.sort(np.random.permutation(cfg.n_prop)[:a.pdm_props])  # proposals imagined + scored this step
            subT = torch.from_numpy(sub).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, train_traj=b["traj"] if full else None, perception_only=not full, score_idx=subT)
            loss = {}
            tg = L.det_targets(b["gt_box"], b["gt_vel"], b["gt_cls"], b["gt_mask"], G)
            d = L.det_loss(out["det"], tg)
            loss["hm"], loss["box"] = d["hm"], d["box"]
            loss["seg"] = L.seg_loss(out["det"]["seg"], L.bev_on_grid(b["bev"], G), b["bev_ok"])
            if "det_prev" in out:  # v3 slot history: t-1 detections in the t0 frame, t-1 GT boxes
                dp = L.det_loss(out["det_prev"], L.det_targets(b["gt_box_prev"], b["gt_vel_prev"], b["gt_cls_prev"],
                                                               b["gt_mask_prev"], G))
                loss["hm_prev"], loss["box_prev"] = dp["hm"], dp["box"]
            tgp = tg if G == PROBE_GRID else L.det_targets(b["gt_box"], b["gt_vel"], b["gt_cls"], b["gt_mask"], PROBE_GRID)
            if "slot_probe" in out:  # v2 role probes: registers -> map classes, slots -> objects (heatmap + boxes)
                loss["probe_seg"] = L.seg_loss_map(out["probe"]["seg"], b["bev"], b["bev_ok"])
                sp = L.det_loss(out["slot_probe"], tgp)
                loss["slot_probe_hm"], loss["slot_probe_box"] = sp["hm"], sp["box"]
            else:
                loss["probe_hm"] = L.det_loss(out["probe"], tgp, with_reg=False)["hm"]
                loss["probe_seg"] = L.seg_loss(out["probe"]["seg"], b["bev"], b["bev_ok"])
            t_pdm = 0.0
            if full:
                props = out["proposals"][-1]
                P = props.detach().float().cpu().numpy()
                has = [tok in mc for tok in batch["token"]]
                if v2:
                    tracks = [s.split(",") if s else [] for s in batch["gt_track"]]
                    futs = [pool.submit(pdm_labels, (mc[tok], P[i][sub], tracks[i]))
                            if h and pool and np.isfinite(P[i][sub]).all() else None
                            for i, (tok, h) in enumerate(zip(batch["token"], has))]
                else:
                    futs = [pool.submit(pdm_scores, (mc[tok], P[i][sub])) if h and pool and np.isfinite(P[i][sub]).all()
                            else None for i, (tok, h) in enumerate(zip(batch["token"], has))]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tgt = model.targets(b["fut_img"], b["fut_cam2ego"], b["fut_K"], b["fut_box"], b["fut_box_ok"])
                by_cls = {"slot_cls": out["slot_cls"], "gt_cls": b["gt_cls"]} if v2 and cfg.n_slot else {}
                match = (L.match_slots(out["slot_box"], out["slot_keep"], b["gt_box"], b["gt_mask"], **by_cls) if cfg.n_slot
                         else torch.zeros(len(P), 0, dtype=torch.long, device="cuda"))
                if v2:
                    if cfg.n_reg:
                        loss["jepa_reg"] = L.reg_view_loss(model.reg_view, out["reg_fut"], tgt["reg"], b["fut_ego"])
                    if cfg.n_slot:
                        loss["view"] = L.view_loss(model.view, out["agent_lat"], tgt, match, b["fut_box"])
                        loss["phys"] = L.phys_loss(out["agent_phys"], out["slot_box"][..., :2], match, b["gt_fut"],
                                                   b["gt_fut_ok"])
                else:
                    loss.update(L.jepa_loss(out["wm_fut"], tgt, match, cfg.n_reg))
                    if cfg.n_slot:
                        loss["motion"] = L.motion_loss(out["wm_motion"], out["slot_box"][..., :2], match, b["gt_fut"], b["gt_fut_ok"])
                loss["traj"] = L.traj_loss(props, b["traj"])
                risk_y = (L.risk_labels(props.detach().index_select(1, subT), match, b["gt_box"], b["gt_fut"], b["gt_fut_ok"])
                          if "slot_risk" in out["scores"] else None)
                tp = time.time()
                results = [None] * len(futs)
                for i, f in enumerate(futs):
                    if f is not None:
                        try:
                            results[i] = f.result()
                        except Exception as e:  # one failed PDM evaluation must not end the run: no label for it
                            n_pdm_fail += 1
                            if n_pdm_fail <= 20:
                                log.write(json.dumps({"pdm_fail": n_pdm_fail, "step": step + 1, "token": batch["token"][i],
                                                      "error": f"{type(e).__name__}: {e}"[:300]}) + "\n")
                t_pdm = time.time() - tp
                if v2:
                    lab, hasT = L.collate_labels(results, len(sub), M_MAX)
                    lab, hasT = {k: v.cuda() for k, v in lab.items()}, hasT.cuda()
                    diag = {}
                    pw = None
                    if a.holdout_cell:  # L2: no scorer label for candidates that perform the held-out cell
                        hmask = cell_candidates(props.detach().index_select(1, subT), b["gt_box"], b["gt_mask"],
                                                b["gt_cls"], b["gt_fut"], b["gt_fut_ok"], a.holdout_cell)
                        pw = (~hmask).float()
                        diag["l2_masked"] = float(hmask.float().mean())
                    if "scores" in out:  # structured readout (context heads; pair heads in slot variants)
                        loss.update(L.score_loss_v2(out["scores"], lab, hasT, match, pw))
                    if "scores_mono" in out:
                        loss.update(L.score_loss_mono(out["scores_mono"], lab, hasT, pw=pw))
                    if "scores_pmono" in out:  # pair-mono readout (test 1, middle rung)
                        loss.update(L.score_loss_mono(out["scores_pmono"], lab, hasT, prefix="pmono_", pw=pw))
                    if cfg.n_slot:
                        fy, fm = L.pair_labels(lab["fault"], match)
                        fm = fm.expand_as(fy) & hasT[:, None, None]
                        diag["pair_fault_pos"] = float(fy[fm].mean()) if fm.any() else 0.0
                        diag["pair_labelled"] = float(fm.float().sum(-1).mean())
                        diag["unsafe_props"], diag["unsafe_uncovered"] = L.fault_coverage(lab, hasT, match)
                    diag["fault_unmatched"] = float(lab["fault_unmatched"][hasT].mean()) if hasT.any() else 0.0
                else:
                    pdm = torch.zeros(len(P), len(sub), 7)
                    for i, r in enumerate(results):
                        if r is not None:
                            pdm[i] = torch.from_numpy(r)
                    hasT = torch.tensor([r is not None for r in results], device="cuda")
                    loss.update(L.score_loss(out["scores"], pdm.cuda(), hasT, risk_y))
            total_loss = sum(W.get(k, 1.0) * v for k, v in loss.items())
            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 5.0)
            if bool(torch.isfinite(total_loss)) and bool(torch.isfinite(gn)):
                opt.step()
                if full:
                    model.update_ema()
                n_bad_run = 0
            else:  # skip the update; record what was non-finite; keep the first case for offline replay
                n_bad += 1
                n_bad_run += 1
                rec = {"nonfinite_step": step + 1, "stage": 2 if full else 1, "grad_norm": float(gn),
                       "terms": [k for k, v in loss.items() if not bool(torch.isfinite(v))],
                       "grad_params": [n for n, p in model.named_parameters()
                                       if p.grad is not None and not bool(torch.isfinite(p.grad).all())][:20],
                       "tokens": list(batch["token"])}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
                if n_bad == 1:
                    torch.save({"model": model.state_dict(), "batch": batch, "cuda_rng": cuda_rng, "sub": sub,
                                "step": step, "full": full, "config": cfg.__dict__}, a.out / "nonfinite_dump.pt")
                opt.zero_grad(set_to_none=True)
                if n_bad_run >= 20 or n_bad >= 200:
                    raise SystemExit(f"stopping: {n_bad} non-finite steps ({n_bad_run} in a row)")
            sched.step()
            torch.cuda.synchronize()
            t_gpu = time.time() - t0
            step += 1
            acc["t_data"] += t_data
            acc["t_step"] += t_gpu
            acc["n"] += 1
            if step % a.log_every == 0 or step <= 3 or (a.max_steps and step == a.max_steps):
                rec = {"step": step, "epoch": ep, "stage": 2 if full else 1, "loss": float(total_loss),
                       **{k: round(float(v), 5) for k, v in loss.items()}, "grad_norm": float(gn),
                       "lr": sched.get_last_lr()[0], "t_data": round(t_data, 3), "t_step": round(t_gpu, 3),
                       "t_pdm_wait": round(t_pdm, 3), "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                       "t_data_avg": round(acc["t_data"] / acc["n"], 3), "t_step_avg": round(acc["t_step"] / acc["n"], 3),
                       "s_per_step": round((time.time() - acc["wall"]) / acc["n"], 3), "time": round(time.time()),
                       "skipped_nonfinite": n_bad, "pdm_fail": n_pdm_fail}
                acc = {"t_data": 0.0, "t_step": 0.0, "n": 0, "wall": time.time()}
                if not all(bool(torch.isfinite(p).all()) for p in params):
                    raise SystemExit(f"non-finite parameters at step {step}")
                if full:
                    rec["ade_selected_sub"] = float((out["trajectory"][..., :2] - b["traj"][..., :2]).norm(dim=-1).mean())
                    rec["min_ade_all"] = float((props.detach()[..., :2] - b["traj"][:, None, :, :2]).norm(dim=-1).mean(-1).amin(1).mean())
                    rec["slots_kept"] = float(out["slot_keep"].float().sum(1).mean()) if cfg.n_slot else 0.0
                    rec["slots_matched"] = float((match >= 0).float().sum(1).mean()) if cfg.n_slot else 0.0
                    if "slot_hist_ok" in out:
                        rec["slots_with_history"] = float(out["slot_hist_ok"].float().sum(1).mean())
                    rec.update({k: round(v, 4) for k, v in diag.items()})
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
            if step % 2000 == 0:  # debugging snapshot only (resume uses last.pt at epoch boundaries)
                torch.save({"model": model.state_dict(), "step": step}, a.out / "mid.tmp.pt")
                os.replace(a.out / "mid.tmp.pt", a.out / "mid.pt")
            if not full and check_every and step % check_every == 0 and step >= stage1_min:
                r = validate(perception_only=True) or {}
                rec = {"warmup_check": step, **r}
                cur = r.get("val_recall_macro")
                if cur is not None and last_recall is not None and cur - last_recall < a.warmup_tol:
                    stage1 = step
                    rec["warmup_stop"] = step
                last_recall = cur if cur is not None else last_recall
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
            t_last = time.time()
            if a.max_steps and step >= a.max_steps:
                done = True
                break
            if step >= total:
                break
            if not future and step >= stage1:  # stage boundary: end this pass, the next one loads future images
                break
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "config": cfg.__dict__,
              "step": step, "epoch": ep, "stage1": stage1}
        torch.save(ck, a.out / "last.tmp.pt")
        os.replace(a.out / "last.tmp.pt", a.out / "last.pt")
        rec = {"epoch_done": ep, "step": step, "hours": round((time.time() - t_ep) / 3600, 2)}
        rec.update(validate(perception_only=step <= stage1) or {})
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        ep += 1
    if pool:
        pool.shutdown()


if __name__ == "__main__":
    main()
