"""HGO losses (docs/METHOD.md). Labels come only from the dataset's target keys; nothing here feeds the model."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .geometry import N_CLS, ego_rects, rect_corners_2d, rects_overlap
from .model import PROBE_GRID, SUB_SCORES, Grid


def det_targets(gt_box, gt_vel, gt_cls, gt_mask, grid: Grid = PROBE_GRID):
    """CenterNet targets on `grid` (pilot: 64 x 64, 1 m; v3: 160 x 128, 0.5 m): heatmap (B,C,Rx,Ry), centre pixel (B,M)
    flat index, regression (B,M,10), valid (B,M). Gaussian sigma = sqrt(l w) / 4 in cells, at least 0.8 cell."""
    B, M, _ = gt_box.shape
    Rx, Ry, cell = grid.rows, grid.cols, grid.cell
    fi = (gt_box[..., 0] - grid.x0) / cell - 0.5
    fj = (gt_box[..., 1] - grid.y0) / cell - 0.5
    i, j = fi.round(), fj.round()
    ok = gt_mask & (i >= 0) & (i < Rx) & (j >= 0) & (j < Ry)
    sigma = (0.25 * (gt_box[..., 3] * gt_box[..., 4]).clamp(min=1e-3).sqrt() / cell).clamp(min=0.8)
    ax = torch.arange(Rx, device=gt_box.device, dtype=gt_box.dtype)
    ay = torch.arange(Ry, device=gt_box.device, dtype=gt_box.dtype)
    s2 = 2 * sigma ** 2
    gx = torch.exp(-(ax[None, None] - i[..., None]) ** 2 / s2[..., None])           # B,M,Rx (separable Gaussian)
    gy = torch.exp(-(ay[None, None] - j[..., None]) ** 2 / s2[..., None])           # B,M,Ry
    hm = []
    for c in range(N_CLS):
        w = (ok & (gt_cls == c)).to(gt_box.dtype)
        hm.append((gx[..., :, None] * gy[..., None, :] * w[..., None, None]).amax(1) if M else gt_box.new_zeros(B, Rx, Ry))
    hm = torch.stack(hm, 1)
    reg = torch.stack([fi - i, fj - j, gt_box[..., 3].clamp(min=0.05).log(), gt_box[..., 4].clamp(min=0.05).log(),
                       gt_box[..., 2], gt_box[..., 5].clamp(min=0.05).log(), torch.sin(gt_box[..., 6]),
                       torch.cos(gt_box[..., 6]), gt_vel[..., 0], gt_vel[..., 1]], -1)
    pix = (i.clamp(0, Rx - 1) * Ry + j.clamp(0, Ry - 1)).long()
    return hm, pix, reg, ok


def bev_on_grid(bev, grid: Grid):
    """Cached BEV semantics (B,64,64) on the pilot grid -> labels on `grid` (B,rows,cols); cells outside the cached area
    get -100 (ignored by cross_entropy). Identity on the pilot grid."""
    if grid == PROBE_GRID:
        return bev
    src = PROBE_GRID
    xs = grid.x0 + (torch.arange(grid.rows, device=bev.device) + 0.5) * grid.cell
    ys = grid.y0 + (torch.arange(grid.cols, device=bev.device) + 0.5) * grid.cell
    si = torch.floor((xs - src.x0) / src.cell).long()
    sj = torch.floor((ys - src.y0) / src.cell).long()
    oki, okj = (si >= 0) & (si < src.rows), (sj >= 0) & (sj < src.cols)
    out = bev[:, si.clamp(0, src.rows - 1)][:, :, sj.clamp(0, src.cols - 1)]
    return out.masked_fill(~(oki[:, None] & okj[None]), -100)


def focal_hm(logits, gt):
    p = logits.float().sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    lp = -(torch.log(p) * (1 - p) ** 2 * pos).sum()
    ln = -(torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * neg).sum()
    return (lp + ln) / pos.sum().clamp(min=1)


def det_loss(out, tg, with_reg=True):
    hm, pix, reg, ok = tg
    loss = {"hm": focal_hm(out["hm"], hm)}
    if with_reg:
        pr = out["reg"].float().flatten(2).gather(2, pix[:, None].expand(-1, 10, -1)).transpose(1, 2)
        w = torch.tensor([1, 1, 1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5], device=pr.device)
        l1 = (F.l1_loss(pr, reg, reduction="none") * w).sum(-1)
        loss["box"] = (l1 * ok).sum() / ok.sum().clamp(min=1)
    return loss


def seg_loss(logits, bev, bev_ok):
    """BEV semantics; labels -100 (outside the cached area, see bev_on_grid) are ignored."""
    if not bev_ok.any():
        return logits.sum() * 0
    return F.cross_entropy(logits[bev_ok].float(), bev[bev_ok], ignore_index=-100)


def seg_loss_map(logits, bev, bev_ok):
    """BEV semantics restricted to the map classes (road, walkway, centerline, none): cells of objects (classes 4-6) hide
    the map class below them and are ignored. Used by the register probe (v2 role probes)."""
    if not bev_ok.any():
        return logits.sum() * 0
    t = bev[bev_ok].clone()
    t[t >= 4] = -100
    return F.cross_entropy(logits[bev_ok].float(), t, ignore_index=-100)


@torch.no_grad()
def match_slots(slot_box, slot_keep, gt_box, gt_mask, max_dist=2.0, slot_cls=None, gt_cls=None):
    """Hungarian matching of selected slots to t0 GT boxes on centre distance (<= max_dist m): (B,S) GT index or -1.
    With slot_cls and gt_cls (v2), a slot only matches a GT box of its own class. Pairs beyond max_dist (or of another
    class) are gated BEFORE the assignment with one large equal cost, so the assignment maximizes the number of valid
    matches and then minimizes their distance (gating after the assignment can drop valid pairs)."""
    B, S = slot_keep.shape
    out = torch.full((B, S), -1, dtype=torch.long)
    sb, sk, gb, gm = (t.detach().float().cpu() if t.is_floating_point() else t.cpu() for t in (slot_box, slot_keep, gt_box, gt_mask))
    by_cls = slot_cls is not None and gt_cls is not None
    if by_cls:
        sc, gc = slot_cls.cpu(), gt_cls.cpu()
    big = 1e6
    for b in range(B):
        si = torch.nonzero(sk[b]).squeeze(1)
        gi = torch.nonzero(gm[b]).squeeze(1)
        if len(si) == 0 or len(gi) == 0:
            continue
        d = torch.cdist(sb[b, si, :2], gb[b, gi, :2])
        bad = d > max_dist
        if by_cls:
            bad |= sc[b, si][:, None] != gc[b, gi][None]
        d = torch.where(bad, torch.full_like(d, big), d).numpy()
        r, c = linear_sum_assignment(d)
        good = d[r, c] < big
        out[b, si[r[good]]] = gi[c[good]]
    return out.to(slot_box.device)


@torch.no_grad()
def fault_coverage(lab, has, match):
    """How much of the safety supervision the slots can carry: among proposals with an at-fault collision (PDM), the share
    whose colliding objects include no track matched to a slot (missed by the detector, not among the selected slots,
    unmatched, or not a GT track of the sample). Returns (share of proposals with a fault, share of those uncovered)."""
    if not has.any():
        return 0.0, 0.0
    f = lab["fault"][has] > 0                                   # B,P,M
    unsafe = f.any(-1) | (lab["fault_unmatched"][has] > 0)       # B,P
    matched = torch.zeros(f.shape[0], f.shape[2], dtype=torch.bool, device=f.device)
    m = match[has]
    if m.numel():
        matched.scatter_(1, m.clamp(min=0), m >= 0)
        matched[:, 0] = (m == 0).any(1)  # index 0 also receives the clamped -1 entries: recompute it exactly
    covered = (f & matched[:, None]).any(-1)
    n = unsafe.sum()
    return float(unsafe.float().mean()), float((unsafe & ~covered).sum() / n.clamp(min=1))


def gather_m(t, idx):
    """t (B,M,...) indexed by idx (B,S) (-1 -> 0) -> (B,S,...)."""
    i = idx.clamp(min=0)
    return t.gather(1, i.view(*i.shape, *([1] * (t.dim() - 2))).expand(-1, -1, *t.shape[2:]))


def jepa_loss(pred_fut, tgt, match, n_reg):
    """pred_fut (B,H,N,d) from the logged action; targets: registers (B,H,n_reg,d), slots (B,H,M,d) + ok (B,H,M)."""
    out = {}
    if n_reg:
        out["jepa_reg"] = F.smooth_l1_loss(pred_fut[:, :, :n_reg].float(), tgt["reg"])
    if "slot" in tgt and pred_fut.shape[2] > n_reg:
        ps = pred_fut[:, :, n_reg:].float()  # B,H,S,d
        H = ps.shape[1]
        ts = torch.stack([gather_m(tgt["slot"][:, h], match) for h in range(H)], 1)
        ok = torch.stack([gather_m(tgt["slot_ok"][:, h], match) for h in range(H)], 1) & (match >= 0)[:, None]
        l = F.smooth_l1_loss(ps, ts, reduction="none").mean(-1)
        out["jepa_slot"] = (l * ok).sum() / ok.sum().clamp(min=1)
    return out


def motion_loss(motion, slot_xy, match, gt_fut, gt_fut_ok):
    """motion (B,S,8,2) offsets from the detected slot position; target = GT future of the matched track."""
    tgt = gather_m(gt_fut[..., :2], match) - slot_xy[:, :, None]
    ok = gather_m(gt_fut_ok, match) & (match >= 0)[..., None]
    l = F.smooth_l1_loss(motion.float(), tgt, reduction="none").sum(-1)
    return (l * ok).sum() / ok.sum().clamp(min=1)


def traj_loss(props, traj):
    """DrivoR: L1 over (x, y, yaw), mean over poses, min over proposals (prev_weight 0: last refinement only)."""
    return torch.linalg.norm(props.float() - traj[:, None], dim=-1, ord=1).mean(-1).amin(1).mean()


@torch.no_grad()
def risk_labels(props, match, gt_box, gt_fut, gt_fut_ok):
    """1 where the ego footprint along a proposal overlaps the GT footprint of the slot's matched track at any future
    step (non-reactive replay, as the PDM scorer); 0 for unmatched slots. (B,P,S)"""
    B, P = props.shape[:2]
    S = match.shape[1]
    ego = ego_rects(props.float())  # B,P,8,4,2
    f = gather_m(gt_fut, match)  # B,S,8,3
    ok = gather_m(gt_fut_ok, match) & (match >= 0)[..., None]
    lw = gather_m(gt_box[..., 3:5], match)  # B,S,2
    ag = rect_corners_2d(f[..., :2], f[..., 2], lw[..., None, 0].expand_as(f[..., 2]), lw[..., None, 1].expand_as(f[..., 2]))
    shape = (B, P, S) + ego.shape[2:]
    hit = rects_overlap(ego[:, :, None].expand(shape), ag[:, None].expand(shape))  # B,P,S,8
    return (hit & ok[:, None]).any(-1).float()


def log1mexp(x):
    """log(1 - exp(x)) for x <= 0."""
    x = x.clamp(max=-1e-6)
    return torch.where(x > -0.693, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def score_loss(logits, pdm, has, risk_y=None):
    """DrivoR's scorer loss on proposals with PDM labels pdm (B,P,7: nc, dac, ep, ttc, comfort, ddc, final); samples
    without a metric-cache entry (has=False) are skipped. NC includes the factorized slot risk when present."""
    if not has.any():
        return {"score": logits["comfort"].sum() * 0}
    lg = {k: v[has].float() for k, v in logits.items()}
    y = pdm[has].float()
    two = lambda v: torch.where(v == 0.5, torch.zeros_like(v), v)
    bce = F.binary_cross_entropy_with_logits
    out = {}
    lp = F.logsigmoid(lg["no_at_fault_collisions"])
    if "slot_risk" in lg:
        lp = lp + F.logsigmoid(-lg["slot_risk"]).sum(-1)
    yn = two(y[..., 0])
    out["nc"] = -(yn * lp + (1 - yn) * log1mexp(lp)).mean()
    out["dac"] = bce(lg["drivable_area_compliance"], y[..., 1])
    out["ep"] = bce(lg["ego_progress"], y[..., 2])
    m = (y[..., 3] != 2.0).float()
    out["ttc"] = (bce(lg["time_to_collision_within_bound"], y[..., 3], reduction="none") * m).sum() / m.sum().clamp(min=1)
    out["comfort"] = bce(lg["comfort"], y[..., 4])
    out["ddc"] = bce(lg["driving_direction_compliance"], two(y[..., 5]))
    if "slot_risk" in lg and risk_y is not None:
        out["risk"] = bce(lg["slot_risk"], risk_y[has])
    return out


# ---------------------------------------------------------------- v2 (docs/METHOD_RATIONALE.md §3.3, §3.5)

LABEL_KEYS = ["sub", "ep_raw", "hcomf", "flags", "fault", "ttc_pair", "ttc_ok", "fault_unmatched"]


def collate_labels(results, P, M):
    """Per-sample outputs of pdm.proposal_labels (None where missing) -> dict of (B,P,...) tensors + has (B,).
    fault / ttc_pair (P, n_tracks) are padded to the dataset's M_MAX GT slots (the tracks are the GT boxes in order)."""
    B = len(results)
    lab = {"sub": torch.zeros(B, P, 7), "ep_raw": torch.zeros(B, P), "hcomf": torch.zeros(B, P),
           "flags": torch.zeros(B, P, 8, 4), "fault": torch.zeros(B, P, M), "ttc_pair": torch.zeros(B, P, M),
           "ttc_ok": torch.zeros(B, P, dtype=torch.bool), "fault_unmatched": torch.zeros(B, P)}
    has = torch.zeros(B, dtype=torch.bool)
    for i, r in enumerate(results):
        if r is None:
            continue
        has[i] = True
        for k in ("sub", "ep_raw", "hcomf", "flags", "ttc_ok", "fault_unmatched"):
            lab[k][i] = torch.as_tensor(np.asarray(r[k])).to(lab[k].dtype)
        n = r["fault"].shape[1]
        lab["fault"][i, :, :n] = torch.from_numpy(r["fault"].astype(np.float32))
        lab["ttc_pair"][i, :, :n] = torch.from_numpy(r["ttc_pair"].astype(np.float32))
    return lab, has


def pair_labels(y, match):
    """y (B,P,M) per GT track -> (B,P,S) per slot through the slot -> GT matching; mask (B,1,S) = matched slots.
    Unmatched slots get no label (neither 'safe' nor 'unsafe')."""
    i = match.clamp(min=0)[:, None].expand(-1, y.shape[1], -1)
    return y.gather(2, i), (match >= 0)[:, None]


def wbce(logit, y, w):
    """Weighted mean BCE; w broadcasts to the logit shape (0 = no label)."""
    l = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    w = w.expand_as(l).float()
    return (l * w).sum() / w.sum().clamp(min=1)


def score_loss_v2(lg, lab, has, match, pw=None):
    """Structured readout: BCE per head on the v2 labels (pdm.proposal_labels). Context heads on proposal labels (DDC's 0.5
    counts as 0, as DrivoR); EP on the raw-progress label; comfort on history comfort; area flags per pose; pair heads (when
    the readout has them) per (proposal, matched slot) on PDM's at-fault and TTC outcome for that track.
    pw (B,P): per-proposal label weight (0 = masked, the L2 holdout of the CG protocol); None = all 1."""
    if not has.any():
        return {"dac": lg["dac"].sum() * 0}
    two = lambda v: torch.where(v == 0.5, torch.zeros_like(v), v)
    g = lambda k: lg[k][has].float()
    y = {k: v[has] for k, v in lab.items()}
    w = torch.ones_like(y["ep_raw"]) if pw is None else pw[has].float()                  # B',P
    out = {"dac": wbce(g("dac"), y["sub"][..., 1], w), "ddc": wbce(g("ddc"), two(y["sub"][..., 5]), w),
           "ep": wbce(g("ep"), y["ep_raw"], w), "comfort": wbce(g("comfort"), y["hcomf"], w),
           "area": wbce(g("area"), y["flags"], w[..., None, None])}
    if "pair_fault" not in lg:
        return out
    m = match[has]
    fy, fm = pair_labels(y["fault"], m)
    ty, tm = pair_labels(y["ttc_pair"], m)
    tm = tm & y["ttc_ok"][..., None]
    fm, tm = fm.expand_as(fy).float() * w[..., None], tm.float() * w[..., None]
    out["pair_fault"] = wbce(g("pair_fault"), fy, fm)
    out["pair_ttc"] = wbce(g("pair_ttc"), ty, tm)
    return out


def score_loss_mono(lg, lab, has, prefix="mono_", pw=None):
    """Monolithic readouts (keys prefixed `prefix`: 'mono_', or 'pmono_' for the pair-mono readout): proposal-level BCE
    with the same label definitions as the structured readout. NC keeps PDM's value as a soft target (0.5 for a collision
    with a non-agent object, as the structured selection's v_k = 0.5); DDC's 0.5 counts as 0 in every readout; TTC is the
    proposal's PDM value; EP raw progress; history comfort. pw (B,P): per-proposal label weight, as score_loss_v2."""
    if not has.any():
        return {f"{prefix}comfort": lg["comfort"].sum() * 0}
    two = lambda v: torch.where(v == 0.5, torch.zeros_like(v), v)
    g = lambda k: lg[k][has].float()
    s = lab["sub"][has]
    w = torch.ones_like(s[..., 0]) if pw is None else pw[has].float()
    m = (s[..., 3] != 2.0).float() * w
    out = {"nc": wbce(g("no_at_fault_collisions"), s[..., 0], w),
           "dac": wbce(g("drivable_area_compliance"), s[..., 1], w),
           "ep": wbce(g("ego_progress"), lab["ep_raw"][has], w), "comfort": wbce(g("comfort"), lab["hcomf"][has], w),
           "ddc": wbce(g("driving_direction_compliance"), two(s[..., 5]), w),
           "ttc": wbce(g("time_to_collision_within_bound"), s[..., 3].clamp(max=1), m)}
    return {prefix + k: v for k, v in out.items()}


@torch.no_grad()
def det_recall(slot_box, slot_keep, slot_cls, gt_box, gt_mask, gt_cls, max_dist=2.0):
    """Recall of the GT boxes (inside the perception area) by the selected slots, class-aware gated Hungarian:
    (matched per class (N_CLS,), GT per class (N_CLS,)) summed over the batch (warm-up stopping criterion)."""
    m = match_slots(slot_box, slot_keep, gt_box, gt_mask, max_dist, slot_cls, gt_cls).cpu()
    gc, gm = gt_cls.cpu(), gt_mask.cpu()
    hit = torch.zeros_like(gm)
    for b in range(len(m)):
        j = m[b][m[b] >= 0]
        hit[b, j] = True
    tp = torch.stack([(hit & (gc == c)).sum() for c in range(N_CLS)])
    n = torch.stack([(gm & (gc == c)).sum() for c in range(N_CLS)])
    return tp, n


def ego_pose_feats(fut_ego):
    """Logged ego pose at each horizon in the t0 frame (B,H,4,4) -> (B,H,4): x / 10, y / 10, sin, cos of the yaw."""
    yaw = torch.atan2(fut_ego[..., 1, 0], fut_ego[..., 0, 0])
    return torch.stack([fut_ego[..., 0, 3] / 10, fut_ego[..., 1, 3] / 10, torch.sin(yaw), torch.cos(yaw)], -1)


def reg_view_loss(reg_view, reg_fut, tgt_reg, fut_ego):
    """Future-register JEPA through the register view head: V_R(register forecast, logged ego pose at the horizon) vs the
    EMA registers of the logged future frame (the forecast itself has no action input; METHOD_RATIONALE §3.3)."""
    pose = ego_pose_feats(fut_ego.float())[:, :, None].expand(-1, -1, reg_fut.shape[2], -1)
    pred = reg_view(torch.cat([reg_fut.float(), pose], -1))
    return F.smooth_l1_loss(pred, tgt_reg)


def box_geometry(box):
    """(...,7) x, y, z, l, w, h, yaw -> (...,9) normalized features (view-head input)."""
    return torch.cat([box[..., 0:1] / 32, box[..., 1:2] / 32, box[..., 2:3] / 4, box[..., 3:6].clamp(min=0.05).log(),
                      torch.sin(box[..., 6:7]), torch.cos(box[..., 6:7]),
                      torch.hypot(box[..., 0], box[..., 1])[..., None] / 32], -1)


def view_loss(view_head, lat, tgt, match, fut_box):
    """Agent-latent JEPA through the view head (METHOD_RATIONALE §3.3): V(h_k(h), box of the matched track at horizon h
    in that frame's ego frame, i.e. seen from the LOGGED ego) vs the EMA slot feature at that box."""
    H = lat.shape[1]
    gb = torch.stack([gather_m(fut_box[:, h], match) for h in range(H)], 1)  # B,H,S,7
    pred = view_head(torch.cat([lat.float(), box_geometry(gb.float())], -1))
    ts = torch.stack([gather_m(tgt["slot"][:, h], match) for h in range(H)], 1)
    ok = torch.stack([gather_m(tgt["slot_ok"][:, h], match) for h in range(H)], 1) & (match >= 0)[:, None]
    l = F.smooth_l1_loss(pred, ts, reduction="none").mean(-1)
    return (l * ok).sum() / ok.sum().clamp(min=1)


def phys_loss(phys, slot_xy, match, gt_fut, gt_fut_ok):
    """Physical forecast (B,S,8,4: dx, dy from the slot position, sin, cos of the yaw) vs the GT future of the matched
    track (t0 ego frame)."""
    f = gather_m(gt_fut, match)  # B,S,8,3
    tgt = torch.cat([f[..., :2] - slot_xy[:, :, None], torch.sin(f[..., 2:3]), torch.cos(f[..., 2:3])], -1)
    ok = gather_m(gt_fut_ok, match) & (match >= 0)[..., None]
    l = F.smooth_l1_loss(phys.float(), tgt, reduction="none")
    l = l[..., :2].sum(-1) + l[..., 2:].sum(-1)
    return (l * ok).sum() / ok.sum().clamp(min=1)
