"""HGO model: shared encoder, perception heads, budgeted state (registers + slots), action-conditioned JEPA world model,
DrivoR-style proposal planner and a scorer that reads imagined futures (docs/METHOD.md).

Pilot variants (R64 / O64 / H32) differ only in (n_reg, n_slot). The v2 design (docs/METHOD_RATIONALE.md §3, presets V2,
V2mono, R64v2) is selected by config fields, so every ablation is a flag:
  wm_kind="v2"     world model run once per scene over the whole state: future register latents, agent latents and the
                   slots' physical forecast (no action input; NAVSIM log-replay: reaction term = 0); used by the readouts
  readout="v2"     structured scorer: context heads read (proposal, registers, register forecast); per-pair heads read (pair geometry, slot,
                   agent latent, predicted context flags); no attention across proposals
  readout="mono"   monolithic scorer on the same state and futures
  readout="both"   both scorers trained with all their losses on one shared model; `select` ("v2" | "mono" | "pmono")
                   picks the rule used for the trajectory. Test 1 compares the rules on the same trained state and forecasts.
  pair_mono        (readout="both", slots) a third, monolithic readout over the structured readout's per-pair features:
                   mono -> pmono adds the per-pair relative features, pmono -> v2 replaces the learned combination over
                   slots by the fixed product / min
  role_probes      register probe -> map classes, slot probe -> objects
v3 (docs/METHOD.md §4, presets V3 / V3pmono / V3mono = V2 + the fields below; each is an ablation flag):
  grid="v3"        perception area x in [-32, 48] m, y in [-32, 32] m; heatmap and boxes on 0.5 m cells (160 x 128), so
                   two objects ~1 m apart give two peaks (the 1 m grid merged 36 % of the hazardous pedestrians, §7)
  prev_frame       the four t-0.5 s images go through the same encoder (no gradient unless prev_grad), their patch
                   grid is average-pooled (prev_pool) and embedded with the rays of the t-0.5 s cameras expressed in the
                   t0 ego frame (ego-motion compensation), and the BEV decoder cross-attends [patches t ; patches t-1]
  slot_hist        the BEV decoder also decodes the t-1 patches alone (detections of t-1 in the t0 frame, supervised by
                   the t-1 GT boxes); every slot is matched to the nearest t-1 detection of its class around its
                   constant-velocity prior, and its 0.5 s displacement enters the slot token
  n_reg=64         16 scene tokens per camera
"""

import copy
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from ade2e_wm.agents.drivor.layers.image_encoder.dinov2_lora import LoRA_ViT_timm, timm_ViT
from ade2e_wm.agents.drivor.layers.image_encoder.grid_mask import GridMask

from .geometry import N_CLS, project_boxes_t

N_CAM = 4
BEV_RES, BEV_RANGE = 64, 32.0      # pilot perception grid: 64 x 64 cells of 1 m over |x|, |y| <= 32 m (row = x, col = y);
                                   # BEV_RANGE also normalizes slot geometry. The BEV semantics cache uses this grid.
BEV_CLASSES = 7                    # 0 none, 1 road, 2 walkway, 3 centerline, 4 static object, 5 vehicle, 6 pedestrian


@dataclass(frozen=True)
class Grid:
    """Perception grid in the t0 ego frame: rows along x (forward), columns along y (left), square cells."""
    x0: float
    x1: float
    y0: float
    y1: float
    cell: float

    @property
    def rows(self):
        return int(round((self.x1 - self.x0) / self.cell))

    @property
    def cols(self):
        return int(round((self.y1 - self.y0) / self.cell))

    @property
    def area(self):
        return (self.x0, self.x1, self.y0, self.y1)


GRIDS = {"pilot": Grid(-BEV_RANGE, BEV_RANGE, -BEV_RANGE, BEV_RANGE, 2 * BEV_RANGE / BEV_RES),
         "v3": Grid(-32.0, 48.0, -32.0, 32.0, 0.5)}
PROBE_GRID = GRIDS["pilot"]        # state probes decode on the pilot grid in every variant (BEV cache, comparability)
SUB_SCORES = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound",
              "comfort", "driving_direction_compliance"]  # order of DrivoR's get_scores columns 0..5 (6 = final)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


@dataclass
class Config:
    n_reg: int = 32
    n_slot: int = 32
    d: int = 256
    ffn: int = 1024
    heads: int = 8
    image_wh: tuple = (1148, 672)
    vit_name: str = "timm/vit_small_patch14_reg4_dinov2.lvd142m"
    vit_weights: str = ""
    lora_rank: int = 32
    grid_mask: bool = True
    bev_layers: int = 2
    det_topk: int = 100
    det_thr: float = 0.3
    horizons: tuple = (2, 4, 8)           # JEPA horizons in 0.5 s frames (1, 2, 4 s)
    wm_layers: int = 4
    ema: float = 0.996
    n_prop: int = 64
    n_poses: int = 8
    ref_num: int = 4
    scorer_layers: int = 4
    slot_risk: bool = True
    score_w: dict = field(default_factory=lambda: dict(noc=10.0, dac=13.0, ddc=6.0, ttc=14.0, ep=15.0, comfort=2.0))
    # v2 (docs/METHOD_RATIONALE.md §3); the defaults reproduce the pilot (H32 checkpoints load unchanged)
    wm_kind: str = "h32"          # "h32" | "v2"
    readout: str = "h32"          # "h32" | "v2" | "mono" | "both"
    select: str = "v2"            # readout="both": selection rule, "v2" (structured), "mono" or "pmono"
    role_probes: bool = False
    latent_in_scorer: bool = True  # v2 readout: pair heads read the agent latent (ablation (c))
    pair_flags: bool = True        # v2 readout: pair heads read the predicted context flags a(p)
    pair_mono: bool = False        # readout="both" + slots: also a monolithic readout over the per-pair features (test 1)
    # v3 (docs/METHOD.md §4); the defaults reproduce V2 and the pilot (their checkpoints load unchanged)
    grid: str = "pilot"            # perception grid: "pilot" (64 x 64, 1 m, |x|, |y| <= 32 m) | "v3" (160 x 128, 0.5 m)
    prev_frame: bool = False       # fuse the t-0.5 s patches into the BEV decoder memory (ego-motion compensated rays)
    prev_pool: int = 2             # average pooling of the t-1 patch grid (memory size)
    prev_grad: bool = False        # back-propagate through the t-1 encoder pass (default: shared weights, no gradient)
    slot_hist: bool = False        # t-1 detections -> 0.5 s displacement of every slot (needs prev_frame)
    hist_gate: float = 2.0         # m, around the slot's constant-velocity position at t-0.5 s

    @property
    def n_state(self):
        return self.n_reg + self.n_slot

    @property
    def perc_grid(self):
        return GRIDS[self.grid]


V2 = dict(wm_kind="v2", slot_risk=False)
V3 = dict(V2, n_reg=64, n_slot=32, readout="both", role_probes=True, pair_mono=True, grid="v3", prev_frame=True,
          slot_hist=True)
PRESETS = {"R64": dict(n_reg=64, n_slot=0), "O64": dict(n_reg=0, n_slot=64), "H32": dict(n_reg=32, n_slot=32),
           "H16": dict(n_reg=16, n_slot=48), "H48": dict(n_reg=48, n_slot=16),
           # V2, V2pmono and V2mono train the same model (three readouts, all losses); only the selection rule differs
           "V2": dict(n_reg=32, n_slot=32, readout="both", select="v2", role_probes=True, pair_mono=True, **V2),
           "V2pmono": dict(n_reg=32, n_slot=32, readout="both", select="pmono", role_probes=True, pair_mono=True, **V2),
           "V2mono": dict(n_reg=32, n_slot=32, readout="both", select="mono", role_probes=True, pair_mono=True, **V2),
           # register-only: no pair terms; the structured scorer keeps its context / area / comfort heads as auxiliaries
           "R64v2": dict(n_reg=64, n_slot=0, readout="both", select="mono", **V2),
           # v3: one model, three selection rules (as V2)
           "V3": dict(V3, select="v2"), "V3pmono": dict(V3, select="pmono"), "V3mono": dict(V3, select="mono")}


def make_config(variant="H32", **kw):
    c = Config(**{**PRESETS.get(variant, {}), **kw})
    assert c.n_reg % N_CAM == 0, "registers are split evenly over the four cameras"
    assert c.wm_kind in ("h32", "v2") and c.readout in ("h32", "v2", "mono", "both")
    assert c.select in ("v2", "mono", "pmono")
    assert (c.wm_kind == "v2") == (c.readout != "h32"), "the v2 readouts need the v2 world model and vice versa"
    assert c.n_slot > 0 or (c.readout != "v2" and not (c.readout == "both" and c.select == "v2")), \
        "the structured selection needs slots"
    assert not c.pair_mono or (c.n_slot > 0 and c.readout == "both"), "the pair-mono readout needs slots and readout='both'"
    assert c.select != "pmono" or c.pair_mono, "select='pmono' needs pair_mono=True"
    assert c.grid in GRIDS, c.grid
    assert not c.slot_hist or (c.prev_frame and c.n_slot), "slot history needs prev_frame and slots"
    g = c.perc_grid
    assert g.rows % 4 == 0 and g.cols % 4 == 0
    return c


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class MHA(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x, mem, keep=None):
        """keep: (B, Nk) bool, True = attend. Every row must keep at least one key."""
        B, Nq, D = x.shape
        Nk = mem.shape[1]
        q = self.q(x).view(B, Nq, self.h, -1).transpose(1, 2)
        k = self.k(mem).view(B, Nk, self.h, -1).transpose(1, 2)
        v = self.v(mem).view(B, Nk, self.h, -1).transpose(1, 2)
        m = None if keep is None else keep[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return self.o(y.transpose(1, 2).reshape(B, Nq, D))


class Block(nn.Module):
    """Pre-norm block: self-attention (optional), cross-attention (optional), FFN."""

    def __init__(self, d, heads, ffn, self_attn=True, cross=False):
        super().__init__()
        self.sa = MHA(d, heads) if self_attn else None
        self.ca = MHA(d, heads) if cross else None
        self.n1, self.n2, self.n3, self.nm = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = mlp(d, ffn, d)

    def forward(self, x, keep=None, mem=None, mem_keep=None):
        if self.sa is not None:
            h = self.n1(x)
            x = x + self.sa(h, h, keep)
        if self.ca is not None:
            x = x + self.ca(self.n2(x), self.nm(mem), mem_keep)
        return x + self.ff(self.n3(x))


# ---------------------------------------------------------------- encoder

class Encoder(nn.Module):
    """DINOv2 ViT-S/14-reg4 + LoRA, with n_reg / 4 learned scene tokens per camera (DrivoR's register mechanism)."""

    def __init__(self, c: Config):
        super().__init__()
        W, H = c.image_wh
        overlay = {"file": c.vit_weights} if c.vit_weights else None
        vit = timm.create_model(c.vit_name, pretrained=bool(c.vit_weights), pretrained_cfg_overlay=overlay,
                                img_size=(H, W), num_classes=0)
        vit.__class__ = timm_ViT
        self.n_prefix = 1 + vit.num_reg_tokens
        self.width = vit.num_features
        self.vit = LoRA_ViT_timm(vit, r=c.lora_rank)
        self.per_cam = c.n_reg // N_CAM
        self.scene = nn.Parameter(torch.randn(1, N_CAM, self.per_cam, self.width) * 1e-6) if self.per_cam else None
        self.neck = nn.Linear(self.width, c.d)
        self.grid = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7) if c.grid_mask else None
        self.hw = (H // 14, W // 14)

    def forward(self, img_u8):
        """img_u8 (B,4,3,H,W) uint8 -> registers (B, n_reg, d), patch grid (B*4, width, h, w)."""
        B = img_u8.shape[0]
        x = (img_u8.float() / 255.0 - MEAN.to(img_u8.device)) / STD.to(img_u8.device)
        x = x.flatten(0, 1)
        if self.grid is not None and self.training:
            x = self.grid(x)
        scene = self.scene.expand(B, -1, -1, -1).flatten(0, 1) if self.scene is not None else None
        tok = self.vit(x, scene)
        n = self.per_cam
        reg = self.neck(tok[:, :n]).reshape(B, N_CAM * n, self.neck.out_features)
        h, w = self.hw
        grid = tok[:, n + self.n_prefix:].reshape(B * N_CAM, h, w, self.width).permute(0, 3, 1, 2)
        return reg, grid


def patch_rays(cam2ego, K, hw, patch=14):
    """Unit ray directions in the ego frame for every patch centre: (B, 4*h*w, 3), plus camera centres (B, 4*h*w, 3)."""
    B = cam2ego.shape[0]
    h, w = hw
    v, u = torch.meshgrid(torch.arange(h, device=K.device), torch.arange(w, device=K.device), indexing="ij")
    pix = torch.stack([(u + 0.5) * patch, (v + 0.5) * patch, torch.ones_like(u, dtype=torch.float)], -1).float()
    d = torch.einsum("bcij,pj->bcpi", torch.linalg.inv(K), pix.view(-1, 3))
    d = torch.einsum("bcij,bcpj->bcpi", cam2ego[..., :3, :3], d)
    d = F.normalize(d, dim=-1)
    o = cam2ego[..., None, :3, 3].expand_as(d)
    return d.reshape(B, -1, 3), o.reshape(B, -1, 3)


# ---------------------------------------------------------------- perception

class BEVDecoder(nn.Module):
    """Grid of BEV queries cross-attending a memory, upsampled by `up` (power of 2) to the output grid (rows x cols).
    Heads: BEV semantics (7), CenterNet heatmap (N_CLS), box regression (dx, dy, log l, log w, z, log h, sin, cos, vx, vy).
    Pilot: perception head 32 x 32 queries -> 64 x 64; probes 16 x 16 -> 64 x 64. v3 perception: 40 x 32 -> 160 x 128."""

    def __init__(self, c: Config, up, layers, grid: Grid = PROBE_GRID):
        super().__init__()
        assert grid.rows % up == 0 and grid.cols % up == 0 and up & (up - 1) == 0
        self.gx, self.gy = gx, gy = grid.rows // up, grid.cols // up
        self.q = nn.Parameter(torch.randn(gx * gy, c.d) * 0.02)
        ys, xs = torch.meshgrid(torch.arange(gx), torch.arange(gy), indexing="ij")
        self.register_buffer("cell", torch.stack([(ys + 0.5) / gx * 2 - 1, (xs + 0.5) / gy * 2 - 1], -1).view(-1, 2).float())
        self.cell_in = mlp(2, c.d, c.d)
        self.blocks = nn.ModuleList([Block(c.d, c.heads, c.ffn, self_attn=True, cross=True) for _ in range(layers)])
        layers_up = []
        ch, f = c.d, 1
        while f < up:
            layers_up += [nn.ConvTranspose2d(ch, ch // 2, 2, 2), nn.GroupNorm(8, ch // 2), nn.GELU()]
            ch, f = ch // 2, f * 2
        self.up = nn.Sequential(*layers_up, nn.Conv2d(ch, ch, 3, padding=1), nn.GELU())
        self.seg = nn.Conv2d(ch, BEV_CLASSES, 1)
        self.hm = nn.Conv2d(ch, N_CLS, 1)
        self.reg = nn.Conv2d(ch, 10, 1)
        nn.init.constant_(self.hm.bias, -2.19)  # CenterNet prior 0.1

    def forward(self, mem, mem_keep=None):
        B = mem.shape[0]
        x = (self.q + self.cell_in(self.cell))[None].expand(B, -1, -1)
        for blk in self.blocks:
            x = blk(x, None, mem, mem_keep)
        x = self.up(x.transpose(1, 2).reshape(B, -1, self.gx, self.gy))
        return {"seg": self.seg(x), "hm": self.hm(x), "reg": self.reg(x)}


def decode_boxes(out, topk, thr, grid: Grid = PROBE_GRID):
    """CenterNet decoding on `grid` -> boxes (B,K,7) ego frame, vel (B,K,2), cls (B,K), score (B,K) (score < thr: 0).
    Peaks: 3 x 3 max-pool, so two peaks are >= 2 cells apart (2 m on the pilot grid, 1 m on the v3 grid)."""
    hm = out["hm"].float().sigmoid()
    B, C, Rx, Ry = hm.shape
    assert (Rx, Ry) == (grid.rows, grid.cols), (hm.shape, grid)
    peak = hm * (F.max_pool2d(hm, 3, 1, 1) == hm)
    score, idx = peak.flatten(1).topk(topk)
    cls = idx // (Rx * Ry)
    pix = idx % (Rx * Ry)
    iy, ix = pix // Ry, pix % Ry
    reg = out["reg"].float().flatten(2).gather(2, pix[:, None].expand(-1, 10, -1)).transpose(1, 2)  # B,K,10
    cell = grid.cell
    x = grid.x0 + (iy.float() + 0.5 + reg[..., 0]) * cell
    y = grid.y0 + (ix.float() + 0.5 + reg[..., 1]) * cell
    l, w, h = reg[..., 2].clamp(-3, 4).exp(), reg[..., 3].clamp(-3, 3).exp(), reg[..., 5].clamp(-3, 3).exp()
    yaw = torch.atan2(reg[..., 6], reg[..., 7])
    box = torch.stack([x, y, reg[..., 4], l, w, h, yaw], -1)
    return box, reg[..., 8:10], cls, score.masked_fill(score < thr, 0.0)


# ---------------------------------------------------------------- slots

class SlotEncoder(nn.Module):
    """slot = RoIAlign of the patch grid at the box projected into the camera where it is largest (+ no-vis token)
    plus an embedding of the box geometry, class and score."""

    def __init__(self, c: Config, width):
        super().__init__()
        self.feat = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, c.d))
        self.geom = mlp(10 + N_CLS + 1, c.d, c.d)
        self.novis = nn.Parameter(torch.zeros(c.d))
        self.wh = c.image_wh

    def forward(self, grid, box, vel, cls, score, cam2ego, K):
        """grid (B*4, width, h, w); box (B,S,7) ego frame; returns (B,S,d)."""
        B, S, _ = box.shape
        rois, area = project_boxes_t(box.detach().float(), cam2ego.float(), K.float(), self.wh)  # B,S,4,4 / B,S,4
        best = area.argmax(-1)
        vis = area.amax(-1) > 0
        r = rois.gather(2, best[..., None, None].expand(-1, -1, 1, 4)).squeeze(2)  # B,S,4
        bidx = (torch.arange(B, device=box.device)[:, None] * N_CAM + best).float()
        f = roi_align(grid.float(), torch.cat([bidx.reshape(-1, 1), r.reshape(-1, 4)], 1), (2, 2),
                      spatial_scale=1.0 / 14, sampling_ratio=2, aligned=True).mean((2, 3)).view(B, S, -1)
        app = torch.where(vis[..., None], self.feat(f.to(grid.dtype)), self.novis.to(grid.dtype).expand(B, S, -1))
        g = torch.cat([box[..., 0:1] / BEV_RANGE, box[..., 1:2] / BEV_RANGE, box[..., 2:3] / 4,
                       box[..., 3:6].clamp(min=0.05).log(), torch.sin(box[..., 6:7]), torch.cos(box[..., 6:7]),
                       vel / 10, F.one_hot(cls.clamp(min=0), N_CLS).float(), score[..., None]], -1)
        return app + self.geom(g.to(app.dtype)), vis


HIST_FEATS = 6     # matched flag, 0.5 s displacement dx, dy (/5 m), sin / cos of the yaw change, t-1 detection score


def match_prev(box, vel, cls, keep, pbox, pcls, pscore, gate, dt=0.5):
    """Slot history (v3): every kept slot -> the nearest valid t-1 detection (score > 0, same class) within `gate` m of
    the slot's constant-velocity position at t - dt (greedy; one t-1 detection may serve several slots). All boxes are in
    the t0 ego frame. Returns feats (B,S,HIST_FEATS), zero when unmatched, and the matched mask (B,S)."""
    prior = box[..., :2].float() - dt * vel.float()
    d = torch.cdist(prior, pbox[..., :2].float())                                   # B,S,K
    ok = (cls[:, :, None] == pcls[:, None]) & (pscore[:, None] > 0) & keep[..., None]
    dmin, j = d.masked_fill(~ok, float("inf")).min(-1)
    m = dmin <= gate
    pb = pbox.float().gather(1, j[..., None].expand(-1, -1, pbox.shape[-1]))
    dyaw = box[..., 6].float() - pb[..., 6]
    f = torch.cat([torch.ones_like(dyaw)[..., None], (box[..., :2].float() - pb[..., :2]) / 5.0,
                   torch.sin(dyaw)[..., None], torch.cos(dyaw)[..., None], pscore.float().gather(1, j)[..., None]], -1)
    return f * m[..., None], m


def select_slots(box, vel, cls, score, n):
    """Deterministic selection: valid detections (score > 0 after the threshold), nearest first, at most n."""
    dist = torch.hypot(box[..., 0], box[..., 1]).masked_fill(score <= 0, float("inf"))
    order = dist.argsort(1)[:, :n]
    take = lambda t: t.gather(1, order.view(*order.shape, *([1] * (t.dim() - 2))).expand(-1, -1, *t.shape[2:]))
    keep = torch.isfinite(dist.gather(1, order))
    return take(box), take(vel), take(cls), take(score), keep


# ---------------------------------------------------------------- world model, planner, scorer

def traj_feats(traj):
    """(...,8,3) x, y, yaw -> (...,8,4) x/10, y/10, cos, sin."""
    return torch.cat([traj[..., :2] / 10.0, torch.cos(traj[..., 2:3]), torch.sin(traj[..., 2:3])], -1)


class WorldModel(nn.Module):
    """Action-conditioned predictor over [ego; action tokens; state now; state queries per horizon].
    A query is its token's current state + a horizon embedding, so every slot keeps its identity in the imagined
    future; slots share all weights and interact through attention."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        H = len(c.horizons)
        self.act = mlp(4, c.d, c.d)
        self.t_emb = nn.Parameter(torch.randn(c.n_poses, c.d) * 0.02)
        self.h_emb = nn.Parameter(torch.randn(H, c.d) * 0.02)
        self.now = nn.Parameter(torch.randn(c.d) * 0.02)
        self.blocks = nn.ModuleList([Block(c.d, c.heads, c.ffn) for _ in range(c.wm_layers)])
        self.norm = nn.LayerNorm(c.d)
        self.out_reg = nn.Linear(c.d, c.d)
        self.out_slot = nn.Linear(c.d, c.d)
        self.motion = mlp(c.d, c.d, c.n_poses * 2)

    def forward(self, state, keep, ego, traj):
        """state (B,N,d), keep (B,N), ego (B,d), traj (B,8,3) -> fut (B,H,N,d), motion offsets (B,N,8,2)."""
        B, N, D = state.shape
        H = len(self.c.horizons)
        a = self.act(traj_feats(traj)) + self.t_emb
        q = (state[:, None] + self.h_emb[None, :, None]).reshape(B, H * N, D)
        x = torch.cat([ego[:, None], a, state + self.now, q], 1)
        k = torch.cat([torch.ones(B, 1 + a.shape[1], dtype=torch.bool, device=state.device), keep, keep.repeat(1, H)], 1)
        for blk in self.blocks:
            x = blk(x, k)
        x = self.norm(x)
        s0 = 1 + a.shape[1]
        cur, fut = x[:, s0:s0 + N], x[:, s0 + N:].reshape(B, H, N, D)
        nr = self.c.n_reg
        fut = torch.cat([self.out_reg(fut[:, :, :nr]), self.out_slot(fut[:, :, nr:])], 2)
        return fut, self.motion(cur[:, nr:]).view(B, N - nr, self.c.n_poses, 2)


class Planner(nn.Module):
    """DrivoR's proposal decoder: n_prop learned queries + ego token, refined over the state; one head per refinement."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.init = nn.Embedding(c.n_prop, c.d)
        self.blocks = nn.ModuleList([Block(c.d, c.heads, c.ffn, cross=True) for _ in range(c.ref_num)])
        self.heads = nn.ModuleList([mlp(c.d, c.ffn, c.n_poses * 3) for _ in range(c.ref_num + 1)])

    def forward(self, mem, mem_keep, ego):
        B = mem.shape[0]
        q = ego[:, None] + self.init.weight[None]
        out = [self.heads[0](q).view(B, -1, self.c.n_poses, 3)]
        for i, blk in enumerate(self.blocks):
            q = blk(q, None, mem, mem_keep)
            out.append(self.heads[i + 1](q).view(B, -1, self.c.n_poses, 3))
        return out


class Scorer(nn.Module):
    """One token per proposal; self-attention across proposals, cross-attention to the current state and that proposal's
    imagined future; DrivoR's six sub-score heads. Slot variants add a per-slot collision risk (factorized readout)."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.embed = mlp(c.n_poses * 3, c.ffn, c.d)
        self.sa = nn.ModuleList([Block(c.d, c.heads, c.ffn) for _ in range(c.scorer_layers)])
        self.ca = nn.ModuleList([Block(c.d, c.heads, c.ffn, self_attn=False, cross=True) for _ in range(c.scorer_layers)])
        self.heads = nn.ModuleDict({k: mlp(c.d, c.ffn, 1) for k in SUB_SCORES})
        self.fut_type = nn.Parameter(torch.randn(len(c.horizons), c.d) * 0.02)
        self.risk = None
        if c.slot_risk and c.n_slot > 0:
            H = len(c.horizons)
            self.risk = mlp(c.d * (H + 2) + c.n_poses * 2 + c.n_poses * 4, c.ffn, 1)
            nn.init.constant_(self.risk[-1].bias, -4.0)  # prior: a slot is rarely hit (keeps the NC product near 1)

    def forward(self, props, state, keep, ego, fut, motion, slot_xy):
        """props (B,P,8,3); state (B,N,d); fut (B*P,H,N,d); motion (B*P,S,8,2) offsets; slot_xy (B,S,2)."""
        B, P = props.shape[:2]
        N, D = state.shape[1:]
        H = fut.shape[1]
        x = self.embed(props.flatten(2)) + ego[:, None]
        mem = torch.cat([ego[:, None], state, ], 1).repeat_interleave(P, 0)
        mem = torch.cat([mem, (fut + self.fut_type[None, :, None]).flatten(1, 2)], 1)
        mk = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=keep.device), keep], 1)
        mk = torch.cat([mk.repeat_interleave(P, 0), keep.repeat(1, H).repeat_interleave(P, 0)], 1)
        for sa, ca in zip(self.sa, self.ca):
            x = sa(x)
            x = ca(x.reshape(B * P, 1, D), None, mem, mk).view(B, P, D)
        logits = {k: h(x).squeeze(-1) for k, h in self.heads.items()}
        if self.risk is not None:
            nr, S = self.c.n_reg, self.c.n_slot
            sf = fut[:, :, nr:].permute(0, 2, 1, 3).reshape(B, P, S, H * D)  # imagined slot latents
            now = state[:, nr:][:, None].expand(B, P, S, D)
            mot = (motion + slot_xy.repeat_interleave(P, 0)[:, :, None]).view(B, P, S, -1) / 10.0
            pr = traj_feats(props).flatten(2)[:, :, None].expand(B, P, S, -1)
            r = self.risk(torch.cat([x[:, :, None].expand(B, P, S, D), sf, now, mot, pr], -1)).squeeze(-1)
            logits["slot_risk"] = r.masked_fill(~keep[:, None, nr:], -30.0)  # empty slots: no risk
        return logits


def selection_score(logits, w):
    """DrivoR's weighted log score; the NC term includes the factorized slot risk when present."""
    ls = F.logsigmoid
    nc = ls(logits["no_at_fault_collisions"])
    if "slot_risk" in logits:
        nc = nc + ls(-logits["slot_risk"]).sum(-1)
    sig = lambda k: logits[k].sigmoid()
    return (w["noc"] * nc + w["dac"] * ls(logits["drivable_area_compliance"])
            + w["ddc"] * ls(logits["driving_direction_compliance"])
            + (w["ttc"] * sig("time_to_collision_within_bound") + w["ep"] * sig("ego_progress")
               + w["comfort"] * sig("comfort")).clamp(min=1e-6).log())


# ---------------------------------------------------------------- v2 (docs/METHOD_RATIONALE.md §3.3, §3.5)

N_FLAGS = 4        # ego area along a proposal: multiple lanes, non-drivable area, oncoming traffic, intersection
STATIC_CLS = 3     # slot class of non-agent objects: a collision with them gives PDM no-at-fault-collision 0.5, not 0
PAIR_GEO = 6       # per pose: dx, dy in the ego frame of p(t), sin / cos relative yaw, distance, ego speed
EGO_POSE = 4       # logged ego pose at a JEPA horizon in the t0 frame: x / 10, y / 10, sin, cos of the yaw (view heads)


class StateWM(nn.Module):
    """World model run once per scene over the whole state (docs/METHOD_RATIONALE.md §3.3). Every token (register or
    slot) gets one query per JEPA horizon; shared weights; slots have no index embedding; relational (slot-slot) and scene
    (slot-register) context through attention. No action input: on NAVSIM log-replay the reaction term is 0.
    Outputs: future register latents (B,H,n_reg,d), agent latents (B,H,S,d), and the physical forecast of the slots
    (B,S,8,4: dx, dy from the current slot position in the t0 ego frame, unnormalized sin / cos of the yaw).
    The JEPA targets of both latents come from the LOGGED future views; view heads (training only) take the logged ego
    pose, so the forecasts themselves need no action and are used by the readouts at test time."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        H = len(c.horizons)
        self.h_emb = nn.Parameter(torch.randn(H, c.d) * 0.02)
        self.now = nn.Parameter(torch.randn(c.d) * 0.02)
        self.blocks = nn.ModuleList([Block(c.d, c.heads, c.ffn) for _ in range(c.wm_layers)])
        self.norm = nn.LayerNorm(c.d)
        self.out_reg = nn.Linear(c.d, c.d)
        self.out_slot = nn.Linear(c.d, c.d)
        self.phys = mlp(c.d, c.d, c.n_poses * 4) if c.n_slot else None

    def forward(self, state, keep, ego):
        c = self.c
        B, N, D = state.shape
        nr, S, H = c.n_reg, N - c.n_reg, len(c.horizons)
        q = (state[:, None] + self.h_emb[None, :, None]).reshape(B, H * N, D)
        x = torch.cat([ego[:, None], state + self.now, q], 1)
        k = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=state.device), keep, keep.repeat(1, H)], 1)
        for blk in self.blocks:
            x = blk(x, k)
        x = self.norm(x)
        fut = x[:, 1 + N:].reshape(B, H, N, D)
        reg_fut = self.out_reg(fut[:, :, :nr])
        lat = self.out_slot(fut[:, :, nr:]) if S else None
        phys = self.phys(x[:, 1 + nr:1 + N]).view(B, S, c.n_poses, 4) if S else None
        return reg_fut, lat, phys


def ego_speed(props):
    """(...,8,3) poses at 0.5 s -> (...,8) speed in m/s from consecutive poses (the first from the t0 origin)."""
    xy = props[..., :2]
    prev = torch.cat([torch.zeros_like(xy[..., :1, :]), xy[..., :-1, :]], -2)
    return (xy - prev).norm(dim=-1) / 0.5


def pair_geometry(props, slot_xy, phys):
    """Future of every slot relative to the ego driving every proposal. props (B,P,8,3) rear-axle poses in the t0 ego frame;
    slot_xy (B,S,2); phys (B,S,8,4) -> (B,P,S,8,PAIR_GEO): (dx, dy) of the slot centre in the frame of p(t) (/10 m),
    sin / cos of the slot yaw relative to the ego yaw, distance (/10 m), ego speed along p (/10 m/s)."""
    pos = slot_xy[:, :, None] + phys[..., :2]                                     # B,S,8,2
    yaw_a = torch.atan2(phys[..., 2], phys[..., 3])                               # B,S,8
    ex, ey, eyaw = props[..., 0][:, :, None], props[..., 1][:, :, None], props[..., 2][:, :, None]  # B,P,1,8
    dx, dy = pos[:, None, ..., 0] - ex, pos[:, None, ..., 1] - ey                 # B,P,S,8
    c, s = torch.cos(eyaw), torch.sin(eyaw)
    rx, ry = c * dx + s * dy, -s * dx + c * dy
    dyaw = yaw_a[:, None] - eyaw
    spd = ego_speed(props)[:, :, None].expand_as(rx)
    return torch.stack([rx / 10, ry / 10, torch.sin(dyaw), torch.cos(dyaw), torch.hypot(rx, ry) / 10, spd / 10], -1)


class StructuredScorer(nn.Module):
    """Structured readout. Context heads (dac, ddc, ep, area flags) read only the proposal, the current registers and the
    register forecast (+ ego token); the comfort head reads the proposal and the ego history; the pair heads (slot
    variants) read only the pair geometry, the slot (token, class, size, agent latent, physical forecast) and the predicted
    context flags along the proposal. No attention across proposals. Without slots it has only the context heads
    (auxiliary supervision of R64v2). It reads the same information as MonoScorer; only the routing and the composition
    of the score differ (test 1)."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        H = len(c.horizons)
        self.embed = mlp(c.n_poses * 4, c.ffn, c.d)
        self.ctx = nn.ModuleList([Block(c.d, c.heads, c.ffn, self_attn=False, cross=True) for _ in range(c.scorer_layers)])
        self.reg_fut_in = nn.Linear(H * c.d, c.d) if c.n_reg else None  # register forecast tokens, as in MonoScorer
        self.fut_type = nn.Parameter(torch.randn(1, c.d) * 0.02)
        self.g = nn.ModuleDict({k: mlp(c.d, c.ffn, 1) for k in ("dac", "ddc", "ep")})
        self.area = mlp(c.d, c.ffn, c.n_poses * N_FLAGS)
        self.comfort = mlp(c.n_poses * 4 + 4 * 11, c.ffn, 1)
        self.pairs = bool(c.n_slot)
        if self.pairs:
            self.lat_proj = nn.Linear(H * c.d, c.d) if c.latent_in_scorer else None
            self.pair_geo = nn.Linear(c.n_poses * PAIR_GEO, c.ffn)
            self.pair_slot = nn.Linear(c.d + 2 + N_CLS + (c.d if c.latent_in_scorer else 0), c.ffn)
            self.pair_flag = nn.Linear(c.n_poses * N_FLAGS, c.ffn) if c.pair_flags else None
            self.pair_out = nn.Sequential(nn.GELU(), nn.Linear(c.ffn, c.ffn), nn.GELU(), nn.Linear(c.ffn, 2))
            nn.init.constant_(self.pair_out[-1].bias, -4.0)  # prior: a pair rarely collides / violates TTC

    def forward(self, props, ego_tok, ego_raw, reg, reg_keep, slots=None, slot_keep=None, slot_box=None, slot_cls=None,
                lat=None, phys=None, reg_fut=None):
        c = self.c
        B, P = props.shape[:2]
        tf = traj_feats(props).flatten(2)
        z = self.embed(tf) + ego_tok[:, None]
        mem, mk = [ego_tok[:, None], reg], [torch.ones(B, 1, dtype=torch.bool, device=props.device), reg_keep]
        if self.reg_fut_in is not None and reg_fut is not None:  # (B,H,n_reg,d) -> one token per register
            mem.append(self.reg_fut_in(reg_fut.permute(0, 2, 1, 3).flatten(2).to(z.dtype)) + self.fut_type[0])
            mk.append(reg_keep)
        mem, mk = torch.cat(mem, 1), torch.cat(mk, 1)
        for blk in self.ctx:
            z = blk(z, None, mem, mk)
        out = {k: h(z).squeeze(-1) for k, h in self.g.items()}
        out["area"] = self.area(z).view(B, P, c.n_poses, N_FLAGS)
        out["comfort"] = self.comfort(torch.cat([tf, ego_raw.flatten(1)[:, None].expand(B, P, -1)], -1)).squeeze(-1)
        if not self.pairs:
            return out
        dt = z.dtype
        geo = pair_geometry(props.float(), slot_box[..., :2].float(), phys.float()).flatten(-2)  # B,P,S,8*PAIR_GEO
        sf = [slots.to(dt), slot_box[..., 3:5].float().clamp(min=0.05).log().to(dt),
              F.one_hot(slot_cls.clamp(min=0), N_CLS).to(dt)]
        if self.lat_proj is not None:
            sf.append(self.lat_proj(lat.permute(0, 2, 1, 3).flatten(2).to(dt)))
        h = self.pair_geo(geo.to(dt)) + self.pair_slot(torch.cat(sf, -1))[:, None]
        if self.pair_flag is not None:
            h = h + self.pair_flag(out["area"].detach().sigmoid().flatten(2))[:, :, None]
        pr = self.pair_out(h)
        empty = ~slot_keep[:, None]
        out["pair_fault"] = pr[..., 0].masked_fill(empty, -30.0)  # logit: at-fault collision with this slot
        out["pair_ttc"] = pr[..., 1].masked_fill(empty, -30.0)    # logit: TTC-rule violation caused by this slot
        return out


def structured_selection(lg, slot_cls, slot_keep, w):
    """Selection score of the structured readout and its per-proposal probabilities (B,P,6, SUB_SCORES order).
    NC is the expectation of PDM's no-at-fault-collision value under independent per-slot risks r_k. PDM takes the MIN over
    the at-fault collisions of 0 (agent) and 0.5 (non-agent object), so several non-agent collisions still give 0.5:
        E[NC] = prod_agents (1 - r_a) * (0.5 + 0.5 * prod_static (1 - r_s)).
    TTC pass = min over slots of (1 - u_k), i.e. AND (exact for certain violations; a design choice for uncertain ones);
    context and comfort terms as in DrivoR's weighted log score."""
    keep = slot_keep[:, None].float()
    static = (slot_cls == STATIC_CLS)[:, None].float()
    r = (lg["pair_fault"].float().sigmoid() * keep).clamp(max=1 - 1e-6)
    log_free_agent = torch.log1p(-r * (1 - static)).sum(-1)          # log prod_agents (1 - r_a)
    free_static = torch.log1p(-r * static).sum(-1).exp()             # prod_static (1 - r_s)
    log_nc = log_free_agent + torch.log(0.5 + 0.5 * free_static)
    ttc = (1 - lg["pair_ttc"].float().sigmoid()).masked_fill(keep == 0, 1.0).amin(-1)
    sig = lambda k: lg[k].float().sigmoid()
    ls = F.logsigmoid
    score = (w["noc"] * log_nc + w["dac"] * ls(lg["dac"].float()) + w["ddc"] * ls(lg["ddc"].float())
             + (w["ttc"] * ttc + w["ep"] * sig("ep") + w["comfort"] * sig("comfort")).clamp(min=1e-6).log())
    probs = torch.stack([log_nc.exp(), sig("dac"), sig("ep"), ttc, sig("comfort"), sig("ddc")], -1)
    return score, probs


class MonoScorer(nn.Module):
    """Monolithic readout (test 1): one token per proposal cross-attends [ego, state, register future tokens, slot future
    tokens (agent latent + physical forecast)] and predicts the six proposal-level sub-scores; the ego history (comfort
    input) is added to the token. Same information as the structured readout (current state, register forecast, agent
    latent and physical forecast), no per-pair terms, no routing, no attention across proposals."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        H = len(c.horizons)
        self.embed = mlp(c.n_poses * 4, c.ffn, c.d)
        self.hist = nn.Linear(4 * 11, c.d)
        self.ca = nn.ModuleList([Block(c.d, c.heads, c.ffn, self_attn=False, cross=True) for _ in range(c.scorer_layers)])
        self.heads = nn.ModuleDict({k: mlp(c.d, c.ffn, 1) for k in SUB_SCORES})
        self.reg_fut_in = nn.Linear(H * c.d, c.d) if c.n_reg else None
        self.fut_in = nn.Linear(H * c.d + c.n_poses * 4, c.d) if c.n_slot else None
        self.fut_type = nn.Parameter(torch.randn(2, c.d) * 0.02)

    def forward(self, props, ego_tok, ego_raw, state, keep, slot_box=None, lat=None, phys=None, reg_fut=None):
        c = self.c
        B, P = props.shape[:2]
        nr = c.n_reg
        z = self.embed(traj_feats(props).flatten(2)) + ego_tok[:, None] + self.hist(ego_raw.flatten(1))[:, None]
        mem, mk = [ego_tok[:, None], state], [torch.ones(B, 1, dtype=torch.bool, device=props.device), keep]
        if self.reg_fut_in is not None and reg_fut is not None:
            mem.append(self.reg_fut_in(reg_fut.permute(0, 2, 1, 3).flatten(2).to(z.dtype)) + self.fut_type[0])
            mk.append(keep[:, :nr])
        if self.fut_in is not None:
            pos = (slot_box[..., None, :2].float() + phys[..., :2].float()) / 10.0                # B,S,8,2
            ph = torch.cat([pos, F.normalize(phys[..., 2:4].float(), dim=-1)], -1).flatten(2)       # B,S,32
            ft = self.fut_in(torch.cat([lat.permute(0, 2, 1, 3).flatten(2).to(z.dtype), ph.to(z.dtype)], -1))
            mem.append(ft + self.fut_type[1])
            mk.append(keep[:, nr:])
        mem, mk = torch.cat(mem, 1), torch.cat(mk, 1)
        for blk in self.ca:
            z = blk(z, None, mem, mk)
        return {k: h(z).squeeze(-1) for k, h in self.heads.items()}


class PairMonoScorer(nn.Module):
    """Middle rung of test 1 (slot variants): monolithic readout over the structured readout's per-pair features. One token
    per proposal cross-attends [ego, registers, register forecast, one token per slot built from the same features as the
    structured pair heads: geometry of the slot's forecast relative to the ego along this proposal, slot token, size,
    class, agent latent] and learned heads predict the six sub-scores. Compared with MonoScorer it adds the per-pair
    relative features; compared with StructuredScorer the combination over slots is learned (attention + MLP) instead of
    the fixed product / min. The slot tokens differ per proposal, so every proposal has its own memory; no attention across
    proposals."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        H = len(c.horizons)
        self.embed = mlp(c.n_poses * 4, c.ffn, c.d)
        self.hist = nn.Linear(4 * 11, c.d)
        self.ca = nn.ModuleList([Block(c.d, c.heads, c.ffn, self_attn=False, cross=True) for _ in range(c.scorer_layers)])
        self.heads = nn.ModuleDict({k: mlp(c.d, c.ffn, 1) for k in SUB_SCORES})
        self.reg_fut_in = nn.Linear(H * c.d, c.d) if c.n_reg else None
        self.lat_proj = nn.Linear(H * c.d, c.d)
        self.pair_geo = nn.Linear(c.n_poses * PAIR_GEO, c.d)
        self.pair_slot = nn.Linear(c.d + 2 + N_CLS + c.d, c.d)
        self.fut_type = nn.Parameter(torch.randn(2, c.d) * 0.02)

    def forward(self, props, ego_tok, ego_raw, reg, reg_keep, slots, slot_keep, slot_box, slot_cls, lat, phys,
                reg_fut=None):
        B, P = props.shape[:2]
        z = self.embed(traj_feats(props).flatten(2)) + ego_tok[:, None] + self.hist(ego_raw.flatten(1))[:, None]
        dt = z.dtype
        ctx, ck = [ego_tok[:, None].to(dt), reg.to(dt)], [torch.ones(B, 1, dtype=torch.bool, device=props.device), reg_keep]
        if self.reg_fut_in is not None and reg_fut is not None:
            ctx.append(self.reg_fut_in(reg_fut.permute(0, 2, 1, 3).flatten(2).to(dt)) + self.fut_type[0])
            ck.append(reg_keep)
        ctx, ck = torch.cat(ctx, 1), torch.cat(ck, 1)                                            # B,Nc,d
        geo = pair_geometry(props.float(), slot_box[..., :2].float(), phys.float()).flatten(-2)  # B,P,S,8*PAIR_GEO
        sf = torch.cat([slots.to(dt), slot_box[..., 3:5].float().clamp(min=0.05).log().to(dt),
                        F.one_hot(slot_cls.clamp(min=0), N_CLS).to(dt),
                        self.lat_proj(lat.permute(0, 2, 1, 3).flatten(2).to(dt))], -1)
        pt = self.pair_geo(geo.to(dt)) + self.pair_slot(sf)[:, None] + self.fut_type[1]          # B,P,S,d
        mem = torch.cat([ctx[:, None].expand(-1, P, -1, -1), pt], 2).flatten(0, 1)               # B*P,Nc+S,d
        mk = torch.cat([ck[:, None].expand(-1, P, -1), slot_keep[:, None].expand(-1, P, -1)], 2).flatten(0, 1)
        q = z.reshape(B * P, 1, -1)
        for blk in self.ca:
            q = blk(q, None, mem, mk)
        q = q.view(B, P, -1)
        return {k: h(q).squeeze(-1) for k, h in self.heads.items()}


# ---------------------------------------------------------------- full model

class HGO(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.enc = Encoder(c)
        self.patch_proj = nn.Linear(self.enc.width, c.d)
        self.ray_in = mlp(6, c.d, c.d)
        self.cam_emb = nn.Parameter(torch.randn(N_CAM, c.d) * 0.02)
        self.pgrid = c.perc_grid
        # perception head on patch tokens (all variants): 2 m query cells, upsampled to the perception grid
        self.bev = BEVDecoder(c, int(round(2.0 / self.pgrid.cell)), c.bev_layers, self.pgrid)
        self.probe = BEVDecoder(c, 4, 2)                       # state probe: reads only the state tokens (pilot grid)
        # role probes (v2): self.probe then reads only the registers (map classes), slot_probe only the slots (objects)
        self.role_probes = bool(c.role_probes and c.n_reg and c.n_slot)
        self.slot_probe = BEVDecoder(c, 4, 2) if self.role_probes else None
        self.slots = SlotEncoder(c, self.enc.width) if c.n_slot else None
        self.prev_emb = nn.Parameter(torch.randn(c.d) * 0.02) if c.prev_frame else None  # "this patch is from t-1"
        self.hist_in = mlp(HIST_FEATS, c.d, c.d) if c.slot_hist else None
        if self.hist_in is not None:  # starts as "no history"
            nn.init.zeros_(self.hist_in[-1].weight)
            nn.init.zeros_(self.hist_in[-1].bias)
        self.reg_pos = nn.Parameter(torch.randn(max(c.n_reg, 1), c.d) * 0.02)
        self.type_emb = nn.Parameter(torch.randn(2, c.d) * 0.02)
        self.empty = nn.Parameter(torch.zeros(c.d))
        self.null = nn.Parameter(torch.zeros(1, 1, c.d))  # always-valid key for the state probe
        self.ego_in = mlp(4 * 11, c.d, c.d)
        self.mono = self.pmono = None
        if c.wm_kind == "v2":
            self.wm = StateWM(c)
            # view heads (training only): forecast latent + how the LOGGED ego saw the target -> EMA feature
            self.view = mlp(c.d + 9, c.ffn, c.d) if c.n_slot else None               # + target box geometry
            self.reg_view = mlp(c.d + EGO_POSE, c.ffn, c.d) if c.n_reg else None     # + logged ego pose at the horizon
            self.scorer = MonoScorer(c) if c.readout == "mono" else StructuredScorer(c)
            if c.readout == "both":
                self.mono = MonoScorer(c)
                self.pmono = PairMonoScorer(c) if c.pair_mono else None
        else:
            self.wm = WorldModel(c)
            self.scorer = Scorer(c)
        self.planner = Planner(c)
        # EMA target encoder (registers and slot appearance) for JEPA targets on future frames
        self.target_enc = copy.deepcopy(self.enc)
        self.target_feat = copy.deepcopy(self.slots.feat) if self.slots is not None else None
        for p in list(self.target_enc.parameters()) + (list(self.target_feat.parameters()) if self.target_feat else []):
            p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.target_enc.eval()  # EMA targets: no GridMask, no dropout
        return self

    # -- state
    def patch_tokens(self, grid, cam2ego, K, patch):
        """Patch grid (B*4, width, h, w) -> tokens (B, 4*h*w, d) + embedding of the patch rays (camera pose in the t0
        ego frame, so earlier frames are ego-motion compensated) + camera embedding."""
        B = cam2ego.shape[0]
        hw = tuple(grid.shape[-2:])
        d, o = patch_rays(cam2ego.float(), K.float(), hw, patch)
        pt = self.patch_proj(grid.flatten(2).transpose(1, 2)).reshape(B, -1, self.c.d)
        cam = self.cam_emb.repeat_interleave(hw[0] * hw[1], 0)[None]
        return pt + self.ray_in(torch.cat([d, o / 10.0], -1)).to(pt.dtype) + cam

    def perceive(self, x):
        """-> registers, patch grid of t, patch tokens of t, patch tokens of t-1 (None without prev_frame)."""
        c = self.c
        reg, grid = self.enc(x["img"])
        pt = self.patch_tokens(grid, x["cam2ego"], x["K"], 14)
        pp = None
        if c.prev_frame:
            with torch.set_grad_enabled(torch.is_grad_enabled() and c.prev_grad):
                _, gp = self.enc(x["img_prev"])
            if c.prev_pool > 1:
                gp = F.avg_pool2d(gp, c.prev_pool)
            pp = self.patch_tokens(gp, x["cam2ego_prev"], x["K_prev"], 14 * c.prev_pool) + self.prev_emb.to(pt.dtype)
        return reg, grid, pt, pp

    def build_state(self, x, reg, grid, det, det_prev=None):
        B = reg.shape[0]
        c = self.c
        toks, keep, out = [], [], {}
        if c.n_reg:
            toks.append(reg + self.reg_pos[:c.n_reg] + self.type_emb[0])
            keep.append(torch.ones(B, c.n_reg, dtype=torch.bool, device=reg.device))
        if c.n_slot:
            # detections enter the state as observations: no gradient from the state back into the detection head
            # (it is trained by the detection loss only; atan2 / top-k paths are also numerically unsafe to back-prop)
            box, vel, cls, score = (t.detach() for t in decode_boxes(det, c.det_topk, c.det_thr, self.pgrid))
            box, vel, cls, score, sk = select_slots(box, vel, cls, score, c.n_slot)
            s, vis = self.slots(grid, box, vel, cls, score, x["cam2ego"], x["K"])
            if self.hist_in is not None and det_prev is not None:
                pb, _, pc, ps = (t.detach() for t in decode_boxes(det_prev, c.det_topk, c.det_thr, self.pgrid))
                hf, hm = match_prev(box, vel, cls, sk, pb, pc, ps, c.hist_gate)
                s = s + self.hist_in(hf.to(s.dtype))
                out["slot_hist_ok"] = hm
            s = torch.where(sk[..., None], s + self.type_emb[1], self.empty.to(s.dtype).expand_as(s))
            toks.append(s)
            keep.append(sk)
            out.update(slot_box=box, slot_vel=vel, slot_cls=cls, slot_score=score, slot_keep=sk, slot_vis=vis)
        out["state"] = torch.cat(toks, 1)
        out["keep"] = torch.cat(keep, 1)
        return out

    def forward(self, x, train_traj=None, perception_only=False, score_idx=None, fut_perm=None, return_imagined=False):
        """x: INPUT_KEYS of a batch. train_traj (B,8,3): logged ego trajectory to condition the world model during
        training (its future is the JEPA label). score_idx (K,): training only; imagine and score just these proposals
        (the ones that get PDM labels this step) instead of all n_prop. Diagnostics (scripts/hgo/dump_plans.py):
        fut_perm (P,): proposal p is scored with the imagined future of proposal fut_perm[p] (future-swap test);
        return_imagined: also return the imagined futures and slot motion of every proposal."""
        c = self.c
        reg, grid, pt, pp = self.perceive(x)
        out = {"det": self.bev(pt if pp is None else torch.cat([pt, pp], 1))}
        if c.slot_hist:  # t-1 detections from the t-1 patches alone (t0 frame), for the slot history
            out["det_prev"] = self.bev(pp)
        out.update(self.build_state(x, reg, grid, out["det"], out.get("det_prev")))
        state, keep = out["state"], out["keep"]
        B = state.shape[0]
        null_keep = torch.ones(B, 1, dtype=torch.bool, device=keep.device)
        null = self.null.expand(B, -1, -1).to(state.dtype)
        if self.role_probes:
            nr = c.n_reg
            out["probe"] = self.probe(torch.cat([null, state[:, :nr]], 1), torch.cat([null_keep, keep[:, :nr]], 1))
            out["slot_probe"] = self.slot_probe(torch.cat([null, state[:, nr:]], 1), torch.cat([null_keep, keep[:, nr:]], 1))
        else:
            out["probe"] = self.probe(torch.cat([null, state], 1), torch.cat([null_keep, keep], 1))
        if perception_only:  # stage 1: encoder + detection / BEV + state probe
            return out
        ego = self.ego_in(x["ego"].flatten(1))
        out["ego_tok"] = ego
        mem = torch.cat([ego[:, None], state], 1)
        mk = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=keep.device), keep], 1)
        props = self.planner(mem, mk, ego)
        out["proposals"] = props
        P = props[-1].detach()
        if score_idx is not None:
            P = P.index_select(1, score_idx)
        if c.wm_kind == "v2":
            return self._forward_v2(x, out, state, keep, ego, P, train_traj)
        if train_traj is not None:
            out["wm_fut"], out["wm_motion"] = self.wm(state, keep, ego, train_traj)
        Pn = P.shape[1]
        fut, mot = self.wm(state.repeat_interleave(Pn, 0), keep.repeat_interleave(Pn, 0), ego.repeat_interleave(Pn, 0),
                           P.flatten(0, 1))
        if return_imagined:
            out["imagined"], out["imagined_motion"] = fut, mot
        if fut_perm is not None:
            fut = fut.reshape(B, Pn, *fut.shape[1:])[:, fut_perm].flatten(0, 1)
            mot = mot.reshape(B, Pn, *mot.shape[1:])[:, fut_perm].flatten(0, 1)
        slot_xy = out["slot_box"][..., :2] if c.n_slot else state.new_zeros(B, 0, 2)
        out["scores"] = self.scorer(P, state, keep, ego, fut, mot, slot_xy)
        out["pdm_score"] = selection_score(out["scores"], c.score_w)
        out["trajectory"] = P[torch.arange(B), out["pdm_score"].argmax(1)]
        return out

    def _forward_v2(self, x, out, state, keep, ego, P, train_traj=None):
        """v2: world model once per scene (no per-proposal imagination on log-replay), then the readout(s).
        out["scores"] = structured readout logits, out["scores_mono"] = monolithic logits (whichever exist);
        out["pdm_score_v2"] / out["pdm_score_mono"] and out["sub_probs_v2"] / out["sub_probs_mono"] for each rule;
        out["pdm_score"], out["sub_probs"], out["trajectory"] follow the configured rule. train_traj is unused: the
        forecasts need no action (their targets' viewpoint enters only through the view heads in the losses)."""
        c = self.c
        B, nr = state.shape[0], c.n_reg
        reg_fut, lat, phys = self.wm(state, keep, ego)
        out["reg_fut"] = reg_fut
        if c.n_slot:
            out["agent_lat"], out["agent_phys"] = lat, phys
        ego_raw = x["ego"].float()
        slot_kw = dict(slots=state[:, nr:], slot_keep=keep[:, nr:], slot_box=out["slot_box"], slot_cls=out["slot_cls"],
                       lat=lat, phys=phys) if c.n_slot else {}
        mono = self.mono if c.readout == "both" else (self.scorer if c.readout == "mono" else None)
        if c.readout in ("v2", "both"):
            lg = self.scorer(P, ego, ego_raw, state[:, :nr], keep[:, :nr], reg_fut=reg_fut, **slot_kw)
            out["scores"] = lg
            if c.n_slot:
                out["pdm_score_v2"], out["sub_probs_v2"] = structured_selection(lg, out["slot_cls"], out["slot_keep"],
                                                                                c.score_w)
        if mono is not None:
            lm = mono(P, ego, ego_raw, state, keep, out.get("slot_box"), lat, phys, reg_fut)
            out["scores_mono"] = lm
            out["pdm_score_mono"] = selection_score(lm, c.score_w)
            out["sub_probs_mono"] = torch.stack([lm[k].float().sigmoid() for k in SUB_SCORES], -1)
        if self.pmono is not None:
            lp = self.pmono(P, ego, ego_raw, state[:, :nr], keep[:, :nr], reg_fut=reg_fut, **slot_kw)
            out["scores_pmono"] = lp
            out["pdm_score_pmono"] = selection_score(lp, c.score_w)
            out["sub_probs_pmono"] = torch.stack([lp[k].float().sigmoid() for k in SUB_SCORES], -1)
        rule = c.select if c.readout == "both" else c.readout
        out["pdm_score"], out["sub_probs"] = out[f"pdm_score_{rule}"], out[f"sub_probs_{rule}"]
        out["trajectory"] = P[torch.arange(B), out["pdm_score"].argmax(1)]
        return out

    @torch.no_grad()
    def targets(self, fut_img, fut_cam2ego, fut_K, fut_box, fut_ok):
        """EMA targets at each horizon: registers (B,H,n_reg,d) and slot appearance at the GT future box (B,H,M,d)."""
        B, H = fut_img.shape[:2]
        reg, grid = self.target_enc(fut_img.flatten(0, 1))
        out = {"reg": F.layer_norm(reg.float(), (reg.shape[-1],)).view(B, H, self.c.n_reg, reg.shape[-1])}
        if self.target_feat is not None:
            M = fut_box.shape[2]
            rois, area = project_boxes_t(fut_box.flatten(0, 1), fut_cam2ego.flatten(0, 1).float(),
                                         fut_K.flatten(0, 1).float(), self.c.image_wh)
            best = area.argmax(-1)
            vis = (area.amax(-1) > 0) & fut_ok.flatten(0, 1)
            r = rois.gather(2, best[..., None, None].expand(-1, -1, 1, 4)).squeeze(2)
            bidx = (torch.arange(B * H, device=r.device)[:, None] * N_CAM + best).float()
            f = roi_align(grid.float(), torch.cat([bidx.reshape(-1, 1), r.reshape(-1, 4)], 1), (2, 2),
                          spatial_scale=1.0 / 14, sampling_ratio=2, aligned=True).mean((2, 3)).view(B * H, M, -1)
            t = self.target_feat(f)
            out["slot"] = F.layer_norm(t.float(), (t.shape[-1],)).view(B, H, M, -1)
            out["slot_ok"] = vis.view(B, H, M)
        return out

    @torch.no_grad()
    def sync_targets(self):
        """Copy the online encoder (and slot feature head) into the EMA targets. Called at the start of stage 2 so that the
        teacher starts from the warmed-up encoder, not from the initialization (RESEARCH_DESIGN §8)."""
        self.target_enc.load_state_dict(self.enc.state_dict())
        if self.target_feat is not None:
            self.target_feat.load_state_dict(self.slots.feat.state_dict())

    @torch.no_grad()
    def update_ema(self):
        m = self.c.ema
        pairs = [(self.target_enc, self.enc)] + ([(self.target_feat, self.slots.feat)] if self.target_feat else [])
        for tgt, src in pairs:
            for pt, ps in zip(tgt.parameters(), src.parameters()):
                pt.mul_(m).add_(ps.detach(), alpha=1 - m)


def count_params(model: HGO):
    """Trainable parameters (the EMA copy and the frozen ViT weights are not counted)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
