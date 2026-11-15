"""Level-2 holdout of the CG protocol (docs/CG_SPLIT.md §2, variant B-L12).

After the token purge (L1) no training window contains the held-out cell, but the scorer is also supervised on
counterfactual candidates: a candidate that turns while a pedestrian crosses its path would teach the held-out
combination through its PDM labels. `cell_candidates` flags such candidates; their labels are masked in every scorer loss
(all readouts; baselines apply the same rule).

Inclusive definition, as the purge (scripts/hgo/cg_cells.py, `in|<maneuver>|<partner>` columns), so the mask is a
superset of the strict cell:
  maneuver  of the candidate over 4 s, thresholds of scripts/hgo/cg_events.py: stop if the final rear-axle displacement is
            < 2 m, left / right if the final heading changes by > 20 deg, else straight (no intersection test)
  partner   an agent of the group (logged GT future, 0..4 s, t0 ego frame) within the group radius of the candidate's
            centre path (VRU 2.5 m, vehicle 3.0 m between centres; crossing or not) with PET < 3 s; vehicle groups by the
            heading of the vehicle relative to the candidate path at that point (veh_cross 45-135 deg, veh_opp > 135,
            veh_same < 45)
"""

import math

import torch

from .geometry import EGO_REAR_TO_CENTER

DT = 0.5
MAN_SETS = {"turn": ("left", "right")}  # the merged main fold (scripts/hgo/cg_split.py)
MAN_CODE = {"stop": 0, "straight": 1, "left": 2, "right": 3}
RADIUS = {"vru": 2.5, "veh": 3.0}
VRU_CLS, VEH_CLS = (1, 2), (0,)
PET = 3.0


def parse_cell(cell):
    """'turn|vru' -> (('left', 'right'), 'vru'); 'left+right|veh_opp' -> (('left', 'right'), 'veh_opp')."""
    man, partner = cell.split("|")
    mans = MAN_SETS.get(man, tuple(man.split("+")))
    assert all(m in MAN_CODE for m in mans), cell
    assert partner in ("vru", "veh_cross", "veh_opp", "veh_same"), cell
    return mans, partner


def candidate_maneuver(props):
    """(...,8,3) rear-axle poses at 0.5 .. 4 s in the t0 ego frame -> (...) MAN_CODE."""
    d = torch.hypot(props[..., -1, 0], props[..., -1, 1])
    dyaw = torch.rad2deg(torch.atan2(torch.sin(props[..., -1, 2]), torch.cos(props[..., -1, 2])))
    code = torch.full_like(d, MAN_CODE["straight"], dtype=torch.long)
    code = torch.where(dyaw > 20, MAN_CODE["left"], code)
    code = torch.where(dyaw < -20, MAN_CODE["right"], code)
    return torch.where(d < 2.0, MAN_CODE["stop"], code)


def centre_path(props):
    """(B,P,8,3) -> centre path (B,P,9,2) at 0 .. 4 s (t0: rear axle at the origin, heading 0) and its yaw (B,P,9)."""
    B, P = props.shape[:2]
    pose = torch.cat([props.new_zeros(B, P, 1, 3), props], 2)
    c = pose[..., :2] + EGO_REAR_TO_CENTER * torch.stack([torch.cos(pose[..., 2]), torch.sin(pose[..., 2])], -1)
    return c, pose[..., 2]


@torch.no_grad()
def cell_candidates(props, gt_box, gt_mask, gt_cls, gt_fut, gt_fut_ok, cell):
    """props (B,P,8,3); GT boxes at t0 (B,M,7), mask (B,M), class (B,M); GT future (B,M,8,3: x, y, yaw), ok (B,M,8).
    -> (B,P) bool: the candidate performs a maneuver of the cell AND conflicts with an agent of the cell's partner group."""
    mans, partner = parse_cell(cell)
    props = props.float()
    man = candidate_maneuver(props)
    in_man = torch.zeros_like(man, dtype=torch.bool)
    for m in mans:
        in_man |= man == MAN_CODE[m]
    base = "vru" if partner == "vru" else "veh"
    cls_ok = torch.zeros_like(gt_mask)
    for c in (VRU_CLS if base == "vru" else VEH_CLS):
        cls_ok |= gt_cls == c
    agent = gt_mask & cls_ok                                                              # B,M
    if not bool((in_man.any(1) & agent.any(1)).any()):
        return torch.zeros_like(in_man)
    pos = torch.cat([gt_box[..., None, :2].float(), gt_fut[..., :2].float()], 2)          # B,M,9,2 (0 .. 4 s)
    yaw = torch.cat([gt_box[..., None, 6].float(), gt_fut[..., 2].float()], 2)            # B,M,9
    valid = torch.cat([gt_mask[..., None], gt_fut_ok], 2) & agent[..., None]              # B,M,9
    C, cyaw = centre_path(props)                                                          # B,P,9,2 / B,P,9
    A, AB = C[:, :, :-1], C[:, :, 1:] - C[:, :, :-1]                                       # B,P,8,2 segments
    D = pos[:, None, :, :, None, :] - A[:, :, None, None]                                  # B,P,M,9,8,2
    L2 = (AB ** 2).sum(-1)[:, :, None, None]                                               # B,P,1,1,8
    u = ((D * AB[:, :, None, None]).sum(-1) / L2.clamp(min=1e-9)).clamp(0, 1)
    dist = (D - u[..., None] * AB[:, :, None, None]).norm(dim=-1)                          # B,P,M,9,8
    te = (torch.arange(8, device=props.device) + u) * DT                                   # ego time at the projection
    ta = (torch.arange(9, device=props.device) * DT)[:, None]                              # agent time
    pet = torch.where((dist < RADIUS[base]) & valid[:, None, :, :, None], (ta - te).abs(), torch.full_like(dist, math.inf))
    if base == "vru":
        hit = pet.amin((-1, -2)) < PET                                                     # B,P,M
    else:  # heading of the vehicle relative to the candidate path where the PET is smallest
        flat = pet.flatten(-2)
        pmin, j = flat.min(-1)                                                             # B,P,M
        k, s = j // 8, j % 8
        seg = AB.gather(2, s.view(*s.shape[:2], -1, 1).expand(-1, -1, -1, 2).clamp(max=7))  # B,P,M,2
        pyaw = torch.where(seg.norm(dim=-1) > 0.1, torch.atan2(seg[..., 1], seg[..., 0]), cyaw.gather(2, s))
        ay = yaw[:, None].expand(-1, props.shape[1], -1, -1).gather(3, k[..., None]).squeeze(-1)
        rel = torch.rad2deg(torch.atan2(torch.sin(ay - pyaw), torch.cos(ay - pyaw))).abs()
        grp = {"veh_cross": (rel >= 45) & (rel <= 135), "veh_opp": rel > 135, "veh_same": rel < 45}[partner]
        hit = (pmin < PET) & grp
    return in_man & hit.any(-1)
