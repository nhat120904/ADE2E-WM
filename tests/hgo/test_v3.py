"""HGO v3 unit tests (CPU, small images; docs/METHOD.md §4): the v3 preset end to end with the t-1 frame, inputs only,
the 0.5 m grid separating two nearby pedestrians (merged on the 1 m grid), BEV labels on the new grid, the slot history
matching, the L2 holdout mask of the CG protocol and the per-proposal label weights; the t-1 inputs of a real sample."""

import math

import numpy as np
import pytest
import torch

from ade2e_wm.hgo import losses as L
from ade2e_wm.hgo.data import M_MAX, HGODataset, input_keys
from ade2e_wm.hgo.holdout import candidate_maneuver, cell_candidates, parse_cell
from ade2e_wm.hgo.model import GRIDS, HGO, PROBE_GRID, count_params, decode_boxes, make_config, match_prev

from test_hgo import ROOT, VIT, WH, fake_batch


def fake_batch_v3(B=2):
    b = fake_batch(B)
    g = torch.Generator().manual_seed(1)
    b["img_prev"] = torch.randint(0, 255, b["img"].shape, generator=g, dtype=torch.uint8)
    b["cam2ego_prev"] = b["cam2ego"].clone()
    b["cam2ego_prev"][..., 0, 3] -= 1.0  # the ego moved 1 m forward between t-1 and t
    b["K_prev"] = b["K"].clone()
    for k in ("gt_box", "gt_vel", "gt_cls", "gt_mask"):
        b[k + "_prev"] = b[k].clone()
    return b


@pytest.fixture(scope="module")
def v3_model():
    torch.manual_seed(0)
    return HGO(make_config("V3", image_wh=WH, vit_weights=VIT, det_thr=0.0)).eval()


def test_v3_forward(v3_model):
    m, b = v3_model, fake_batch_v3()
    c = m.c
    assert (c.n_reg, c.n_slot, c.grid, c.prev_frame, c.slot_hist) == (64, 32, "v3", True, True)
    x = {k: b[k] for k in input_keys(True)}
    with torch.no_grad():
        out = m(x)
    G = GRIDS["v3"]
    assert (G.rows, G.cols) == (160, 128)
    assert out["det"]["hm"].shape == (2, 4, 160, 128) and out["det_prev"]["hm"].shape == (2, 4, 160, 128)
    assert out["state"].shape == (2, 96, 256) and out["keep"].shape == (2, 96)
    assert out["slot_hist_ok"].shape == (2, 32)
    assert out["probe"]["seg"].shape[-2:] == (64, 64)  # probes stay on the pilot grid
    assert out["trajectory"].shape == (2, 8, 3) and torch.isfinite(out["pdm_score"]).all()
    for r in ("v2", "mono", "pmono"):
        assert out[f"pdm_score_{r}"].shape == (2, 64)
    with torch.no_grad():
        o1 = m(x, perception_only=True)
    assert "slot_box" in o1 and "proposals" not in o1
    # the t-1 images reach the detection head (fusion is wired)
    x2 = dict(x, img_prev=torch.flip(x["img_prev"], dims=[-1]))
    with torch.no_grad():
        o2 = m(x2, perception_only=True)
    assert not torch.allclose(o1["det"]["hm"], o2["det"]["hm"])
    v2 = HGO(make_config("V2", image_wh=WH, vit_weights=VIT, det_thr=0.0))
    print("trainable params V3", count_params(m), "V2", count_params(v2))


def test_v3_reads_only_inputs(v3_model):
    b = fake_batch_v3()
    keys = input_keys(True)

    class Guard(dict):
        def __getitem__(self, k):
            assert k in keys, f"model read target key {k}"
            return dict.__getitem__(self, k)

    with torch.no_grad():
        v3_model(Guard(dict(b)))


def test_v3_ablation_flags():
    c = make_config("V3", prev_frame=False, slot_hist=False, grid="pilot", n_reg=32)
    assert (c.grid, c.prev_frame, c.slot_hist, c.n_reg) == ("pilot", False, False, 32)
    with pytest.raises(AssertionError):
        make_config("V3", prev_frame=False)  # slot history needs the t-1 frame
    old = make_config("V2")  # V2 / pilot checkpoints keep their defaults
    assert (old.grid, old.prev_frame, old.slot_hist) == ("pilot", False, False)


def _two_peds():
    gt_box = torch.zeros(1, M_MAX, 7)
    gt_box[0, :2] = torch.tensor([[10.0, 0.0, 0.9, 0.6, 0.6, 1.7, 0.0], [10.0, 1.0, 0.9, 0.6, 0.6, 1.7, 0.0]])
    gt_mask = torch.zeros(1, M_MAX, dtype=torch.bool)
    gt_mask[0, :2] = True
    gt_cls = torch.full((1, M_MAX), -1)
    gt_cls[0, :2] = 1
    return gt_box, torch.zeros(1, M_MAX, 2), gt_cls, gt_mask


@pytest.mark.parametrize("name,expect", [("v3", 2), ("pilot", 1)])
def test_grid_separates_nearby_pedestrians(name, expect):
    """Two pedestrians 1 m apart: two peaks on the 0.5 m grid, one on the 1 m grid (the merge seen in METHOD §7)."""
    G = GRIDS[name]
    gt_box, gt_vel, gt_cls, gt_mask = _two_peds()
    hm, pix, reg, ok = L.det_targets(gt_box, gt_vel, gt_cls, gt_mask, G)
    assert hm.shape == (1, 4, G.rows, G.cols) and ok.sum() == 2
    logits = torch.where(hm == 1, torch.full_like(hm, 8.0), torch.full_like(hm, -8.0))
    regmap = torch.zeros(1, 10, G.rows * G.cols)
    regmap.scatter_(2, pix[:, None].expand(-1, 10, -1), reg.transpose(1, 2))
    box, _, cls, score = decode_boxes({"hm": logits, "reg": regmap.view(1, 10, G.rows, G.cols)}, 10, 0.5, G)
    found = box[0, score[0] > 0.5, :2]
    assert len(found) == expect
    if expect == 2:  # both recovered exactly
        for i in range(2):
            assert (found - gt_box[0, i, :2]).norm(dim=-1).min() < 1e-4


def test_v3_box_beyond_32m_round_trip():
    G = GRIDS["v3"]
    gt_box = torch.zeros(1, M_MAX, 7)
    gt_box[0, 0] = torch.tensor([45.3, -7.2, 0.8, 4.5, 1.9, 1.6, 0.4])
    gt_mask = torch.zeros(1, M_MAX, dtype=torch.bool)
    gt_mask[0, 0] = True
    gt_cls = torch.full((1, M_MAX), -1)
    gt_cls[0, 0] = 0
    hm, pix, reg, ok = L.det_targets(gt_box, torch.zeros(1, M_MAX, 2), gt_cls, gt_mask, G)
    assert ok[0, 0]
    _, _, _, ok_pilot = L.det_targets(gt_box, torch.zeros(1, M_MAX, 2), gt_cls, gt_mask, PROBE_GRID)
    assert not ok_pilot[0, 0]
    logits = torch.where(hm == 1, torch.full_like(hm, 8.0), torch.full_like(hm, -8.0))
    regmap = torch.zeros(1, 10, G.rows * G.cols)
    regmap.scatter_(2, pix[:, None].expand(-1, 10, -1), reg.transpose(1, 2))
    box, _, cls, score = decode_boxes({"hm": logits, "reg": regmap.view(1, 10, G.rows, G.cols)}, 5, 0.5, G)
    j = int(score[0].argmax())
    assert torch.allclose(box[0, j], gt_box[0, 0], atol=1e-4) and int(cls[0, j]) == 0


def test_bev_on_grid():
    bev = torch.randint(0, 7, (2, 64, 64))
    assert torch.equal(L.bev_on_grid(bev, PROBE_GRID), bev)
    t = L.bev_on_grid(bev, GRIDS["v3"])
    assert t.shape == (2, 160, 128)
    i, j = torch.meshgrid(torch.arange(128), torch.arange(128), indexing="ij")
    assert torch.equal(t[:, :128], bev[:, i // 2, j // 2])  # x in [-32, 32): 2 x 2 cells per cached cell
    assert (t[:, 128:] == -100).all()                        # x in [32, 48]: not in the cache, ignored
    logits = torch.randn(2, 7, 160, 128)
    assert torch.isfinite(L.seg_loss(logits, t, torch.tensor([True, True])))


def test_match_prev():
    box = torch.tensor([[[10.0, 0, 0, 4, 2, 1.5, 0.1], [20.0, 5, 0, 4, 2, 1.5, 0.0], [0, 0, 0, 1, 1, 1, 0]]])
    vel = torch.tensor([[[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    cls = torch.tensor([[0, 0, 1]])
    keep = torch.tensor([[True, True, False]])
    pbox = torch.tensor([[[9.1, 0, 0, 4, 2, 1.5, 0.0], [20.0, 5, 0, 4, 2, 1.5, 0.0], [10.0, 0, 0, 1, 1, 1, 0]]])
    pcls = torch.tensor([[0, 1, 0]])
    pscore = torch.tensor([[0.6, 0.9, 0.0]])
    f, m = match_prev(box, vel, cls, keep, pbox, pcls, pscore, gate=2.0)
    assert m.tolist() == [[True, False, False]]  # slot 2: only a t-1 detection of another class; slot 3: not kept
    assert torch.allclose(f[0, 0], torch.tensor([1.0, 0.9 / 5, 0.0, math.sin(0.1), math.cos(0.1), 0.6]), atol=1e-5)
    assert (f[0, 1:] == 0).all()


def _arc_left(R=10.0):
    yaw = (math.pi / 2) * torch.arange(1, 9).float() / 8
    return torch.stack([R * torch.sin(yaw), R * (1 - torch.cos(yaw)), yaw], -1)


def _agents(entries):
    """entries: (cls, x, y, yaw) stationary agents -> GT tensors with B = 1."""
    gt_box = torch.zeros(1, M_MAX, 7)
    gt_cls = torch.full((1, M_MAX), -1)
    gt_mask = torch.zeros(1, M_MAX, dtype=torch.bool)
    gt_fut = torch.zeros(1, M_MAX, 8, 3)
    for i, (c, x, y, yaw) in enumerate(entries):
        gt_box[0, i] = torch.tensor([x, y, 0.8, 1.0, 1.0, 1.7, yaw])
        gt_cls[0, i], gt_mask[0, i] = c, True
        gt_fut[0, i] = torch.tensor([x, y, yaw])
    return gt_box, gt_mask, gt_cls, gt_fut, gt_mask[..., None].expand(-1, -1, 8).clone()


def test_holdout_mask():
    left = _arc_left()
    straight = torch.stack([torch.arange(1, 9).float() * 2, torch.zeros(8), torch.zeros(8)], -1)
    stop = torch.zeros(8, 3)
    props = torch.stack([left, straight, stop])[None]                                     # 1,3,8,3
    assert candidate_maneuver(props).tolist() == [[2, 1, 0]]                               # left, straight, stop
    assert parse_cell("turn|vru") == (("left", "right"), "vru")
    # a pedestrian standing on the left-turn path at its 2 s point (the ego centre, 1.461 m ahead of the rear axle)
    yaw4 = math.pi / 4
    px, py = 10 * math.sin(yaw4) + 1.461 * math.cos(yaw4), 10 * (1 - math.cos(yaw4)) + 1.461 * math.sin(yaw4)
    ped = _agents([(1, px, py, 0.0)])
    assert cell_candidates(props, *ped, "turn|vru").tolist() == [[True, False, False]]
    assert cell_candidates(props, *ped, "left|veh_opp").tolist() == [[False, False, False]]  # no vehicle
    nobody = _agents([])
    assert not cell_candidates(props, *nobody, "turn|vru").any()
    # an oncoming vehicle (heading pi) on the left-turn path at its 1 s point: opposing, not crossing
    yaw2 = math.pi / 8
    vx, vy = 10 * math.sin(yaw2) + 1.461 * math.cos(yaw2), 10 * (1 - math.cos(yaw2)) + 1.461 * math.sin(yaw2)
    veh = _agents([(0, vx, vy, math.pi)])
    assert cell_candidates(props, *veh, "left|veh_opp").tolist() == [[True, False, False]]
    assert cell_candidates(props, *veh, "left|veh_cross").tolist() == [[False, False, False]]
    assert not cell_candidates(props, *veh, "turn|vru").any()
    # a pedestrian far from every path: nothing masked
    far = _agents([(1, -20.0, 15.0, 0.0)])
    assert not cell_candidates(props, *far, "turn|vru").any()


def test_weighted_score_losses():
    from test_v2 import _fake_labels
    B, P, S = 2, 3, 4
    lab, has = L.collate_labels([_fake_labels(P=P), _fake_labels(P=P)], P, M_MAX)
    match = torch.tensor([[0, 1, -1, -1], [1, -1, 0, -1]])
    g = torch.Generator().manual_seed(0)
    lg = {k: torch.randn(B, P, generator=g, requires_grad=True) for k in ("dac", "ddc", "ep", "comfort")}
    lg["area"] = torch.randn(B, P, 8, 4, generator=g, requires_grad=True)
    lg["pair_fault"] = torch.randn(B, P, S, generator=g, requires_grad=True)
    lg["pair_ttc"] = torch.randn(B, P, S, generator=g, requires_grad=True)
    a = L.score_loss_v2(lg, lab, has, match)
    b = L.score_loss_v2(lg, lab, has, match, torch.ones(B, P))
    assert all(torch.allclose(a[k], b[k]) for k in a)
    pw = torch.ones(B, P)
    pw[0, 1] = 0.0
    out = L.score_loss_v2(lg, lab, has, match, pw)
    sum(out.values()).backward()
    for k, v in lg.items():
        assert (v.grad[0, 1] == 0).all(), k        # a masked candidate gets no gradient from any head
        assert v.grad.abs().sum() > 0, k
    mono = {k: torch.randn(B, P, requires_grad=True) for k in ["no_at_fault_collisions", "drivable_area_compliance",
                                                                 "ego_progress", "time_to_collision_within_bound",
                                                                 "comfort", "driving_direction_compliance"]}
    sum(L.score_loss_mono(mono, lab, has, pw=pw).values()).backward()
    assert all((v.grad[0, 1] == 0).all() and v.grad.abs().sum() > 0 for v in mono.values())


def test_det_recall():
    gt_box, _, gt_cls, gt_mask = _two_peds()
    slot_box = gt_box[:, :3].clone()
    slot_box[0, 1, 0] += 3.0          # second slot 3 m off: unmatched
    keep = torch.tensor([[True, True, False]])
    tp, n = L.det_recall(slot_box, keep, gt_cls[:, :3], gt_box, gt_mask, gt_cls)
    assert tp.tolist() == [0, 1, 0, 0] and n.tolist() == [0, 2, 0, 0]


@pytest.mark.skipif(not (ROOT / "data/navsim_logs/test").exists(), reason="no data")
def test_real_sample_prev_frame():
    log = (ROOT / "data/_lists/phase1_test_logs.txt").read_text().split()[0]
    G = GRIDS["v3"]
    ds = HGODataset([log], ROOT / "data/navsim_logs/test", ROOT / "data/sensor_blobs/test", wh=WH, train=True,
                    area=G.area, prev=True)
    ds.load_future_images = False
    s = ds[len(ds) // 2]
    for k in input_keys(True) + ["gt_box_prev", "gt_mask_prev"]:
        assert k in s, k
    assert s["img_prev"].shape == s["img"].shape and not np.array_equal(s["img_prev"], s["img"])
    # t-1 camera poses in the t0 frame = pose of the ego at t-1 (ego status row -2) applied to the camera extrinsics
    x, y, yaw = s["ego"][-2, :3]
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    exp = s["cam2ego"][:, :3, 3] @ Rz.T + np.array([x, y, 0.0])
    assert np.abs(s["cam2ego_prev"][:, :3, 3] - exp).max() < 0.05
    n = int(s["gt_mask"].sum())
    assert (s["gt_box"][:n, 0] <= 48 + 1e-4).all() and (s["gt_box"][:n, 0] >= -32 - 1e-4).all()
    assert (np.abs(s["gt_box"][:n, 1]) <= 32 + 1e-4).all()
    # eval mode (no gap / camera checks) also yields the t-1 inputs
    ev = HGODataset([log], ROOT / "data/navsim_logs/test", ROOT / "data/sensor_blobs/test", wh=WH, train=False,
                    area=G.area, prev=True)
    assert "img_prev" in ev[0]
