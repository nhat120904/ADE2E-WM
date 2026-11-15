"""HGO unit tests (CPU, small images): data frames, budgets, no target leakage, detection round trip, collision labels."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ade2e_wm.hgo import losses as L
from ade2e_wm.hgo.data import INPUT_KEYS, M_MAX, HGODataset
from ade2e_wm.hgo.geometry import ego_rects, rect_corners_2d, rects_overlap
from ade2e_wm.hgo.model import HGO, count_params, decode_boxes, make_config

ROOT = Path(__file__).resolve().parents[2]
WH = (224, 126)  # 16 x 9 patches
VIT = str(ROOT / "checkpoints/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors")


def fake_batch(B=2):
    g = torch.Generator().manual_seed(0)
    cam2ego = torch.eye(4).repeat(B, 4, 1, 1)
    # cameras at the origin looking along +x (F0), -x (B0), +y (L0), -y (R0): camera z -> ego direction
    R = {0: [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], 1: [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
         2: [[1, 0, 0], [0, 0, 1], [0, -1, 0]], 3: [[-1, 0, 0], [0, 0, -1], [0, -1, 0]]}
    for c, r in R.items():
        cam2ego[:, c, :3, :3] = torch.tensor(r, dtype=torch.float)
        cam2ego[:, c, 2, 3] = 1.5
    K = torch.tensor([[100.0, 0, WH[0] / 2], [0, 100.0, WH[1] / 2], [0, 0, 1]]).repeat(B, 4, 1, 1)
    gt_box = torch.zeros(B, M_MAX, 7)
    gt_box[:, :3] = torch.tensor([[10.0, 0.0, 0.8, 4.5, 1.9, 1.6, 0.0], [5.0, 4.0, 0.9, 0.6, 0.6, 1.7, 1.0],
                                  [-8.0, -3.0, 0.8, 4.0, 1.8, 1.5, 3.0]])
    gt_mask = torch.zeros(B, M_MAX, dtype=torch.bool)
    gt_mask[:, :3] = True
    gt_cls = torch.full((B, M_MAX), -1)
    gt_cls[:, :3] = torch.tensor([0, 1, 0])
    return {"img": torch.randint(0, 255, (B, 4, 3, WH[1], WH[0]), generator=g, dtype=torch.uint8),
            "cam2ego": cam2ego, "K": K, "ego": torch.randn(B, 4, 11, generator=g),
            "traj": torch.stack([torch.arange(1, 9).float() * 2, torch.zeros(8), torch.zeros(8)], -1).repeat(B, 1, 1),
            "gt_box": gt_box, "gt_vel": torch.zeros(B, M_MAX, 2), "gt_cls": gt_cls, "gt_mask": gt_mask}


@pytest.fixture(scope="module")
def models():
    out = {}
    for v in ["R64", "O64", "H32"]:
        torch.manual_seed(0)
        out[v] = HGO(make_config(v, image_wh=WH, vit_weights=VIT, det_thr=0.0)).eval()
    return out


def test_budget_and_shapes(models):
    b = fake_batch()
    x = {k: b[k] for k in INPUT_KEYS}
    params = {}
    for v, m in models.items():
        with torch.no_grad():
            out = m(x, train_traj=b["traj"])
        c = m.c
        assert c.n_reg + c.n_slot == 64 and c.d == 256
        assert out["state"].shape == (2, 64, 256)
        assert out["proposals"][-1].shape == (2, 64, 8, 3)
        assert out["wm_fut"].shape == (2, 3, 64, 256)
        assert out["trajectory"].shape == (2, 8, 3)
        assert torch.isfinite(out["pdm_score"]).all()
        assert ("slot_risk" in out["scores"]) == (c.n_slot > 0)
        params[v] = count_params(m)
    print("trainable params", params)
    lo, hi = min(params.values()), max(params.values())
    assert hi / lo < 1.10, params  # same width everywhere; differences = slot encoder / scene tokens / risk head


def test_model_reads_only_inputs(models):
    b = fake_batch()

    class Guard(dict):
        def __getitem__(self, k):
            assert k in INPUT_KEYS, f"model read target key {k}"
            return dict.__getitem__(self, k)

    with torch.no_grad():
        models["H32"](Guard({k: v for k, v in b.items()}))


def test_detection_round_trip():
    b = fake_batch()
    hm, pix, reg, ok = L.det_targets(b["gt_box"], b["gt_vel"], b["gt_cls"], b["gt_mask"])
    assert ok.sum() == 6
    logits = torch.where(hm == 1, torch.full_like(hm, 8.0), torch.full_like(hm, -8.0))
    R = hm.shape[-1]
    regmap = torch.zeros(2, 10, R * R)
    regmap.scatter_(2, pix[:, None].expand(-1, 10, -1), reg.transpose(1, 2))
    box, vel, cls, score = decode_boxes({"hm": logits, "reg": regmap.view(2, 10, R, R)}, 10, 0.5)
    for i in range(3):
        d = (box[0, :, :2] - b["gt_box"][0, i, :2]).norm(dim=-1)
        j = int(d.argmin())
        assert d[j] < 1e-4 and score[0, j] > 0.5 and int(cls[0, j]) == int(b["gt_cls"][0, i])
        assert abs(math.remainder(float(box[0, j, 6] - b["gt_box"][0, i, 6]), 2 * math.pi)) < 1e-4
        assert torch.allclose(box[0, j, 3:6], b["gt_box"][0, i, 3:6], atol=1e-4)


def test_overlap_and_risk_labels():
    a = rect_corners_2d(torch.tensor([0.0, 0.0]), torch.tensor(0.0), torch.tensor(4.0), torch.tensor(2.0))
    assert rects_overlap(a, rect_corners_2d(torch.tensor([3.0, 0.0]), torch.tensor(0.5), torch.tensor(4.0), torch.tensor(2.0)))
    assert not rects_overlap(a, rect_corners_2d(torch.tensor([0.0, 3.0]), torch.tensor(0.0), torch.tensor(4.0), torch.tensor(2.0)))
    # ego drives straight at 2 m per step; a stopped car 10 m ahead is hit, one 10 m to the left is not
    props = torch.stack([torch.arange(1, 9).float() * 2, torch.zeros(8), torch.zeros(8)], -1)[None, None]
    gt_box = torch.zeros(1, 2, 7)
    gt_box[0, :, 3:5] = torch.tensor([4.5, 1.9])
    gt_fut = torch.zeros(1, 2, 8, 3)
    gt_fut[0, 0, :, 0] = 10.0
    gt_fut[0, 1, :, 0], gt_fut[0, 1, :, 1] = 10.0, 10.0
    y = L.risk_labels(props, torch.tensor([[0, 1]]), gt_box, gt_fut, torch.ones(1, 2, 8, dtype=torch.bool))
    assert y.tolist() == [[[1.0, 0.0]]]
    assert ego_rects(props).shape == (1, 1, 8, 4, 2)


def test_score_loss_finite():
    B, P, S = 2, 64, 32
    lg = {k: torch.randn(B, P) for k in ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
                                         "time_to_collision_within_bound", "comfort", "driving_direction_compliance"]}
    lg["slot_risk"] = torch.randn(B, P, S)
    pdm = torch.rand(B, P, 7).round()
    out = L.score_loss(lg, pdm, torch.tensor([True, False]), torch.zeros(B, P, S))
    assert all(torch.isfinite(v) for v in out.values())


@pytest.mark.skipif(not (ROOT / "data/navsim_logs/test").exists(), reason="no data")
def test_real_sample_frames():
    log = (ROOT / "data/_lists/phase1_test_logs.txt").read_text().split()[0]
    ds = HGODataset([log], ROOT / "data/navsim_logs/test", ROOT / "data/sensor_blobs/test", wh=WH, train=True)
    assert len(ds) > 0
    s = ds[len(ds) // 2]
    # ego status: the last history pose is the t0 origin; the plan starts near it and moves with the t0 velocity
    assert np.allclose(s["ego"][-1, :3], 0, atol=1e-5)
    v = s["ego"][-1, 3:5]
    assert np.linalg.norm(s["traj"][0, :2] - 0.5 * v) < 1.0 + 0.2 * np.linalg.norm(v)
    assert s["img"].shape == (4, 3, WH[1], WH[0]) and s["fut_img"].shape == (3, 4, 3, WH[1], WH[0])
    n = int(s["gt_mask"].sum())
    assert n > 0 and (np.abs(s["gt_box"][:n, :2]) <= 32).all()
    # the GT future of a box at its first step stays close to its t0 position (2 Hz, < 20 m/s)
    ok = s["gt_fut_ok"][:n, 0]
    assert (np.linalg.norm(s["gt_fut"][:n][ok, 0, :2] - s["gt_box"][:n][ok, :2], axis=1) < 10).all()
    # boxes project into the cameras: most nearby GT boxes are visible at t0 and at the first horizon
    from ade2e_wm.hgo.geometry import project_boxes_t
    near = np.hypot(s["gt_box"][:n, 0], s["gt_box"][:n, 1]) < 20
    _, area = project_boxes_t(torch.from_numpy(s["gt_box"][:n][near])[None], torch.from_numpy(s["cam2ego"])[None],
                              torch.from_numpy(s["K"])[None], WH)
    assert (area.amax(-1) > 0).float().mean() > 0.5
    fb = s["fut_box"][0][s["fut_box_ok"][0]]
    _, area = project_boxes_t(torch.from_numpy(fb)[None], torch.from_numpy(s["fut_cam2ego"][0])[None],
                              torch.from_numpy(s["fut_K"][0])[None], WH)
    assert (area.amax(-1) > 0).any()
    # perception warm-up mode skips the future images but keeps every other key
    ds.load_future_images = False
    s2 = ds[len(ds) // 2]
    assert "fut_img" not in s2 and set(s) - set(s2) == {"fut_img", "fut_cam2ego", "fut_K"}
    assert np.array_equal(s2["fut_box"], s["fut_box"]) and np.array_equal(s2["gt_fut"], s["gt_fut"])


def test_state_does_not_backprop_into_detections(models):
    """Detections are observations for the slots: the probe loss must not reach the detection regression head
    through the slot geometry (decode_boxes uses atan2 / top-k; their gradients were the suspected NaN source)."""
    m = models["H32"]
    b = fake_batch()
    out = m({k: b[k] for k in INPUT_KEYS}, perception_only=True)
    assert not out["slot_box"].requires_grad and not out["slot_score"].requires_grad
    m.zero_grad(set_to_none=True)
    out["probe"]["hm"].float().sum().backward()
    head = [p for n, p in m.named_parameters() if n.startswith("bev.") and p.grad is not None]
    assert all(torch.isfinite(p.grad).all() for p in head)


MC = ROOT / "exp/metric_cache/navtest_navsim2.2_np1.26.4"


@pytest.mark.skipif(not MC.exists(), reason="no navtest metric cache")
def test_pdm_adapter_on_v22_cache():
    """DrivoR's train scorer on an official v2.2 cache: the logged trajectory scores well, driving off sideways does not."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts/hgo"))
    from train import metric_cache_paths
    from ade2e_wm.hgo.pdm import proposal_scores
    paths = metric_cache_paths(MC)
    log = (ROOT / "data/_lists/phase1_test_logs.txt").read_text().split()[0]
    toks = set((ROOT / "data/_lists/navtest_tokens.txt").read_text().split())
    ds = HGODataset([log], ROOT / "data/navsim_logs/test", ROOT / "data/sensor_blobs/test", wh=WH, tokens=toks, train=False)
    finals = []
    for i in range(0, len(ds), max(1, len(ds) // 5))[:5]:
        s = ds[i]
        human = s["traj"]
        side = human.copy()
        side[:, 1] += np.linspace(3, 30, 8)  # 30 m to the left after 4 s
        sc = proposal_scores(paths[s["token"]], np.stack([human, side]))
        assert sc.shape == (2, 7) and np.isfinite(sc).all()
        finals.append(sc[0, 6])
        assert sc[0, 3] in (0.0, 1.0)  # TTC computed (2.0 = DrivoR scorer error fallback)
        assert sc[1, 1] <= sc[0, 1]  # drivable-area compliance cannot improve by swerving 30 m
    print("human final scores", finals)
    assert np.mean(finals) > 0.6
