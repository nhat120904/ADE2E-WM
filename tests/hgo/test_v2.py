"""HGO v2 unit tests (CPU, small images; docs/METHOD_RATIONALE.md §3): shapes, inputs only, the two readouts of one
model, routing and locality of the structured readout, selection, matching, label collation, coverage and losses; the v2
PDM labels on a real v2.2 cache checked against the official NAVSIM scorer when present."""

import math

import numpy as np
import pytest
import torch

from ade2e_wm.hgo import losses as L
from ade2e_wm.hgo.data import INPUT_KEYS, M_MAX, HGODataset
from ade2e_wm.hgo.model import (HGO, N_FLAGS, STATIC_CLS, SUB_SCORES, PairMonoScorer, StructuredScorer, count_params,
                                make_config, mlp, pair_geometry, structured_selection)

from test_hgo import MC, ROOT, VIT, WH, fake_batch


@pytest.fixture(scope="module")
def v2_models():
    out = {}
    for v in ["V2", "V2mono", "R64v2", "H32"]:
        torch.manual_seed(0)
        out[v] = HGO(make_config(v, image_wh=WH, vit_weights=VIT, det_thr=0.0)).eval()
    return out


def test_v2_presets_forward(v2_models):
    b = fake_batch()
    x = {k: b[k] for k in INPUT_KEYS}
    params, outs = {}, {}
    for v, m in v2_models.items():
        params[v] = count_params(m)
        if v == "H32":
            continue
        with torch.no_grad():
            out = outs[v] = m(x)
        c = m.c
        assert out["proposals"][-1].shape == (2, 64, 8, 3) and out["trajectory"].shape == (2, 8, 3)
        assert out["pdm_score"].shape == (2, 64) and torch.isfinite(out["pdm_score"]).all()
        assert out["sub_probs"].shape == (2, 64, 6)
        assert (out["sub_probs"] >= 0).all() and (out["sub_probs"] <= 1).all()
        assert out["reg_fut"].shape == (2, 3, c.n_reg, 256)
        assert "scores" in out and "scores_mono" in out and "pdm_score_mono" in out  # readout="both"
        assert torch.equal(out["pdm_score"], out[f"pdm_score_{c.select}"])
        rules = ["mono"]
        if c.n_slot:
            assert out["agent_lat"].shape == (2, 3, 32, 256) and out["agent_phys"].shape == (2, 32, 8, 4)
            assert "slot_probe" in out  # role probes
            assert out["scores"]["pair_fault"].shape == (2, 64, 32) and "pdm_score_v2" in out
            assert out["pdm_score_pmono"].shape == (2, 64) and out["sub_probs_pmono"].shape == (2, 64, 6)  # pair-mono
            rules += ["v2", "pmono"]
        else:
            assert "agent_lat" not in out and "slot_probe" not in out
            assert "pair_fault" not in out["scores"] and "pdm_score_v2" not in out  # context auxiliaries only
            assert "scores_pmono" not in out
        assert out["scores"]["area"].shape == (2, 64, 8, N_FLAGS)
        # no attention across proposals: scoring a subset gives the same scores for those proposals, for every rule
        with torch.no_grad():
            o2 = m(x, score_idx=torch.tensor([3, 17, 40]))
        assert torch.allclose(o2["pdm_score"], out["pdm_score"][:, [3, 17, 40]], atol=1e-4)
        for r in rules:
            assert torch.allclose(o2[f"pdm_score_{r}"], out[f"pdm_score_{r}"][:, [3, 17, 40]], atol=1e-4), r
    # V2, V2pmono and V2mono are the same model (same init here): only the selection rule differs
    assert params["V2"] == params["V2mono"]
    assert torch.equal(outs["V2"]["pdm_score_mono"], outs["V2mono"]["pdm_score"])
    assert torch.equal(outs["V2mono"]["pdm_score_v2"], outs["V2"]["pdm_score"])
    assert torch.equal(outs["V2"]["pdm_score_pmono"], outs["V2mono"]["pdm_score_pmono"])
    cv, cp = make_config("V2"), make_config("V2pmono")
    assert {k: v for k, v in vars(cv).items() if k != "select"} == {k: v for k, v in vars(cp).items() if k != "select"}
    print("trainable params", params)


def test_v2_reads_only_inputs(v2_models):
    b = fake_batch()

    class Guard(dict):
        def __getitem__(self, k):
            assert k in INPUT_KEYS, f"model read target key {k}"
            return dict.__getitem__(self, k)

    with torch.no_grad():
        for v in ["V2", "R64v2"]:
            v2_models[v](Guard({k: v for k, v in b.items()}))


def _scorer_inputs(c, B=1, P=5, S=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    H, d = len(c.horizons), c.d
    props = torch.cumsum(torch.rand(B, P, 8, 3, generator=g) * torch.tensor([2.0, 0.4, 0.05]), 2)
    return dict(props=props, ego_tok=torch.randn(B, d, generator=g), ego_raw=torch.randn(B, 4, 11, generator=g),
                reg=torch.randn(B, c.n_reg, d, generator=g), reg_keep=torch.ones(B, c.n_reg, dtype=torch.bool),
                slots=torch.randn(B, S, d, generator=g), slot_keep=torch.ones(B, S, dtype=torch.bool),
                slot_box=torch.cat([torch.randn(B, S, 3, generator=g) * 10, torch.rand(B, S, 3, generator=g) + 1,
                                    torch.randn(B, S, 1, generator=g)], -1),
                slot_cls=torch.tensor([[0, 1, 2, 3]]).repeat(B, 1)[:, :S],
                lat=torch.randn(B, H, S, d, generator=g), phys=torch.randn(B, S, 8, 4, generator=g),
                reg_fut=torch.randn(B, H, c.n_reg, d, generator=g))


def test_structured_readout_routing():
    """Pair terms of slot k depend only on slot k (and the proposal and flags); context terms never read slots but read
    the registers and the register forecast; without flags, pair terms read neither; a proposal's scores do not depend
    on the other proposals."""
    for flags in (True, False):
        c = make_config("V2", pair_flags=flags)
        torch.manual_seed(0)
        sc = StructuredScorer(c).eval()
        inp = _scorer_inputs(c)
        with torch.no_grad():
            o = sc(**inp)
            inp2 = dict(inp)
            for k in ("slots", "lat", "phys", "slot_box"):
                t = inp[k].clone()
                (t[:, 2] if k != "lat" else t[:, :, 2]).add_(5.0)
                inp2[k] = t
            o2 = sc(**inp2)
            for k in ("dac", "ddc", "ep", "comfort", "area"):
                assert torch.equal(o[k], o2[k]), k
            for k in ("pair_fault", "pair_ttc"):
                same = [0, 1, 3]
                assert torch.allclose(o[k][..., same], o2[k][..., same], atol=1e-5), k
                assert not torch.allclose(o[k][..., 2], o2[k][..., 2]), k
            # random change (a constant shift would vanish in the LayerNorm of the cross-attention memory)
            for key in ("reg", "reg_fut"):
                noise = torch.randn(inp[key].shape, generator=torch.Generator().manual_seed(1))
                o3 = sc(**dict(inp, **{key: inp[key] + noise}))
                assert not torch.allclose(o["dac"], o3["dac"]), key
                if not flags:
                    assert torch.allclose(o["pair_fault"], o3["pair_fault"], atol=1e-5), key
            props = inp["props"].clone()
            props[:, 0] += 3.0
            o4 = sc(**dict(inp, props=props))
            for k in ("dac", "ep", "pair_fault", "pair_ttc"):
                assert torch.allclose(o[k][:, 1:], o4[k][:, 1:], atol=1e-5), k
    # register-only: context heads only (reading the register forecast as well)
    c = make_config("R64v2")
    inp = _scorer_inputs(c)
    sc = StructuredScorer(c).eval()
    with torch.no_grad():
        o = sc(inp["props"], inp["ego_tok"], inp["ego_raw"], inp["reg"], inp["reg_keep"], reg_fut=inp["reg_fut"])
        o5 = sc(inp["props"], inp["ego_tok"], inp["ego_raw"], inp["reg"], inp["reg_keep"],
                reg_fut=inp["reg_fut"] + torch.randn(inp["reg_fut"].shape, generator=torch.Generator().manual_seed(2)))
    assert set(o) == {"dac", "ddc", "ep", "area", "comfort"}
    assert not torch.allclose(o["dac"], o5["dac"])


def test_pair_mono_readout():
    """Pair-mono readout: reads the per-pair features of every kept slot (a change of one slot can move every score: the
    combination is learned), ignores masked slots, reads the registers and the register forecast, and a proposal's
    scores do not depend on the other proposals."""
    c = make_config("V2")
    torch.manual_seed(0)
    sc = PairMonoScorer(c).eval()
    inp = _scorer_inputs(c)
    with torch.no_grad():
        o = sc(**inp)
        assert set(o) == set(SUB_SCORES) and all(v.shape == (1, 5) for v in o.values())
        slots = inp["slots"].clone()
        slots[:, 2] += 5.0
        assert not torch.allclose(o["no_at_fault_collisions"], sc(**dict(inp, slots=slots))["no_at_fault_collisions"])
        keep = inp["slot_keep"].clone()
        keep[:, 2] = False
        o_m = sc(**dict(inp, slot_keep=keep))
        o_m2 = sc(**dict(inp, slot_keep=keep, slots=slots))
        for k in SUB_SCORES:
            assert torch.allclose(o_m[k], o_m2[k], atol=1e-5), k  # a masked slot is not read
        for key in ("reg", "reg_fut"):
            noise = torch.randn(inp[key].shape, generator=torch.Generator().manual_seed(1))
            assert not torch.allclose(o["ego_progress"], sc(**dict(inp, **{key: inp[key] + noise}))["ego_progress"]), key
        props = inp["props"].clone()
        props[:, 0] += 3.0
        o4 = sc(**dict(inp, props=props))
        for k in SUB_SCORES:
            assert torch.allclose(o[k][:, 1:], o4[k][:, 1:], atol=1e-5), k


def test_pair_geometry():
    t = torch.arange(1, 9).float()
    props = torch.stack([2 * t, torch.zeros(8), torch.zeros(8)], -1)[None, None]       # straight along x, 4 m/s
    phys = torch.zeros(1, 1, 8, 4)
    phys[..., 3] = 1.0                                                                 # static, yaw 0
    g = pair_geometry(props, torch.tensor([[[10.0, 0.0]]]), phys)[0, 0, 0]             # (8, 6)
    assert torch.allclose(g[:, 0] * 10, 10 - 2 * t, atol=1e-5) and torch.allclose(g[:, 1], torch.zeros(8), atol=1e-6)
    assert torch.allclose(g[:, 3], torch.ones(8)) and torch.allclose(g[:, 5] * 10, torch.full((8,), 4.0))
    # ego heading +y (yaw 90 deg) standing at the origin; a slot 10 m along +y is ahead (dx > 0) in the ego frame
    props2 = torch.zeros(1, 1, 8, 3)
    props2[..., 2] = math.pi / 2
    g2 = pair_geometry(props2, torch.tensor([[[0.0, 10.0]]]), phys)[0, 0, 0]
    assert torch.allclose(g2[:, 0] * 10, torch.full((8,), 10.0), atol=1e-4) and torch.allclose(g2[:, 1], torch.zeros(8), atol=1e-4)


def test_structured_selection():
    w = make_config("V2").score_w
    lg = {k: torch.zeros(1, 2) for k in ("dac", "ddc", "ep", "comfort")}
    lg["pair_fault"] = torch.full((1, 2, 2), -30.0)
    lg["pair_ttc"] = torch.full((1, 2, 2), -30.0)
    lg["pair_fault"][0, 0, 0] = 30.0                         # proposal 0 hits slot 0
    keep = torch.ones(1, 2, dtype=torch.bool)
    s_agent, p_agent = structured_selection(lg, torch.tensor([[1, 0]]), keep, w)             # slot 0 = pedestrian
    s_static, p_static = structured_selection(lg, torch.tensor([[STATIC_CLS, 0]]), keep, w)  # slot 0 = static object
    assert s_agent[0, 1] > s_agent[0, 0] and s_static[0, 1] > s_static[0, 0] and s_static[0, 0] > s_agent[0, 0]
    assert p_agent[0, 0, 0] < 1e-3 and abs(float(p_static[0, 0, 0]) - 0.5) < 1e-3 and p_agent[0, 1, 0] > 0.999
    # empty slots never count, even with a high logit
    s_empty, _ = structured_selection(lg, torch.tensor([[1, 0]]), torch.tensor([[False, True]]), w)
    assert torch.allclose(s_empty[0, 0], s_empty[0, 1])
    # PDM semantics: NC is the MIN over collisions (0 agent, 0.5 non-agent), so two static hits still give 0.5, a static and
    # an agent hit give 0, and uncertain risks give the expectation prod_agents (1 - r) * (0.5 + 0.5 prod_static (1 - r))
    lg2 = dict(lg, pair_fault=torch.full((1, 1, 2), 30.0), pair_ttc=torch.full((1, 1, 2), -30.0),
               **{k: torch.zeros(1, 1) for k in ("dac", "ddc", "ep", "comfort")})
    nc = lambda cls, pf=None: float(structured_selection(dict(lg2, **({"pair_fault": pf} if pf is not None else {})),
                                                         torch.tensor([cls]), keep, w)[1][0, 0, 0])
    assert abs(nc([STATIC_CLS, STATIC_CLS]) - 0.5) < 1e-3
    assert nc([STATIC_CLS, 0]) < 1e-3 and nc([1, 2]) < 1e-3
    half = torch.zeros(1, 1, 2)                                 # r = 0.5 for both slots
    assert abs(nc([0, STATIC_CLS], half) - 0.5 * (0.5 + 0.5 * 0.5)) < 1e-4
    assert abs(nc([STATIC_CLS, STATIC_CLS], half) - (0.5 + 0.5 * 0.25)) < 1e-4


def test_match_gates_before_assignment():
    """Distances [[0, 1.5], [1.5, 2.12]] with a 2 m gate: the crossed assignment keeps two valid matches (gating after the
    assignment would keep only one); a slot never matches a box of another class."""
    sb = torch.zeros(1, 2, 7)
    sb[0, 1, 0] = 1.5
    gb = torch.zeros(1, 2, 7)
    gb[0, 1, 1] = 1.5
    keep, gm = torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool)
    assert L.match_slots(sb, keep, gb, gm).tolist() == [[1, 0]]
    sb1, gb1 = torch.zeros(1, 1, 7), torch.zeros(1, 1, 7)
    gb1[0, 0, 0] = 0.5
    k1 = torch.ones(1, 1, dtype=torch.bool)
    assert L.match_slots(sb1, k1, gb1, k1).item() == 0
    assert L.match_slots(sb1, k1, gb1, k1, slot_cls=torch.tensor([[1]]), gt_cls=torch.tensor([[0]])).item() == -1
    assert L.match_slots(sb1, k1, gb1, k1, slot_cls=torch.tensor([[0]]), gt_cls=torch.tensor([[0]])).item() == 0


def _fake_labels(P=3, n=2):
    return {"sub": np.ones((P, 7), np.float32), "ep_raw": np.full(P, 0.5, np.float32), "hcomf": np.ones(P, np.float32),
            "flags": np.zeros((P, 8, 4), np.uint8), "fault": np.array([[1, 0], [0, 0], [0, 1]], np.uint8),
            "ttc_pair": np.zeros((P, n), np.uint8), "ttc_ok": np.ones(P, bool),
            "fault_unmatched": np.array([0, 0, 0], np.int16)}


def test_collate_coverage_and_v2_losses():
    P, n = 3, 2
    lab, has = L.collate_labels([_fake_labels(), None], P, M_MAX)
    assert has.tolist() == [True, False] and lab["fault"].shape == (2, P, M_MAX)
    assert lab["fault"][0, :, :n].tolist() == [[1, 0], [0, 0], [0, 1]] and lab["fault"][0, :, n:].sum() == 0
    match = torch.tensor([[1, -1, 0], [0, 1, 2]])            # sample 0: slot 0 -> track 1, slot 2 -> track 0
    y, m = L.pair_labels(lab["fault"], match)
    assert y[0, :, 0].tolist() == [0, 0, 1] and y[0, :, 2].tolist() == [1, 0, 0] and m[0, 0].tolist() == [True, False, True]
    # coverage: both unsafe proposals of sample 0 collide with a matched track; then track 1 loses its slot
    assert L.fault_coverage(lab, has, match) == (pytest.approx(2 / 3), 0.0)
    assert L.fault_coverage(lab, has, torch.tensor([[-1, -1, 0], [0, 1, 2]]))[1] == pytest.approx(0.5)
    lab_u = dict(lab, fault_unmatched=torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))  # proposal 1: object not a track
    assert L.fault_coverage(lab_u, has, match) == (pytest.approx(1.0), pytest.approx(1 / 3))
    lg = {k: torch.randn(2, P) for k in ("dac", "ddc", "ep", "comfort")}
    lg["area"] = torch.randn(2, P, 8, 4)
    assert set(L.score_loss_v2(dict(lg), lab, has, match)) == {"dac", "ddc", "ep", "comfort", "area"}  # no pair heads
    lg["pair_fault"], lg["pair_ttc"] = torch.randn(2, P, 3), torch.randn(2, P, 3)
    out = L.score_loss_v2(lg, lab, has, match)
    assert all(torch.isfinite(v) for v in out.values())
    # a confident correct pair prediction gives a small pair loss, a confident wrong one a large loss
    good = dict(lg, pair_fault=torch.where(y > 0.5, 20.0, -20.0).float())
    bad = dict(lg, pair_fault=torch.where(y > 0.5, -20.0, 20.0).float())
    assert L.score_loss_v2(good, lab, has, match)["pair_fault"] < 1e-3 < L.score_loss_v2(bad, lab, has, match)["pair_fault"]
    mono = {k: torch.randn(2, P) for k in ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
                                          "time_to_collision_within_bound", "comfort", "driving_direction_compliance"]}
    mo = L.score_loss_mono(mono, lab, has)
    assert all(k.startswith("mono_") for k in mo) and all(torch.isfinite(v) for v in mo.values())
    # NC = 0.5 (collision with a non-agent object) stays a soft target 0.5 for the monolithic NC head
    lab05 = dict(lab, sub=lab["sub"].clone())
    lab05["sub"][0, :, 0] = 0.5
    z = dict(mono, no_at_fault_collisions=torch.zeros(2, P))  # predicts 0.5
    assert L.score_loss_mono(z, lab05, has)["mono_nc"] == pytest.approx(math.log(2), abs=1e-5)
    pm = L.score_loss_mono(mono, lab, has, prefix="pmono_")  # pair-mono readout: same targets, own keys
    assert set(pm) == {k.replace("mono_", "pmono_") for k in mo}
    assert all(torch.equal(pm["p" + k], v) for k, v in mo.items())


def test_view_losses():
    B, H, S, d = 2, 3, 4, 256
    view = mlp(d + 9, 64, d)
    match = torch.tensor([[0, 1, -1, -1], [2, -1, -1, -1]])
    tgt = {"slot": torch.randn(B, H, M_MAX, d), "slot_ok": torch.ones(B, H, M_MAX, dtype=torch.bool)}
    fut_box = torch.rand(B, H, M_MAX, 7) + 0.5
    assert torch.isfinite(L.view_loss(view, torch.randn(B, H, S, d), tgt, match, fut_box))
    gt_fut = torch.randn(B, M_MAX, 8, 3)
    assert torch.isfinite(L.phys_loss(torch.randn(B, S, 8, 4), torch.randn(B, S, 2), match, gt_fut,
                                      torch.ones(B, M_MAX, 8, dtype=torch.bool)))
    fut_ego = torch.eye(4).repeat(B, H, 1, 1)
    fut_ego[:, :, 0, 3] = 10.0
    f = L.ego_pose_feats(fut_ego)
    assert torch.allclose(f[0, 0], torch.tensor([1.0, 0.0, 0.0, 1.0]))
    reg_view = mlp(d + 4, 64, d)
    assert torch.isfinite(L.reg_view_loss(reg_view, torch.randn(B, H, 8, d), torch.randn(B, H, 8, d), fut_ego))


@pytest.mark.skipif(not MC.exists(), reason="no navtest metric cache")
def test_pdm_labels_on_v22_cache():
    """v2 labels against the OFFICIAL NAVSIM v2.2 scorer on the same simulated states: history comfort equals the official
    history comfort; the official NC and TTC equal the train scorer's; NC = 0 / 0.5 / 1 agrees with the classes of the
    per-track at-fault labels; a TTC violation agrees with the per-track TTC labels. (The track identities come from the
    train scorer's own bookkeeping; they are not re-derived independently here.)"""
    import sys
    sys.path.insert(0, str(ROOT / "scripts/hgo"))
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex as OM
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex as OW
    from train import metric_cache_paths
    from ade2e_wm.hgo import pdm
    import lzma
    import pickle
    paths = metric_cache_paths(MC)
    log = (ROOT / "data/_lists/phase1_test_logs.txt").read_text().split()[0]
    toks = set((ROOT / "data/_lists/navtest_tokens.txt").read_text().split())
    ds = HGODataset([log], ROOT / "data/navsim_logs/test", ROOT / "data/sensor_blobs/test", wh=WH, tokens=toks, train=False)
    official = PDMScorer(pdm.simulator.proposal_sampling, PDMScorerConfig())
    n_coll = n_off = n_ttc = 0
    for i in range(0, len(ds), max(1, len(ds) // 6))[:6]:
        s = ds[i]
        tracks = s["gt_track"].split(",") if s["gt_track"] else []
        n = len(tracks)
        assert n == int(s["gt_mask"].sum())
        cls = s["gt_cls"][:n]
        human = s["traj"]
        side = human.copy()
        side[:, 1] += np.linspace(3, 30, 8)
        fast = np.stack([np.linspace(8, 64, 8), np.zeros(8), np.zeros(8)], -1).astype(np.float32)  # 16 m/s straight
        r = pdm.proposal_labels(paths[s["token"]], np.stack([human, side, fast]), tracks)
        states = pdm.scorer._states.copy()
        mc = pickle.load(lzma.open(paths[s["token"]], "rb"))
        official._reset(states, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map,
                        mc.past_human_trajectory)
        official._calculate_ego_area()
        official._calculate_no_at_fault_collision()
        official._calculate_ttc()
        official._calculate_history_comfort()
        P = 3
        assert r["sub"].shape == (P, 7) and r["fault"].shape == (P, n) and r["flags"].shape == (P, 8, 4)
        assert set(np.unique(r["flags"])) <= {0, 1} and ((r["ep_raw"] >= 0) & (r["ep_raw"] <= 1)).all()
        assert np.array_equal(r["hcomf"], official._weighted_metrics[OW.HISTORY_COMFORT].astype(np.float32))
        assert np.array_equal(r["sub"][:, 0], official._multi_metrics[OM.NO_COLLISION].astype(np.float32))
        assert np.array_equal(r["flags"][..., :3].astype(bool), official._ego_areas[:, pdm.POSE_IDX, :3])
        for p in range(P):
            nc, fault = r["sub"][p, 0], r["fault"][p].astype(bool)
            if nc == 1:
                assert not fault.any() and r["fault_unmatched"][p] == 0, (i, p)
            elif nc == 0.5:  # only non-agent objects were hit
                assert (cls[fault] == STATIC_CLS).all(), (i, p)
            else:
                assert nc == 0 and ((cls[fault] != STATIC_CLS).any() or r["fault_unmatched"][p] > 0), (i, p)
            n_coll += int(nc < 1)
            if r["ttc_ok"][p]:
                assert r["sub"][p, 3] == official._weighted_metrics[OW.TTC][p], (i, p)
                assert (r["sub"][p, 3] == 0) == bool(r["ttc_pair"][p].any() or r["ttc_unmatched"][p] > 0), (i, p)
                n_ttc += int(r["sub"][p, 3] == 0)
            if r["flags"][p, :, 1].any():  # DAC = 0 iff a 10 Hz state leaves the drivable area; flags sample 8 of them
                assert r["sub"][p, 1] == 0, (i, p)
        n_off += int(r["flags"][:, :, 1].any())
    print("proposals with an at-fault collision:", n_coll, "| TTC violations:", n_ttc, "| samples with off-road:", n_off)
    assert n_off > 0 and n_coll > 0
