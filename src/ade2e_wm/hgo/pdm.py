"""Online PDM sub-scores of proposals for training (DrivoR's train scorer) on official NAVSIM v2.2 metric caches.

DrivoR's train scorer needs `pdm_progress`, the raw progress of the PDM-Closed reference trajectory, which DrivoR stores
in its own train cache. The official v2.2 cache holds the PDM-Closed trajectory itself (`trajectory`), so the value is
recomputed here exactly as DrivoR's cache processor does (simulate it, no-at-fault-collision x drivable-area compliance x
raw progress), with DrivoR's train scorer and the cache's centerline / drivable-area map. One extra proposal per call.
Columns of the result: nc, dac, ep, ttc, comfort, ddc, final (as DrivoR's get_sub_score).
"""

import lzma
import pickle
from enum import IntEnum

import numpy as np

import ade2e_wm.agents.drivor.score_module.compute_navsim_score as cns
import ade2e_wm.agents.drivor.score_module.train_pdm_scorer as tps


class MultiMetricIndex(IntEnum):
    """DrivoR's NAVSIM fork (third_party/DrivoR .../pdm_enums.py); NAVSIM v2.2 adds traffic-light and driving-direction
    rows that DrivoR's train scorer never fills (their product would be 0)."""
    NO_COLLISION = 0
    DRIVABLE_AREA = 1


class WeightedMetricIndex(IntEnum):
    PROGRESS = 0
    TTC = 1
    COMFORTABLE = 2
    DRIVING_DIRECTION = 3


# the vendored train scorer resolves these names at call time; the rest of its enums match NAVSIM v2.2
tps.MultiMetricIndex, tps.WeightedMetricIndex = MultiMetricIndex, WeightedMetricIndex
cns.MultiMetricIndex, cns.WeightedMetricIndex = MultiMetricIndex, WeightedMetricIndex


from ade2e_wm.agents.drivor.score_module.train_pdm_scorer import *  # noqa: F401,F403 (names used by _calculate_ttc)
MultiMetricIndex, WeightedMetricIndex = tps.MultiMetricIndex, tps.WeightedMetricIndex


class TrainScorer(tps.PDMScorer):
    """DrivoR's train scorer; only _calculate_ttc differs (copied, with the v2.2 evaluation range)."""

    def _calculate_ttc(self):
        """
        Re-implementation of nuPlan's time-to-collision metric.
        """

        ttc_scores = np.ones(self._num_proposals, dtype=np.float64)
        self.temp_collided_track_ids = {
            proposal_idx: copy.deepcopy(self._observation.collided_track_ids)
            for proposal_idx in range(self._num_proposals)
        }
        self.ttc_collided_track_ids = {
            proposal_idx: copy.deepcopy(self._observation.collided_track_ids)
            for proposal_idx in range(self._num_proposals)
        }

        # calculate TTC for 1s in the future with less temporal resolution.
        future_time_idcs = np.arange(0, 10, 3)
        n_future_steps = len(future_time_idcs)

        # create polygons for each ego position and 1s future projection
        coords_exterior = self._ego_coords.copy()
        coords_exterior[:, :, BBCoordsIndex.CENTER, :] = coords_exterior[:, :, BBCoordsIndex.FRONT_LEFT, :]
        coords_exterior_time_steps = np.repeat(coords_exterior[:, :, None], n_future_steps, axis=2)

        speeds = np.hypot(
            self._states[..., StateIndex.VELOCITY_X],
            self._states[..., StateIndex.VELOCITY_Y],
        )

        dxy_per_s = np.stack(
            [
                np.cos(self._states[..., StateIndex.HEADING]) * speeds,
                np.sin(self._states[..., StateIndex.HEADING]) * speeds,
            ],
            axis=-1,
        )

        for idx, future_time_idx in enumerate(future_time_idcs):
            delta_t = float(future_time_idx) * self.proposal_sampling.interval_length
            coords_exterior_time_steps[:, :, idx] = (
                    coords_exterior_time_steps[:, :, idx] + dxy_per_s[:, :, None] * delta_t
            )

        polygons = creation.polygons(coords_exterior_time_steps)

        # check collision for each proposal and projection
        # NAVSIM v2.2 caches hold 4 s of observations (41 steps), DrivoR's train cache 5 s: as the official v2.2 scorer,
        # evaluate TTC only where the look-ahead stays inside the observation horizon
        n_eval = min(self.proposal_sampling.num_poses, len(self._observation._global_to_local_idcs) - 1 - int(max(future_time_idcs)))
        for time_idx in range(n_eval + 1):
            for step_idx, future_time_idx in enumerate(future_time_idcs):
                current_time_idx = time_idx + future_time_idx
                polygons_at_time_step = polygons[:, time_idx, step_idx]
                intersecting = self._observation[current_time_idx].query(polygons_at_time_step, predicate="intersects")

                if len(intersecting) == 0:
                    continue

                for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                    token = self._observation[current_time_idx].tokens[geometry_idx]
                    if (
                            (self._observation.red_light_token in token)
                            or (token in self.temp_collided_track_ids[proposal_idx])
                            or (speeds[proposal_idx, time_idx] < self._config.stopped_speed_threshold)
                    ):
                        continue

                    ego_in_multiple_lanes_or_nondrivable_area = (
                            self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                            or self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA]
                    )
                    ego_rear_axle: StateSE2 = StateSE2(*self._states[proposal_idx, time_idx, StateIndex.STATE_SE2])

                    centroid = self._observation[current_time_idx][token].centroid
                    track_heading = self._observation.unique_objects[token].box.center.heading
                    track_state = StateSE2(centroid.x, centroid.y, track_heading)
                    # TODO: fix ego_area for intersection
                    if is_agent_ahead(ego_rear_axle, track_state) or (
                            (
                                    ego_in_multiple_lanes_or_nondrivable_area
                                    or self._drivable_area_map.is_in_layer(
                                ego_rear_axle.point, layer=SemanticMapLayer.INTERSECTION
                            )
                            )
                            and not is_agent_behind(ego_rear_axle, track_state)
                    ):
                        ttc_scores[proposal_idx] = np.minimum(ttc_scores[proposal_idx], 0.0)
                        self._ttc_time_idcs[proposal_idx] = min(time_idx, self._ttc_time_idcs[proposal_idx])

                        self.ttc_collided_track_ids[proposal_idx].append(token)
                    else:
                        self.temp_collided_track_ids[proposal_idx].append(token)

        self._weighted_metrics[WeightedMetricIndex.TTC] = ttc_scores


scorer = TrainScorer(cns.proposal_sampling, tps.PDMScorerConfig())
simulator, Trajectory = cns.simulator, cns.Trajectory
get_trajectory_as_array, transform_trajectory = cns.get_trajectory_as_array, cns.transform_trajectory


def reference_progress(mc):
    s = simulator.proposal_sampling
    states = get_trajectory_as_array(mc.trajectory, s, mc.ego_state.time_point)[None]
    sim = simulator.simulate_proposals(states, mc.ego_state)
    scorer._reset(sim, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map)
    scorer._calculate_ego_area()
    scorer._calculate_no_at_fault_collision()
    scorer._calculate_drivable_area_compliance()
    scorer._calculate_progress()
    return scorer._progress_raw * scorer._multi_metrics.prod(axis=0)


def _score(cache_path, poses):
    with lzma.open(cache_path, "rb") as f:
        mc = pickle.load(f)
    pdm_progress = mc.pdm_progress if hasattr(mc, "pdm_progress") else reference_progress(mc)
    s = simulator.proposal_sampling
    states = np.stack([get_trajectory_as_array(transform_trajectory(Trajectory(p), mc.ego_state), s,
                                               mc.ego_state.time_point) for p in poses])
    sim = simulator.simulate_proposals(states, mc.ego_state)
    final = scorer.score_proposals(sim, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map,
                                   pdm_progress)
    m, w = scorer._multi_metrics, scorer._weighted_metrics
    sub = np.stack([m[MultiMetricIndex.NO_COLLISION], m[MultiMetricIndex.DRIVABLE_AREA], w[WeightedMetricIndex.PROGRESS],
                    w[WeightedMetricIndex.TTC], w[WeightedMetricIndex.COMFORTABLE],
                    w[WeightedMetricIndex.DRIVING_DIRECTION], final], -1).astype(np.float32)
    return mc, pdm_progress, sub


def proposal_scores(cache_path, poses):
    """poses (P, 8, 3) in the t0 ego rear-axle frame -> (P, 7) float32."""
    return _score(cache_path, poses)[2]


# ---------------------------------------------------------------- v2 labels (docs/METHOD_RATIONALE.md §3.5)

from nuplan.common.actor_state.state_representation import TimePoint  # noqa: E402
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (  # noqa: E402
    ego_states_to_state_array,
)

POSE_IDX = np.arange(5, 41, 5)   # scorer states are at 10 Hz (41 incl. t0); the 8 planned poses are at 0.5 s
FLAG_NAMES = ["multiple_lanes", "non_drivable_area", "oncoming_traffic", "intersection"]


def history_comfort(states, past_trajectory, interval):
    """NAVSIM v2.2 history comfort (official pdm_scorer._calculate_history_comfort): the proposal states padded with the
    human past trajectory at 10 Hz must be comfortable. states (P, T, StateIndex.size()) -> (P,) float32."""
    if past_trajectory is None:
        return np.ones(len(states), np.float32)
    start, end = past_trajectory.start_time.time_us, past_trajectory.end_time.time_us
    t_us = np.clip(np.arange(start, end, int(0.1 * 1e6), dtype=np.int64), start, end)
    hist = ego_states_to_state_array(past_trajectory.get_state_at_times([TimePoint(int(t)) for t in t_us[:-1]]))
    n = len(hist) + states.shape[1]
    pad = np.zeros((len(states), n, StateIndex.size()), np.float64)
    pad[:, :len(hist)] = hist
    pad[:, len(hist):] = states
    return ego_is_comfortable(pad, np.arange(n, dtype=np.float64) * interval).all(-1).astype(np.float32)


def proposal_labels(cache_path, poses, tracks):
    """Labels of the v2 scorer for proposals poses (P, 8, 3), with tracks = the t0 GT track tokens of the sample (the
    dataset's gt boxes, in order). Same simulation and train scorer as proposal_scores, plus:
      sub (P,7)          as proposal_scores (nc, dac, ep, ttc, comfort, ddc, final; DrivoR's labels)
      ep_raw (P,)        raw progress normalized by the PDM-Closed progress, clipped to [0, 1], WITHOUT the safety mask
                         of DrivoR's EP label (1 when both progresses are below the progress threshold)
      hcomf (P,)         history comfort (official v2.2 definition)
      flags (P,8,4)      ego area along the proposal at the 8 poses: FLAG_NAMES (PDM's _ego_areas + intersection layer)
      fault (P,M)        1 if the proposal has an at-fault collision with track m (PDM's own at-fault logic; tracks the
                         observation already counts as collided at t0 are ignored)
      ttc_pair (P,M)     1 if the proposal violates the TTC rule because of track m
      ttc_ok (P,)        False where the train scorer's TTC failed (its fallback value 2.0): ttc_pair unknown
      fault_unmatched (P,) at-fault collisions with objects that are not among the tracks (not representable by slots)
      ttc_unmatched (P,) TTC violations caused by objects that are not among the tracks
    """
    mc, pdm_progress, sub = _score(cache_path, poses)
    P = len(poses)
    raw = scorer._progress_raw.astype(np.float64)
    den = np.maximum(raw, float(np.asarray(pdm_progress).reshape(-1)[0]))
    ep_raw = np.where(den > scorer._config.progress_distance_threshold, np.clip(raw / np.maximum(den, 1e-9), 0.0, 1.0), 1.0)
    areas = scorer._ego_areas[:, POSE_IDX, :3]
    inter = np.array([[scorer._drivable_area_map.is_in_layer(
        StateSE2(*scorer._states[p, t, StateIndex.STATE_SE2]).point, layer=SemanticMapLayer.INTERSECTION)
        for t in POSE_IDX] for p in range(P)], bool)
    flags = np.concatenate([areas, inter[..., None]], -1).astype(np.uint8)
    init = set(mc.observation.collided_track_ids)
    col = {t: j for j, t in enumerate(tracks)}
    fault = np.zeros((P, len(tracks)), np.uint8)
    unmatched = np.zeros(P, np.int16)
    for p in range(P):
        for tok in set(scorer.proposal_fault_collided_track_ids[p]) - init:
            j = col.get(tok)
            if j is None:
                unmatched[p] += 1
            else:
                fault[p, j] = 1
    ttc_ok = scorer._weighted_metrics[WeightedMetricIndex.TTC] != 2.0
    ttc_pair = np.zeros((P, len(tracks)), np.uint8)
    ttc_unmatched = np.zeros(P, np.int16)
    if hasattr(scorer, "ttc_collided_track_ids"):
        for p in range(P):
            for tok in set(scorer.ttc_collided_track_ids[p]) - init:
                j = col.get(tok)
                if j is None:
                    ttc_unmatched[p] += 1
                else:
                    ttc_pair[p, j] = 1
    hcomf = history_comfort(scorer._states, getattr(mc, "past_human_trajectory", None),
                            simulator.proposal_sampling.interval_length)
    return {"sub": sub, "ep_raw": ep_raw.astype(np.float32), "hcomf": hcomf, "flags": flags, "fault": fault,
            "ttc_pair": ttc_pair, "ttc_ok": ttc_ok, "fault_unmatched": unmatched, "ttc_unmatched": ttc_unmatched}
