"""NAVSIM agent that returns trajectories computed beforehand, so the official PDM scorer can grade any dumped plans.

The plans file holds token[N] and plan[N, 8, 3]: (x, y, heading) at 0.5-4 s in the
t0 ego rear-axle frame, the NAVSIM trajectory convention. A token missing from the file raises, so the devkit marks the
scene failed instead of scoring a made-up trajectory.
"""

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory


class PrecomputedAgent(AbstractAgent):
    def __init__(self, plans_path: str,
                 trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)):
        super().__init__(trajectory_sampling, requires_scene=True)  # the scene only for its token, never its future
        self._plans_path = plans_path
        self._plans = None

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        z = np.load(self._plans_path)
        assert z["plan"].shape[1:] == (self._trajectory_sampling.num_poses, 3)
        self._plans = dict(zip(z["token"].tolist(), z["plan"].astype(np.float32)))

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        return Trajectory(self._plans[scene.scene_metadata.initial_token], self._trajectory_sampling)
