from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from pso_mpc_parking.config import SimulationConfig
from pso_mpc_parking.environment.collision import CollisionChecker, CollisionResult, TrajectoryCollisionResult
from pso_mpc_parking.models.vehicle_model import FloatArray, VehicleModel


@dataclass(frozen=True)
class SimulationResult:
    times: FloatArray  # (N+1,)
    states: FloatArray  # (N+1, 5), rear-axle states
    controls: FloatArray  # (N, 2), clipped requested controls; bounds may hold v/delta
    collisions: tuple[CollisionResult, ...]
    segment_checks: tuple[TrajectoryCollisionResult, ...]
    stopped_on_collision: bool


class Simulator:
    def __init__(self, model: VehicleModel, checker: CollisionChecker,
                 config: SimulationConfig | None = None) -> None:
        self.model, self.checker = model, checker
        self.config = config or SimulationConfig()

    def run(self, controls: ArrayLike, initial_state: ArrayLike | None = None,
            stop_on_collision: bool = True) -> SimulationResult:
        """Execute an externally supplied open-loop sequence; no controller.

        If a segment collides, its endpoint is recorded and the run terminates.
        This detects a crossing; it does not rewind to the precise contact time.
        """
        u = np.asarray(controls, dtype=float)
        if u.ndim != 2 or u.shape[1] != 2 or not np.all(np.isfinite(u)):
            raise ValueError("controls must be finite with shape (N, 2)")
        initial = self.model.state if initial_state is None else self.model.reset(initial_state)
        states, used, segments = [initial], [], []
        collisions = [self.checker.check_state(initial)]
        stopped = stop_on_collision and collisions[0].collision
        if not stopped:
            for control in u:
                clipped = self.model.constrain_control(control)
                state = self.model.step(clipped, self.config.dt, self.config.integration_method)
                segment = self.checker.check_trajectory(np.array([states[-1], state]))
                states.append(state)
                used.append(clipped)
                segments.append(segment)
                collisions.append(self.checker.check_state(state))
                if stop_on_collision and segment.collision:
                    stopped = True
                    break
        return SimulationResult(np.arange(len(states)) * self.config.dt, np.array(states),
                                np.asarray(used).reshape(-1, 2), tuple(collisions),
                                tuple(segments), bool(stopped))
