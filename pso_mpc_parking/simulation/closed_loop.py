from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pso_mpc_parking.config.settings import ClosedLoopConfig
from pso_mpc_parking.controllers.mpc import MPCController
from pso_mpc_parking.environment.geometry import vehicle_vertices
from pso_mpc_parking.models.vehicle_model import (finite_vector,wrap_angle,)
from pso_mpc_parking.simulation.simulator import SimulationResult

@dataclass(frozen=True)
class ClosedLoopResult:
    success: bool
    reason: str
    simulation: SimulationResult

    solve_times: NDArray[np.float64]
    optimization_costs: NDArray[np.float64]

class ClosedLoopSimulator:
    def __init__(
            self,
            controller: MPCController,
            config: ClosedLoopConfig | None = None,
    ) -> None:
        self.controller = controller
        self.config = config or ClosedLoopConfig()

        evalutor = controller.optimizer.evaluator
        self.model = evalutor.model
        self.checker = evalutor.checker
        self.environment = self.checker.environment

        if self.model.config != self.checker.vehicle_config:
            raise ValueError("车辆模型与碰撞器配置不一致")

    def is_parked(
            self,
            state: ArrayLike,
            reference: ArrayLike,
    ) -> bool:
        state = finite_vector(state, 5,"state")
        reference = finite_vector(reference, 5, "reference")

        if not np.allclose(
            reference[3:], 0.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("泊车目标的速度和前轮转角必须为零")

        config = self.config

        position_error = np.linalg.norm(
            state[:2] - reference[:2]
        )
        heading_error = abs(wrap_angle(state[2] - reference[2]))

        errors_satisfied = (
            position_error <= config.position_tolerance
            and heading_error <= config.heading_tolerance
            and abs(state[3]) <= config.steering_tolerance
            and abs(state[4]) <= config.steering_tolerance
        )

        body = vehicle_vertices(state, self.model.config)
        slot = self.environment.slot_vertices

        inside_slot = bool(
            np.all(body >= slot.min(axis= 0) -1e-9)
            and np.all(body <= slot.max(axis= 0)+ 1e-9)
        )

        safe = self.checker.check_state(state).safe

        return bool(errors_satisfied and inside_slot and safe)

    def run(
            self,
            initial_state: ArrayLike,
            reference: ArrayLike,
            previous_control: ArrayLike | None = None,
    ) -> ClosedLoopResult:
        reference = finite_vector(reference, 5, "reference")

        if not np.allclose(
            reference[3:], 0.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("泊车目标的速度和前轮转角必须为0")

        state = self.model.reset(initial_state)
        self.controller.reset(previous_control)
        dt = self.controller.dt

        states = [state]
        controls = []
        collisions  = [self.checker.check_state(state)]
        segment_checks = []
        solve_times = []
        optimization_costs = []

        def finish(success: bool, reason: str) ->ClosedLoopResult:
            stopped_on_collision = (
                any(item.collision for item in collisions)
                or any(item.collision for item in segment_checks)
            )

            simulation = SimulationResult(
                times = np.arange(len(states),dtype = float) * dt,
                states = np.asarray(states, dtype = float),
                controls = np.asarray(controls, dtype = float).reshape(-1,2),
                collisions = tuple(collisions),
                segment_checks = tuple(segment_checks),
                stopped_on_collision = bool(stopped_on_collision),
            )

            return ClosedLoopResult(
                success = success,
                reason = reason,
                simulation = simulation,
                solve_times = np.asarray(solve_times, dtype = float),
                optimization_costs = np.asarray(optimization_costs,dtype=float),
            )

        if collisions[0].collision:
            return finish(False,"initial_collision_or_boundary")

        if collisions[0].safety_violation:
            return finish(False, "initial_safety_distance")

        if self.is_parked(state, reference):
            return finish(True,"parked")

        for _ in range(self.config.max_steps):
            start_time = perf_counter()

            decision = self.controller.compute_control(
                current_state = states[-1],
                reference = reference,
            )

            solve_times.append(perf_counter() - start_time)
            optimization_costs.append(float(decision.optimization.cost))

            if not decision.success:
                return finish(False, decision.reason)

            if decision.control is None:
                raise ValueError("MPC 报告成功，但没有返回控制输入")

            control = self.model.constrain_control(decision.control)
            previous_state = states[-1]

            state = self.model.step(control,dt,method="rk4",)

            self.controller.record_applied_control(control)

            segment = self.checker.check_trajectory(np.stack([previous_state, state]))
            collision = self.checker.check_state(state)

            controls.append(control)
            states.append(state)
            segment_checks.append(segment)
            collisions.append(collision)

            if segment.collision or collision.collision:
                return finish(False,"collision_or_boundary")

            if segment.safety_violation or collision.safety_violation:
                return finish(False, "safety_distance")

            if self.is_parked(state, reference):
                return finish(True,"parked")

        return finish(False, "max_steps")