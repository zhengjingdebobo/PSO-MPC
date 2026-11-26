import numpy as np
import pytest

from pso_mpc_parking.config import VehicleConfig
from pso_mpc_parking.config.settings import (
    PSOConfig,
    ClosedLoopConfig,
)
from pso_mpc_parking.models import VehicleModel
from pso_mpc_parking.environment import (
    ParkingEnvironment,
    CollisionChecker,
)
from pso_mpc_parking.environment.parking_env import Obstacle
from pso_mpc_parking.controllers.evaluator import TrajectoryEvaluator
from pso_mpc_parking.controllers.pso import PSOOptimizer, PSOResult
from pso_mpc_parking.controllers.mpc import MPCController, MPCResult
from pso_mpc_parking.simulation.closed_loop import ClosedLoopSimulator


def make_simulator(max_steps=3, dt=0.05):
    vehicle = VehicleConfig()
    model = VehicleModel(vehicle)
    environment = ParkingEnvironment(vehicle_config=vehicle)
    checker = CollisionChecker(environment, vehicle)

    evaluator = TrajectoryEvaluator(
        model,
        checker,
        dt=dt,
        Q=np.eye(5),
        R=np.eye(2),
        S=np.eye(2),
        Qf=np.eye(5),
    )

    optimizer = PSOOptimizer(
        evaluator,
        PSOConfig(
            horizon=2,
            particle_count=4,
            max_iterations=2,
            random_seed=7,
        ),
    )

    controller = MPCController(optimizer)
    simulator = ClosedLoopSimulator(
        controller,
        ClosedLoopConfig(max_steps=max_steps),
    )

    return simulator, model, environment


def make_decision(control=None):
    """
    构造可控的 MPC 返回值，只用于检查闭环执行逻辑。
    不在这里检验 PSO 的预测和成本。
    """
    success = control is not None

    optimization = PSOResult(
        success=success,
        cost=1.0 if success else float("inf"),
        controls=(
            np.tile(control, (2, 1))
            if success else None
        ),
        states=np.zeros((3, 5)) if success else None,
        cost_history=np.array([
            1.0 if success else np.inf
        ]),
        evaluations=1,
        iterations=1,
    )

    return MPCResult(
        success=success,
        control=(
            np.array(control, dtype=float)
            if success else None
        ),
        reason="ok" if success else "no_feasible_sequence",
        optimization=optimization,
    )


def test_initially_parked_needs_no_control(monkeypatch):
    simulator, model, environment = make_simulator()
    target = environment.target_state(model.config)

    def forbidden(**kwargs):
        pytest.fail("已经停好时，不应调用 MPC")

    monkeypatch.setattr(
        simulator.controller,
        "compute_control",
        forbidden,
    )

    result = simulator.run(target, target)

    assert result.success
    assert result.reason == "parked"
    assert result.simulation.states.shape == (1, 5)
    assert result.simulation.controls.shape == (0, 2)
    assert result.solve_times.size == 0


@pytest.mark.parametrize(
    "initial_state, expected_reason",
    [
        (
            [4.5, -1.25, 0, 0, 0],
            "initial_collision_or_boundary",
        ),
        (
            [0, 5.0, 0, 0, 0],
            "initial_safety_distance",
        ),
    ],
)
def test_unsafe_initial_state_stops_before_control(
    monkeypatch, initial_state, expected_reason
):
    simulator, model, environment = make_simulator()

    def forbidden(**kwargs):
        pytest.fail("初态不安全时，不应调用 MPC")

    monkeypatch.setattr(
        simulator.controller,
        "compute_control",
        forbidden,
    )

    result = simulator.run(
        initial_state,
        environment.target_state(model.config),
    )

    assert not result.success
    assert result.reason == expected_reason
    assert len(result.simulation.controls) == 0
    assert result.solve_times.size == 0


def test_failed_solve_does_not_execute_control(monkeypatch):
    simulator, model, environment = make_simulator()

    monkeypatch.setattr(
        simulator.controller,
        "compute_control",
        lambda **kwargs: make_decision(),
    )

    initial_state = np.array([0, 2, 0, 0.4, 0], dtype=float)

    result = simulator.run(
        initial_state,
        environment.target_state(model.config),
        previous_control=[0.1, 0.0],
    )

    assert not result.success
    assert result.reason == "no_feasible_sequence"
    assert len(result.simulation.controls) == 0

    # 失败求解仍记录耗时和成本。
    assert result.solve_times.shape == (1,)
    assert np.isinf(result.optimization_costs[0])

    np.testing.assert_allclose(model.state, initial_state)
    np.testing.assert_allclose(
        simulator.controller.previous_control,
        [0.1, 0.0],
    )


def test_feedback_records_and_max_steps(monkeypatch):
    simulator, model, environment = make_simulator(max_steps=3)
    calls = []

    def fixed_control(current_state, reference):
        calls.append((
            current_state.copy(),
            simulator.controller.previous_control,
        ))
        return make_decision([0.2, 0.0])

    monkeypatch.setattr(
        simulator.controller,
        "compute_control",
        fixed_control,
    )

    result = simulator.run(
        [0, 2, 0, 0, 0],
        environment.target_state(model.config),
    )

    data = result.simulation

    assert not result.success
    assert result.reason == "max_steps"
    assert len(calls) == 3

    assert data.states.shape == (4, 5)
    assert data.controls.shape == (3, 2)
    assert len(data.collisions) == 4
    assert len(data.segment_checks) == 3

    np.testing.assert_allclose(
        data.times,
        [0.0, 0.05, 0.10, 0.15],
    )

    # 每次 MPC 都应收到执行后的最新状态。
    for i in range(3):
        np.testing.assert_allclose(
            calls[i][0],
            data.states[i],
        )

        expected_previous = (
            np.zeros(2) if i == 0 else data.controls[i - 1]
        )
        np.testing.assert_allclose(
            calls[i][1],
            expected_previous,
        )

    # 恒定加速度 0.2，持续 0.15 秒。
    assert data.states[-1, 3] == pytest.approx(0.03)
    assert data.states[-1, 0] == pytest.approx(0.00225)

    assert result.solve_times.shape == (3,)
    assert np.all(result.solve_times >= 0)
    np.testing.assert_allclose(
        result.optimization_costs,
        [1.0, 1.0, 1.0],
    )


def test_success_after_execution(monkeypatch):
    simulator, model, environment = make_simulator(dt=0.1)
    target = environment.target_state(model.config)

    initial_state = target.copy()
    initial_state[4] = 0.08  # 初始前轮尚未回正到允许范围。

    monkeypatch.setattr(
        simulator.controller,
        "compute_control",
        lambda **kwargs: make_decision([0.0, -0.5]),
    )

    result = simulator.run(initial_state, target)

    assert result.success
    assert result.reason == "parked"
    assert len(result.simulation.controls) == 1
    assert result.simulation.states[-1, 4] == pytest.approx(0.03)


def test_collision_between_endpoints_stops_run(monkeypatch):
    # 长时间步仅用于检验跨步碰撞，不是正式实验参数。
    simulator, model, environment = make_simulator(dt=8.0)

    environment.obstacles = (
        Obstacle("box", (0.0, 2.0), 0.2, 1.0),
    )

    monkeypatch.setattr(
        simulator.controller,
        "compute_control",
        lambda **kwargs: make_decision([0.0, 0.0]),
    )

    result = simulator.run(
        [-5, 2, 0, 1, 0],
        environment.target_state(model.config),
    )

    data = result.simulation

    assert not result.success
    assert result.reason == "collision_or_boundary"
    assert data.stopped_on_collision
    assert len(data.controls) == 1

    # 起点、终点都无碰撞，但中途穿过障碍。
    assert not data.collisions[0].collision
    assert not data.collisions[-1].collision
    assert data.segment_checks[0].collision


def test_safety_distance_violation_stops_run(monkeypatch):
    simulator, model, environment = make_simulator(dt=0.1)

    monkeypatch.setattr(
        simulator.controller,
        "compute_control",
        lambda **kwargs: make_decision([0.0, 0.0]),
    )

    result = simulator.run(
        [0, 2.15, np.pi / 2, 1, 0],
        environment.target_state(model.config),
    )

    assert not result.success
    assert result.reason == "safety_distance"
    assert not result.simulation.stopped_on_collision
    assert len(result.simulation.controls) == 1


def test_parking_requires_inside_slot_and_correct_state():
    simulator, model, environment = make_simulator()
    target = environment.target_state(model.config)

    assert simulator.is_parked(target, target)

    # 状态与参考位置一致，但根本不在停车位内。
    outside_slot = np.array([0, 2, 0, 0, 0], dtype=float)
    assert not simulator.is_parked(outside_slot, outside_slot)

    moving = target.copy()
    moving[3] = 0.2
    assert not simulator.is_parked(moving, target)

    steered = target.copy()
    steered[4] = 0.1
    assert not simulator.is_parked(steered, target)

    # 航向角应正确处理周期性。
    wrapped_heading = target.copy()
    wrapped_heading[2] = 2 * np.pi - 0.01
    assert simulator.is_parked(wrapped_heading, target)