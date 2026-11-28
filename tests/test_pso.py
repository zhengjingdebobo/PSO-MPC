import numpy as np
import pytest

from pso_mpc_parking.config import VehicleConfig
from pso_mpc_parking.config.settings import PSOConfig
from pso_mpc_parking.models import VehicleModel
from pso_mpc_parking.environment import (
    ParkingEnvironment,
    CollisionChecker,
)
from pso_mpc_parking.controllers.evaluator import TrajectoryEvaluator
from pso_mpc_parking.controllers.pso import PSOOptimizer


def make_optimizer(seed=7):
    vehicle = VehicleConfig()
    model = VehicleModel(vehicle)

    environment = ParkingEnvironment(
        vehicle_config=vehicle
    )
    checker = CollisionChecker(environment, vehicle)

    evaluator = TrajectoryEvaluator(
        model,
        checker,
        dt=0.05,
        Q=np.eye(5),
        R=np.eye(2),
        S=np.eye(2),
        Qf=5.0 * np.eye(5),
    )

    config = PSOConfig(
        horizon=3,
        particle_count=8,
        max_iterations=4,
        random_seed=seed,
    )

    optimizer = PSOOptimizer(evaluator, config)

    return optimizer, evaluator, model


def test_search_returns_feasible_result():
    """在宽松场景下，应找到可行控制序列。"""
    optimizer, evaluator, model = make_optimizer()

    initial_state = np.array([0, 2, 0, 0, 0], dtype=float)
    reference = np.array([1, 2, 0, 0, 0], dtype=float)
    previous_control = np.zeros(2)

    result = optimizer.optimize(
        initial_state,
        reference,
        previous_control,
    )

    assert result.success
    assert np.isfinite(result.cost)
    assert result.controls is not None
    assert result.states is not None

    assert result.controls.shape == (3, 2)
    assert result.states.shape == (4, 5)

    vehicle = model.config

    assert np.all(
        np.abs(result.controls[:, 0])
        <= vehicle.max_acceleration
    )
    assert np.all(
        np.abs(result.controls[:, 1])
        <= vehicle.max_steering_rate
    )

    # 返回结果应与已有评价器一致。
    checked = evaluator.evaluate(
        initial_state=initial_state,
        controls=result.controls,
        reference=reference,
        previous_control=previous_control,
    )

    assert checked.feasible
    assert result.cost == pytest.approx(checked.cost)
    np.testing.assert_allclose(
        result.states,
        checked.states,
    )

    # 初始化评价一次，之后每轮更新再评价一次。
    assert result.evaluations == 8 * (4 + 1)
    assert result.iterations == 4

    # Gbest 历史成本只能保持或下降。
    assert result.cost_history.shape == (5,)
    assert np.all(np.diff(result.cost_history) <= 1e-12)
    assert result.cost_history[-1] == pytest.approx(result.cost)


def test_warm_start_is_used_as_first_particle(monkeypatch):
    optimizer, evaluator, _ = make_optimizer()
    warm_start = np.zeros((optimizer.config.horizon, 2))
    original_evaluate = evaluator.evaluate
    evaluated_controls = []

    def capture_evaluate(**kwargs):
        evaluated_controls.append(kwargs["controls"].copy())
        return original_evaluate(**kwargs)

    monkeypatch.setattr(evaluator, "evaluate", capture_evaluate)
    result = optimizer.optimize(
        initial_state=[0, 2, 0, 0, 0],
        reference=[1, 2, 0, 0, 0],
        previous_control=[0, 0],
        warm_start=warm_start,
    )

    assert result.success
    np.testing.assert_array_equal(evaluated_controls[0], warm_start)


def test_same_seed_is_reproducible():
    """相同配置和种子，新建的两个优化器应产生相同结果。"""
    optimizer_a, _, _ = make_optimizer(seed=10)
    optimizer_b, _, _ = make_optimizer(seed=10)

    arguments = {
        "initial_state": [0, 2, 0, 0, 0],
        "reference": [1, 2, 0, 0, 0],
        "previous_control": [0, 0],
    }

    result_a = optimizer_a.optimize(**arguments)
    result_b = optimizer_b.optimize(**arguments)

    assert result_a.success
    assert result_b.success

    np.testing.assert_array_equal(
        result_a.controls,
        result_b.controls,
    )
    np.testing.assert_array_equal(
        result_a.cost_history,
        result_b.cost_history,
    )


def test_search_does_not_modify_vehicle_or_inputs():
    """搜索只能预测，不能执行车辆控制。"""
    optimizer, _, model = make_optimizer()

    initial_state = np.array([0, 2, 0, 0, 0], dtype=float)
    reference = np.array([1, 2, 0, 0, 0], dtype=float)
    previous_control = np.zeros(2)

    vehicle_before = model.state

    arrays = [
        initial_state,
        reference,
        previous_control,
    ]
    snapshots = [array.copy() for array in arrays]

    optimizer.optimize(
        initial_state,
        reference,
        previous_control,
    )

    np.testing.assert_array_equal(
        model.state,
        vehicle_before,
    )

    for array, snapshot in zip(arrays, snapshots):
        np.testing.assert_array_equal(array, snapshot)


def test_all_infeasible_returns_failure():
    """初始状态已经碰撞时，不能返回可执行的假最优解。"""
    optimizer, _, _ = make_optimizer()

    result = optimizer.optimize(
        initial_state=[4.5, -1.25, 0, 0, 0],
        reference=[-1.35, -1.25, 0, 0, 0],
        previous_control=[0, 0],
    )

    assert not result.success
    assert np.isinf(result.cost)
    assert result.controls is None
    assert result.states is None
    assert np.all(np.isinf(result.cost_history))

    # 应在有限搜索次数内结束。
    assert result.evaluations == 8 * (4 + 1)


@pytest.mark.parametrize(
    "parameters",
    [
        {"horizon": 0},
        {"particle_count": 0},
        {"max_iterations": 0},
        {"velocity_fraction": 0.0},
        {"inertia_weight": float("nan")},
        {"random_seed": -1},
    ],
)
def test_invalid_configuration(parameters):
    with pytest.raises(ValueError):
        PSOConfig(**parameters)
