import numpy as np
import pytest

from pso_mpc_parking.config import VehicleConfig
from pso_mpc_parking.models import VehicleModel
from pso_mpc_parking.environment import (
    ParkingEnvironment,
    CollisionChecker,
)
from pso_mpc_parking.controllers.evaluator import TrajectoryEvaluator


def make_evaluator():
    """创建使用相同车辆配置的模型和碰撞器。"""
    vehicle = VehicleConfig()
    model = VehicleModel(vehicle)
    environment = ParkingEnvironment(vehicle_config=vehicle)
    checker = CollisionChecker(environment, vehicle)

    evaluator = TrajectoryEvaluator(
        model,
        checker,
        dt=0.05,
        Q=np.eye(5),
        R=np.eye(2),
        S=np.eye(2),
        Qf=np.eye(5),
    )

    return evaluator, model


def test_safe_stationary_candidate():
    """安全位置静止，且已经达到参考状态，成本为零。"""
    evaluator, _ = make_evaluator()
    state = np.array([0.0, 2.0, 0.0, 0.0, 0.0])

    result = evaluator.evaluate(
        initial_state=state,
        controls=np.zeros((2, 2)),
        reference=state,
        previous_control=np.zeros(2),
    )

    assert result.feasible
    assert result.reason == "ok"
    assert result.cost == pytest.approx(0.0)
    assert result.states.shape == (3, 5)
    assert result.controls.shape == (2, 2)


@pytest.mark.parametrize(
    "state, expected_reason",
    [
        # 与前方障碍车重叠。
        ([4.5, -1.25, 0, 0, 0], "collision_or_boundary"),

        # 车身上侧超过 y=6 的边界。
        ([0, 5.5, 0, 0, 0], "collision_or_boundary"),

        # 不触边，但上侧与边界仅有 0.1 m 间距。
        ([0, 5.0, 0, 0, 0], "safety_distance"),
    ],
)
def test_infeasible_candidates(state, expected_reason):
    evaluator, _ = make_evaluator()

    result = evaluator.evaluate(
        initial_state=state,
        controls=np.zeros((1, 2)),
        reference=np.zeros(5),
        previous_control=np.zeros(2),
    )

    assert not result.feasible
    assert np.isinf(result.cost)
    assert result.reason == expected_reason


def test_automatic_saturation_is_allowed():
    """触及速度和转角上限，不应单独导致不可行。"""
    evaluator, model = make_evaluator()
    config = model.config

    result = evaluator.evaluate(
        initial_state=[0, 2, 0, 1.99, config.max_steering],
        controls=[[100.0, 100.0]],
        reference=np.zeros(5),
        previous_control=np.zeros(2),
    )

    assert result.feasible
    assert np.isfinite(result.cost)

    np.testing.assert_allclose(
        result.controls[0],
        [config.max_acceleration, config.max_steering_rate],
    )

    assert result.states[-1, 3] == pytest.approx(
        config.max_forward_speed
    )
    assert result.states[-1, 4] == pytest.approx(
        config.max_steering
    )


def test_evaluation_has_no_side_effects():
    """评价不能推进真实车辆，也不能修改传入的数据。"""
    evaluator, model = make_evaluator()

    initial_state = np.array([0.0, 2.0, 0.0, -0.5, 0.1])
    controls = np.zeros((3, 2))
    reference = np.zeros(5)
    previous_control = np.zeros(2)

    model_before = model.state
    arrays = [
        initial_state,
        controls,
        reference,
        previous_control,
    ]
    snapshots = [array.copy() for array in arrays]

    evaluator.evaluate(
        initial_state,
        controls,
        reference,
        previous_control,
    )

    np.testing.assert_array_equal(model.state, model_before)

    for array, snapshot in zip(arrays, snapshots):
        np.testing.assert_array_equal(array, snapshot)


@pytest.mark.parametrize(
    "controls",
    [
        np.empty((0, 2)),
        np.zeros(2),
        np.array([[np.nan, 0.0]]),
    ],
)
def test_invalid_controls(controls):
    evaluator, _ = make_evaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(
            initial_state=[0, 2, 0, 0, 0],
            controls=controls,
            reference=np.zeros(5),
            previous_control=np.zeros(2),
        )


def test_mismatched_vehicle_configuration_is_rejected():
    """防止模型和碰撞器分别使用不同的车辆配置。"""
    model = VehicleModel(
        VehicleConfig(max_forward_speed=1.0)
    )
    checker = CollisionChecker(ParkingEnvironment())

    with pytest.raises(ValueError):
        TrajectoryEvaluator(
            model,
            checker,
            dt=0.05,
            Q=np.eye(5),
            R=np.eye(2),
            S=np.eye(2),
            Qf=np.eye(5),
        )