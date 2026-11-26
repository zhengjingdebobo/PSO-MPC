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
from pso_mpc_parking.controllers.pso import PSOOptimizer, PSOResult
from pso_mpc_parking.controllers.mpc import MPCController


def make_controller():
    vehicle = VehicleConfig()

    model = VehicleModel(
        vehicle,
        initial_state=[0, 2, 0, 0, 0],
    )

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

    return controller, model


def make_success_result():
    """
    用于验证 MPC 接口的可控返回值。

    这里不检验 PSO 或代价计算，它们由各自测试负责。
    """
    return PSOResult(
        success=True,
        cost=1.0,
        controls=np.array([
            [0.2, 0.1],
            [-0.3, -0.1],
        ]),
        states=np.zeros((3, 5)),
        cost_history=np.array([2.0, 1.0]),
        evaluations=8,
        iterations=1,
    )


def make_failure_result():
    return PSOResult(
        success=False,
        cost=float("inf"),
        controls=None,
        states=None,
        cost_history=np.array([np.inf]),
        evaluations=4,
        iterations=1,
    )


def test_returns_only_first_input(monkeypatch):
    controller, model = make_controller()
    optimization = make_success_result()

    monkeypatch.setattr(
        controller.optimizer,
        "optimize",
        lambda **kwargs: optimization,
    )

    model_before = model.state

    decision = controller.compute_control(
        current_state=model.state,
        reference=[1, 2, 0, 0, 0],
    )

    assert decision.success
    assert decision.reason == "ok"
    assert decision.control.shape == (2,)

    np.testing.assert_allclose(
        decision.control,
        [0.2, 0.1],
    )

    # 求解不能推进真实车辆。
    np.testing.assert_array_equal(model.state, model_before)

    # 尚未执行，因此上一周期输入仍为零。
    np.testing.assert_array_equal(
        controller.previous_control,
        np.zeros(2),
    )

    # 修改返回输入不能改变 PSO 原控制序列。
    decision.control[:] = 0.0

    np.testing.assert_allclose(
        optimization.controls[0],
        [0.2, 0.1],
    )


def test_feedback_and_previous_input_are_forwarded(monkeypatch):
    controller, model = make_controller()
    calls = []

    def fake_optimize(**kwargs):
        calls.append({
            name: value.copy()
            for name, value in kwargs.items()
        })
        return make_success_result()

    monkeypatch.setattr(
        controller.optimizer,
        "optimize",
        fake_optimize,
    )

    reference = np.array([1, 2, 0, 0, 0], dtype=float)

    first = controller.compute_control(
        model.state,
        reference,
    )

    np.testing.assert_array_equal(
        calls[0]["previous_control"],
        np.zeros(2),
    )

    # 模拟外部仿真模块执行一个控制周期。
    model.step(first.control, controller.dt, method="rk4")
    controller.record_applied_control(first.control)

    updated_state = model.state

    controller.compute_control(
        updated_state,
        reference,
    )

    assert len(calls) == 2

    # 第二次优化必须使用反馈后的新状态。
    np.testing.assert_array_equal(
        calls[1]["initial_state"],
        updated_state,
    )

    # 第二次优化使用上一轮真正执行过的输入。
    np.testing.assert_array_equal(
        calls[1]["previous_control"],
        first.control,
    )


def test_failure_returns_no_control(monkeypatch):
    controller, model = make_controller()

    controller.record_applied_control([0.1, 0.0])
    previous_before = controller.previous_control
    model_before = model.state

    monkeypatch.setattr(
        controller.optimizer,
        "optimize",
        lambda **kwargs: make_failure_result(),
    )

    decision = controller.compute_control(
        model.state,
        [1, 2, 0, 0, 0],
    )

    assert not decision.success
    assert decision.control is None
    assert decision.reason == "no_feasible_sequence"

    np.testing.assert_array_equal(
        controller.previous_control,
        previous_before,
    )
    np.testing.assert_array_equal(
        model.state,
        model_before,
    )


def test_previous_input_copy_limits_and_reset():
    controller, model = make_controller()

    controller.record_applied_control([100.0, -100.0])

    expected = np.array([
        model.config.max_acceleration,
        -model.config.max_steering_rate,
    ])

    np.testing.assert_allclose(
        controller.previous_control,
        expected,
    )

    # 读取到的是副本。
    copied = controller.previous_control
    copied[:] = 0.0

    np.testing.assert_allclose(
        controller.previous_control,
        expected,
    )

    controller.reset()

    np.testing.assert_array_equal(
        controller.previous_control,
        np.zeros(2),
    )


def test_invalid_success_result_is_rejected(monkeypatch):
    controller, model = make_controller()

    invalid_result = make_failure_result()

    # 构造一个错误的“成功但没有控制序列”的返回值。
    invalid_result = PSOResult(
        success=True,
        cost=1.0,
        controls=None,
        states=None,
        cost_history=invalid_result.cost_history,
        evaluations=4,
        iterations=1,
    )

    monkeypatch.setattr(
        controller.optimizer,
        "optimize",
        lambda **kwargs: invalid_result,
    )

    with pytest.raises(ValueError):
        controller.compute_control(
            model.state,
            [1, 2, 0, 0, 0],
        )


def test_invalid_current_state():
    controller, _ = make_controller()

    with pytest.raises(ValueError):
        controller.compute_control(
            current_state=[0, 2],
            reference=[1, 2, 0, 0, 0],
        )


def test_real_pso_integration():
    """使用真实 PSO 检查模块连接，不要求完成泊车。"""
    controller, model = make_controller()
    model_before = model.state

    decision = controller.compute_control(
        model.state,
        [1, 2, 0, 0, 0],
    )

    assert decision.success
    assert decision.control is not None
    assert np.all(np.isfinite(decision.control))

    np.testing.assert_array_equal(
        decision.control,
        decision.optimization.controls[0],
    )

    np.testing.assert_array_equal(
        model.state,
        model_before,
    )