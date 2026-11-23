import numpy as np
import pytest

from pso_mpc_parking.controllers.objective import compute_objective


def make_weights(q=0.0, r=0.0, s=0.0, qf=0.0):
    """仅为测试方便，生成简单的对角权重矩阵。"""
    return {
        "Q": q * np.eye(5),
        "R": r * np.eye(2),
        "S": s * np.eye(2),
        "Qf": qf * np.eye(5),
    }


def test_zero_cost_at_target():
    """状态、输入及输入变化都为零时，成本应为零。"""
    cost = compute_objective(
        states=np.zeros((3, 5)),
        controls=np.zeros((2, 2)),
        reference=np.zeros(5),
        previous_control=np.zeros(2),
        **make_weights(q=1, r=1, s=1, qf=1),
    )

    assert cost == pytest.approx(0.0)


def test_cost_matches_hand_calculation():
    """用可以手算的例子检查四项成本。"""
    states = np.array([
        [2.0, 0, 0, 0, 0],  # 阶段状态
        [3.0, 0, 0, 0, 0],  # 终端状态
    ])

    cost = compute_objective(
        states=states,
        controls=[[0.5, 0.0]],
        reference=np.zeros(5),
        previous_control=[0.1, 0.0],
        **make_weights(q=1, r=2, s=1, qf=3),
    )

    # 阶段状态：2² = 4
    # 控制输入：2 × 0.5² = 0.5
    # 控制变化：(0.5 - 0.1)² = 0.16
    # 终端状态：3 × 3² = 27
    assert cost == pytest.approx(31.66)


def test_heading_error_uses_shortest_angle():
    """179° 与 -179° 的航向差应为 2°。"""
    states = np.zeros((2, 5))
    states[0, 2] = np.deg2rad(179)

    reference = np.zeros(5)
    reference[2] = np.deg2rad(-179)

    cost = compute_objective(
        states=states,
        controls=np.zeros((1, 2)),
        reference=reference,
        previous_control=np.zeros(2),
        **make_weights(q=1),
    )

    assert cost == pytest.approx(np.deg2rad(2) ** 2)


def test_first_control_change_uses_previous_input():
    """首个控制变化量必须考虑上一周期的输入。"""
    cost = compute_objective(
        states=np.zeros((3, 5)),
        controls=[[0.3, 0.0], [0.3, 0.0]],
        reference=np.zeros(5),
        previous_control=[0.1, 0.0],
        **make_weights(s=1),
    )

    # 第一步变化为 0.2，第二步变化为 0。
    assert cost == pytest.approx(0.04)


def test_inputs_are_not_modified():
    """计算成本不能改变传入的数据。"""
    states = np.zeros((2, 5))
    states[:, 2] = 4.0
    controls = np.array([[0.2, 0.1]])
    reference = np.zeros(5)
    previous_control = np.zeros(2)

    arrays = [states, controls, reference, previous_control]
    snapshots = [array.copy() for array in arrays]

    compute_objective(
        states,
        controls,
        reference,
        previous_control,
        **make_weights(q=1, r=1, s=1, qf=1),
    )

    for array, snapshot in zip(arrays, snapshots):
        np.testing.assert_array_equal(array, snapshot)


def test_invalid_state_shape():
    """错误的数据形状应明确报错。"""
    with pytest.raises(ValueError):
        compute_objective(
            states=np.zeros((2, 4)),  # 错误：应有五个状态分量
            controls=np.zeros((1, 2)),
            reference=np.zeros(5),
            previous_control=np.zeros(2),
            **make_weights(),
        )