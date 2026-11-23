import numpy as np
from numpy.typing import ArrayLike

# 计算代价函数

def compute_objective(
        states: ArrayLike, 
        controls: ArrayLike, 
        reference: ArrayLike,
        previous_control: ArrayLike,
        *,
        Q: ArrayLike,
        R: ArrayLike,
        S: ArrayLike,
        Qf: ArrayLike,
    ) -> float:
    """
    参数：
        states: 预测状态序列，形状 (N+1, 5)，包含初始状态。
        controls: 控制序列，形状 (N, 2)。
        reference: 目标状态，形状 (5,)。
        previous_control: 上一周期实际执行的输入，形状 (2,)。
        Q、Qf: 状态误差权重矩阵，形状 (5, 5)。
        R、S: 输入及输入变化量权重矩阵，形状 (2, 2)。

    权重矩阵应保证为对称半正定矩阵。   
    仅计算代价。
    """

    # 数据预处理
    states = np.asarray(states, dtype=float)
    controls = np.asarray(controls, dtype=float)
    reference = np.asarray(reference, dtype=float)
    previous_control = np.asarray(previous_control, dtype=float)

    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    S = np.asarray(S, dtype=float)
    Qf = np.asarray(Qf, dtype=float)

    if controls.ndim != 2 or controls.shape[1] != 2: # 检查控制量形状
        raise ValueError("Controls必须为(N,2).")

    horizon = controls.shape[0]

    expected_shapes = [
        ("states", states, (horizon + 1, 5)),
        ("reference", reference, (5,)),
        ("previous_control", previous_control, (2,)),
        ("Q", Q, (5, 5)),
        ("R", R, (2, 2)),
        ("S", S, (2, 2)),
        ("Qf", Qf, (5, 5)),
    ]

    for name, array, shape in expected_shapes:
        if array.shape != shape:
            raise ValueError(f"{name}的形状应为{shape}, 但得到{array.shape}.")

    arrays = [states, controls, reference, previous_control, Q, R, S, Qf]
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("输入数组中包含非有限值")


    # 计算各项
    # e
    errors = states - reference  # 计算状态误差
    errors[:, 2] = (errors[:, 2] + np.pi) % (2 * np.pi) - np.pi # 将角度误差限制在[-π, π]范围内
    # ▲u
    delta_controls = np.diff(
        np.vstack([previous_control, controls]), axis=0,
    )

    # 计算代价函数
    total_cost = 0.0
    total_cost = 0.0

    for i in range(horizon):
        error = errors[i]
        control = controls[i]
        delta_control = delta_controls[i]

        total_cost += float(
            error @ Q @ error
            + control @ R @ control
            + delta_control @ S @ delta_control
        )

    # 终端状态代价
    terminal_error = errors[-1]
    total_cost += float(terminal_error @ Qf @ terminal_error)

    return total_cost

