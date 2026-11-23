import numpy as np
from numpy.typing import ArrayLike

# 计算代价函数

def compute_objective(
        state: ArrayLike, 
        controls: ArrayLike, 
        reference: ArrayLike,
        previous_state: ArrayLike,
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
    
    