from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pso_mpc_parking.models import VehicleModel
from pso_mpc_parking.models.vehicle_model import finite_vector
from pso_mpc_parking.environment import CollisionChecker
from pso_mpc_parking.environment.collision import TrajectoryCollisionResult
from pso_mpc_parking.controllers.objective import compute_objective

@dataclass(frozen=True)
class EvaluationResult:
    # 一次候选序列评价的结果。

    feasible: bool
    cost: float
    reason: str
    states: NDArray[np.float64]
    controls: NDArray[np.float64]
    collision_report: TrajectoryCollisionResult

class TrajectoryEvaluator:
    def __init__(
            self,
            model: VehicleModel,
            checker: CollisionChecker,
            dt: float,
            *,
            Q: ArrayLike,
            R: ArrayLike,
            S: ArrayLike,
            Qf: ArrayLike,
    ) -> None:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt 必须是有限正数")

         # 防止运动预测和碰撞检测使用不同的车辆
        if model.config != checker.vehicle_config:
            raise ValueError("模型与碰撞检测器必须使用相同的车辆配置")

        self.model = model
        self.checker = checker
        self.dt = float(dt)

        # 权重只在创建评价器时检查一次
        self._weights = {}

        for name, value, shape in [
            ("Q",Q,(5,5)),
            ("R",R,(2,2)),
            ("S",S,(2,2)),
            ("Qf",Qf,(5,5)),
        ]:
            matrix = np.array(value, dtype=float ,copy=True)

            
            if matrix.shape != shape:
                raise ValueError(f"{name} 的形状必须为 {shape}")

            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"{name} 必须全部为有限数值")

            if not np.allclose(matrix, matrix.T):
                raise ValueError(f"{name} 必须为对称矩阵")

            if np.linalg.eigvalsh(matrix).min() < -1e-10:
                raise ValueError(f"{name} 必须为半正定矩阵")

            self._weights[name] = matrix  

    def evaluate(
            self,
            initial_state: ArrayLike,
            controls: ArrayLike,
            reference: ArrayLike,
            previous_control: ArrayLike,
    ) -> EvaluationResult:
        # 评价一组候选控制序列，不推进真实车辆

        initial_state = finite_vector(initial_state, 5,"initial_state")
        reference = finite_vector(reference, 5, "reference")
        previous_control = finite_vector(
            previous_control, 2, "previous_control"
        )

        controls = np.array(controls, dtype=float, copy=True)

        if (
            controls.ndim != 2
            or controls.shape[1] != 2
            or controls.shape[0] == 0
        ):
            raise ValueError("controls 必须为非空的 (N, 2) 数组")

        if not np.all(np.isfinite(controls)):
            raise ValueError("controls 必须全部为有限数值")

        # 限幅后的输入用于预测、成本计算以及后续执行
        applied_controls = np.stack([
            self.model.constrain_control(control) 
            for control in controls
        ])

        # 预测车辆状态
        states = self.model.predict_trajectory(
            initial_state, applied_controls, self.dt,method="rk4",
        )

        # 使用已有的轨迹插值碰撞检查
        report = self.checker.check_trajectory(states)

        # 碰撞或者越界
        if report.collision:
            feasible = False
            cost = float("inf")
            reason = "collision_or_boundary"

        # 在安全距离外
        elif report.safety_violation:
            feasible = False
            cost = float("inf")
            reason = "safety_distance"

        # 安全
        else:
            feasible = True
            reason = "ok"

            cost = compute_objective(
                states=states,
                controls=applied_controls,
                reference=reference,
                previous_control=previous_control,
                **self._weights,
            )

            if not np.isfinite(cost):
                raise ValueError("代价计算结果不是有限数值")

        return EvaluationResult(
            feasible=feasible,
            cost=cost,
            reason=reason,
            states=states,
            controls=applied_controls,
            collision_report=report,
        )