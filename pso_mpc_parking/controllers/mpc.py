from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike,NDArray

from pso_mpc_parking.controllers.pso import PSOOptimizer,PSOResult
from pso_mpc_parking.models.vehicle_model import finite_vector

@dataclass(frozen=True)
class MPCResult:
    success: bool
    control: NDArray[np.float64] | None
    reason: str
    optimization: PSOResult

class MPCController:
    def __init__(
            self,
            optimizer: PSOOptimizer,
            previous_control: ArrayLike | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.reset(previous_control)

    @property
    def dt(self) ->float:
        return self.optimizer.evaluator.dt

    @property
    def previous_control(self) ->NDArray[np.float64]:
        return self._previous_control.copy()

    def reset(
        # 充值控制器保存的上一周期输入
            self,
            previous_control: ArrayLike | None = None,
    ) -> None:
        control = (
            np.zeros(2)
            if previous_control is None
            else previous_control
        )

        self.record_applied_control(control)

    def record_applied_control(self, control: ArrayLike) -> None:
        # 记录实际下发的输入
        model = self.optimizer.evaluator.model
        self._previous_control = model.constrain_control(control)

    def compute_control(
            self,
            current_state: ArrayLike,
            reference: ArrayLike
    ) -> MPCResult:
        # 根据当前状态优化控制，且返回第一个输入
        current_state = finite_vector(
            current_state, 5, "current_state"
        )
        reference = finite_vector(reference, 5,"reference")

        optimization = self.optimizer.optimize(
            initial_state= current_state,
            reference= reference,
            previous_control= self.previous_control,
        )

        # 检测是否成功
        if not optimization.success:
            return MPCResult(
                success=False,
                control=None,
                reason="no_feasible_sequence",
                optimization=optimization,
            )

         # 检查优化器返回结果是否完整。
        if (
            optimization.controls is None
            or optimization.states is None
        ):
            raise ValueError("PSO 报告成功，但缺少控制序列或预测轨迹")

        horizon = self.optimizer.config.horizon

        controls = np.asarray(
            optimization.controls, dtype=float
        )
        states = np.asarray(
            optimization.states, dtype=float
        )

        if controls.shape != (horizon, 2):
            raise ValueError("PSO 返回的控制序列形状不正确")

        if states.shape != (horizon + 1, 5):
            raise ValueError("PSO 返回的预测轨迹形状不正确")

        if (
            not np.all(np.isfinite(controls))
            or not np.all(np.isfinite(states))
            or not np.isfinite(optimization.cost)
        ):
            raise ValueError("PSO 成功结果中存在非有限数值")

        control = controls[0].copy()

        model = self.optimizer.evaluator.model
        constrained = model.constrain_control(control)

        if not np.array_equal(control, constrained):
             raise ValueError("PSO 返回的首个输入超出范围")

        return MPCResult(
            success= True,
            control= control,
            reason= "ok",
            optimization= optimization,
        )
