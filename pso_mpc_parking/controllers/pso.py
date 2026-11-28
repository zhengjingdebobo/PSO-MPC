from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray,ArrayLike

from pso_mpc_parking.config.settings import PSOConfig
from pso_mpc_parking.controllers.evaluator import(
    EvaluationResult,
    TrajectoryEvaluator,
)
from pso_mpc_parking.models.vehicle_model import finite_vector

@dataclass(frozen=True)
class PSOResult:
    success: bool
    cost: float
    controls: NDArray[np.float64] | None
    states : NDArray[np.float64] | None
    cost_history: NDArray[np.float64]
    evaluations: int
    iterations: int

class PSOOptimizer:
    def __init__(
            self,
            evaluator: TrajectoryEvaluator,
            config: PSOConfig | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or PSOConfig()

        self.rng = np.random.default_rng(self.config.random_seed)

    def optimize(
            self,
            initial_state: ArrayLike,
            reference: ArrayLike,
            previous_control: ArrayLike,
            warm_start: ArrayLike | None = None,
    ) ->PSOResult:
        initial_state = finite_vector(initial_state, 5, "initial_state")
        reference = finite_vector(reference, 5, "reference")
        previous_control = finite_vector(previous_control, 2, "previous_control")

        config = self.config
        vehicle = self.evaluator.model.config

        horizon = config.horizon
        particle_count = config.particle_count
        dimension = horizon * 2

        initial_controls = None
        if warm_start is not None:
            initial_controls = np.asarray(warm_start, dtype=float)
            if initial_controls.shape != (horizon, 2):
                raise ValueError("warm_start must have shape (horizon, 2)")
            if not np.all(np.isfinite(initial_controls)):
                raise ValueError("warm_start must contain only finite values")
            initial_controls = np.stack([
                self.evaluator.model.constrain_control(control)
                for control in initial_controls
            ])

        # 单步变量为[v,w]，重复horizon次，设置上下限
        upper = np.tile(
            [
                vehicle.max_acceleration,
                vehicle.max_steering_rate,
            ],
            horizon,
        )
        lower = -upper   

        # PSO搜索速度
        velocity_limit = (config.velocity_fraction * (upper - lower))

        def sample_population():
            # 随机生成粒子位置和搜索速度
            positions = self.rng.uniform(
                lower,
                upper,
                size=(particle_count, dimension),
            )

            velocities = self.rng.uniform(
                -velocity_limit,
                velocity_limit,
                size=(particle_count, dimension),
            )

            return positions, velocities

        positions, velocities = sample_population()
        if initial_controls is not None:
            # Keep one particle at the shifted plan and seed nearby particles
            # locally, while retaining the remaining global random particles.
            warm_position = initial_controls.reshape(dimension)
            positions[0] = warm_position
            velocities[0] = 0.0
            local_count = (particle_count - 1) // 2
            if local_count:
                local_scale = 0.1 * (upper - lower)
                positions[1:1 + local_count] = np.clip(
                    warm_position
                    + self.rng.uniform(
                        -local_scale,
                        local_scale,
                        size=(local_count, dimension),
                    ),
                    lower,
                    upper,
                )
        
        personal_best_positions = positions.copy()
        personal_best_costs = np.full(
            particle_count,np.inf, dtype=float
        )

        global_best_position = None
        global_best_cost = float("inf")
        best_evaluation:EvaluationResult | None = None

        history = []
        evaluations = 0

        for iteration in range(config.max_iterations +1):
            for particle_index in range(particle_count):
                candidate = positions[particle_index].reshape(horizon, 2)

                evaluation = self.evaluator.evaluate(
                    initial_state = initial_state,
                    controls = candidate,
                    reference = reference,
                    previous_control = previous_control,
                )

                evaluations += 1

                # 限制最优解条件
                if not evaluation.feasible:
                    continue

                if not np.isfinite(evaluation.cost):
                    raise ValueError("可行候选解的成本必须为有限数值")

                # 更行Pbest
                if (evaluation.cost < personal_best_costs[particle_index]):
                    personal_best_costs[particle_index] = evaluation.cost
                    personal_best_positions[particle_index] = positions[particle_index].copy()

                # 更新Gbest
                if evaluation.cost < global_best_cost:
                    global_best_cost = evaluation.cost
                    global_best_position = positions[particle_index].copy()
                    best_evaluation = evaluation

            history.append(global_best_cost)

            # 最后一轮只评价，不生成新位置
            if iteration == config.max_iterations:
                break

            # 如果没有任何可行解，则重新随机采样，继续尝试
            if global_best_position is None:
                positions, velocities = sample_population()
                continue

            r1 = self.rng.random(size = (particle_count,dimension))
            r2 = self.rng.random(size = (particle_count,dimension))

            # 个体认知项
            cognitive_term = (
                config.cognitive_weight
                * r1
                * (personal_best_positions - positions)
            )

            no_personal_best = ~np.isfinite(personal_best_costs)
            cognitive_term[no_personal_best] = 0.0 # 使用的是 NumPy 的布尔索引

            # 社会认知项
            social_term = (
                config.social_weight
                * r2
                *(global_best_position - positions)
            )

            velocities = (
                config.inertia_weight * velocities
                + cognitive_term
                + social_term
            )

            velocities = np.clip(velocities,-velocity_limit,velocity_limit,)

            proposed_positions = positions +velocities
            positions = np.clip(proposed_positions,lower, upper)

            # 出边界速度清零
            outside = (
                (proposed_positions < lower) | (proposed_positions > upper)
            )
            velocities[outside] = 0.0

        if best_evaluation is None:
            return PSOResult(
                success = False,
                cost = float("inf"),
                controls = None,
                states = None,
                cost_history = np.asarray(history,dtype = float),
                evaluations = evaluations,
                iterations = config.max_iterations,
            )

        return PSOResult(
            success = True,
            cost = float(global_best_cost),
            controls = best_evaluation.controls.copy(),
            states = best_evaluation.states.copy(),
            cost_history = np.asarray(history, dtype=float),
            evaluations = evaluations,
            iterations = config.max_iterations,
        )

