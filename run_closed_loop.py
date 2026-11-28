from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from pso_mpc_parking.config import (
    EnvironmentConfig,
    SimulationConfig,
    VehicleConfig,
)
from pso_mpc_parking.config.settings import (
    ClosedLoopConfig,
    PSOConfig,
)
from pso_mpc_parking.controllers.evaluator import TrajectoryEvaluator
from pso_mpc_parking.controllers.mpc import MPCController
from pso_mpc_parking.controllers.pso import PSOOptimizer
from pso_mpc_parking.environment import CollisionChecker, ParkingEnvironment
from pso_mpc_parking.models import VehicleModel
from pso_mpc_parking.simulation.closed_loop import ClosedLoopSimulator
from pso_mpc_parking.visualization import (
    animate_simulation,
    plot_environment,
    plot_state_curves,
)


def main():
    output_dir = Path("outputs/closed_loop_trial4")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 配置、环境和车辆
    vehicle_config = VehicleConfig()
    environment_config = EnvironmentConfig()
    simulation_config = SimulationConfig()

    environment = ParkingEnvironment(
        environment_config,
        vehicle_config,
    )
    model = VehicleModel(
        vehicle_config,
        simulation_config.initial_state,
    )
    checker = CollisionChecker(environment, vehicle_config)

    # 代价权重：请按报告中的 Q、R、S、Qf 修改
    # Lower stage heading penalty permits temporary turning; terminal weights
    # strongly favor the final pose and a stopped vehicle.
    Q = np.diag([8.0, 8.0, 2.0, 0.5, 0.2])
    R = np.diag([0.2, 0.2])
    S = np.diag([0.4, 0.2])
    Qf = np.diag([80.0, 80.0, 180.0, 20.0, 5.0])

    evaluator = TrajectoryEvaluator(
        model=model,
        checker=checker,
        dt=simulation_config.dt,
        Q=Q,
        R=R,
        S=S,
        Qf=Qf,
    )

    # 先用较小规模验证入口；确认运行正常后再增加粒子数和迭代次数
    optimizer = PSOOptimizer(
        evaluator,
        PSOConfig(
            horizon=20,
            particle_count=8,
            max_iterations=8,
            random_seed=0,
        ),
    )
    controller = MPCController(optimizer)

    closed_loop = ClosedLoopSimulator(
        controller,
        ClosedLoopConfig(max_steps=300),
    )

    initial_state = np.asarray(simulation_config.initial_state, dtype=float)
    target_state = environment.target_state(vehicle_config)

    result = closed_loop.run(
        initial_state=initial_state,
        reference=target_state,
    )

    # 保存图像和动画
    plot_environment(
        environment, vehicle_config, result.simulation
    ).savefig(output_dir / "trajectory.png", dpi=160)

    plot_state_curves(
        result.simulation
    ).savefig(output_dir / "state_curves.png", dpi=160)

    animation = animate_simulation(
        environment,
        vehicle_config,
        result.simulation,
        output_dir / "closed_loop.gif",
    )

    # 保存逐时刻状态和实际执行的控制量
    np.savez(
        output_dir / "closed_loop_data.npz",
        times=result.simulation.times,
        states=result.simulation.states,
        controls=result.simulation.controls,
        solve_times=result.solve_times,
        optimization_costs=result.optimization_costs,
    )

    summary = {
        "success": result.success,
        "reason": result.reason,
        "steps": len(result.simulation.controls),
        "duration_seconds": float(result.simulation.times[-1]),
        "initial_state": result.simulation.states[0].tolist(),
        "final_state": result.simulation.states[-1].tolist(),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"结果目录：{output_dir.resolve()}")

    # 保留动画对象，避免程序退出前被回收
    _ = animation
    plt.close("all")


if __name__ == "__main__":
    main()
