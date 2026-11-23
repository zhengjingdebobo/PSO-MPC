"""Run only the prescribed-input stage-1 reverse-turn demonstration."""
import argparse
import json
from pathlib import Path

import numpy as np

from pso_mpc_parking.config import EnvironmentConfig, SimulationConfig, VehicleConfig
from pso_mpc_parking.environment import CollisionChecker, ParkingEnvironment
from pso_mpc_parking.models import VehicleModel
from pso_mpc_parking.simulation import Simulator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--show", action="store_true", help="Open Matplotlib windows")
    parser.add_argument("--method", choices=("rk4", "euler"), default="rk4")
    args = parser.parse_args()
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pso_mpc_parking.visualization import animate_simulation, plot_environment, plot_state_curves

    # 1. 读取车辆、场景和仿真配置
    vehicle, scene = VehicleConfig(), EnvironmentConfig()
    config = SimulationConfig(integration_method=args.method)

    # 2. 创建车辆及环境
    environment = ParkingEnvironment(scene, vehicle)
    model = VehicleModel(vehicle, config.initial_state)
    checker = CollisionChecker(environment, vehicle)
    if not checker.check_state(model.state).safe:
        raise RuntimeError("demo initial state must be collision-free with safety clearance")
    count = round(config.duration / config.dt)
    # Fixed a=0, omega=0: negative initial speed and positive initial steering.
    # This is a model demonstration, not a parking algorithm.

    # 3. 设置输入序列
    controls = np.zeros((count, 2)) # 控制输入序列 [a, w]：a: 加速度  w：前轮转角速度

    # 4. 执行仿真
    result = Simulator(model, checker, config).run(controls) # 输入控制

    args.output.mkdir(parents=True, exist_ok=True)
    plot_environment(environment, vehicle, result).savefig(args.output / "parking_environment.png", dpi=160)
    plot_state_curves(result).savefig(args.output / "state_curves.png", dpi=160)
    animation = None
    if not args.no_animation:
        animation = animate_simulation(environment, vehicle, result, args.output / "reverse_turn.gif")
    np.savez(args.output / "simulation_data.npz", times=result.times, states=result.states, controls=result.controls)
    summary = {
        "stage": 1, "integration": args.method, "steps": len(result.controls),
        "duration_seconds": float(result.times[-1]),
        "initial_state": result.states[0].tolist(), "final_state": result.states[-1].tolist(),
        "physical_collision": any(r.collision for r in result.collisions) or any(r.collision for r in result.segment_checks),
        "safety_violation": any(r.safety_violation for r in result.collisions) or any(r.safety_violation for r in result.segment_checks),
        "minimum_clearance_m": min([r.minimum_clearance for r in result.collisions]
                                   + [r.minimum_clearance for r in result.segment_checks]),
        "stopped_on_collision": result.stopped_on_collision,
    }
    (args.output / "simulation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Artifacts: {args.output.resolve()}")
    if args.show:
        plt.show()
    else:
        plt.close("all")
    # Keep an animation reference alive until the UI closes.
    _ = animation


if __name__ == "__main__":
    main()
