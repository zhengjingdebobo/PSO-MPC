import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from pso_mpc_parking.config import SimulationConfig, VehicleConfig
from pso_mpc_parking.environment import CollisionChecker, ParkingEnvironment
from pso_mpc_parking.models import VehicleModel
from pso_mpc_parking.simulation import Simulator
from pso_mpc_parking.visualization import animate_simulation, plot_environment, plot_state_curves


def test_read_only_plots_and_gif_export(tmp_path):
    env, vehicle, config = ParkingEnvironment(), VehicleConfig(), SimulationConfig()
    result = Simulator(VehicleModel(initial_state=config.initial_state), CollisionChecker(env)).run(np.zeros((2, 2)))
    original = result.states.copy()
    fig = plot_environment(env, vehicle, result)
    assert fig.axes[0].get_aspect() == 1.0
    fig.savefig(tmp_path / "scene.png")
    curves = plot_state_curves(result)
    assert len(curves.axes) == 6
    curves.savefig(tmp_path / "curves.png")
    animation = animate_simulation(env, vehicle, result, tmp_path / "demo.gif", stride=1)
    with Image.open(tmp_path / "demo.gif") as gif:
        assert gif.n_frames == 3
    np.testing.assert_array_equal(result.states, original)
    assert animation is not None
    plt.close("all")
