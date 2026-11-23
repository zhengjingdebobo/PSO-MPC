import numpy as np
import pytest

from pso_mpc_parking.config import EnvironmentConfig, SimulationConfig, VehicleConfig
from pso_mpc_parking.environment import CollisionChecker, ParkingEnvironment
from pso_mpc_parking.environment.geometry import vehicle_vertices


def test_default_scene_and_geometric_goal_fit():
    vehicle, scene = VehicleConfig(), ParkingEnvironment()
    checker = CollisionChecker(scene, vehicle)
    assert checker.check_state(SimulationConfig().initial_state).safe
    goal = scene.target_state(vehicle)
    assert checker.check_state(goal).safe
    vertices = vehicle_vertices(goal, vehicle)
    slot = scene.slot_vertices
    assert np.all(vertices.min(axis=0) >= slot.min(axis=0))
    assert np.all(vertices.max(axis=0) <= slot.max(axis=0))
    assert len(scene.obstacles) == 2


def test_custom_slot_center_and_guide_copy():
    scene = ParkingEnvironment(EnvironmentConfig(slot_center=(2, -1.25)))
    assert scene.target_state(VehicleConfig())[0] == pytest.approx(0.65)
    pose = np.array([1, 2, 0.3])
    scene.set_guide_pose(pose)
    pose[:] = 999
    copy = scene.guide_pose
    copy[:] = 888
    np.testing.assert_allclose(scene.guide_pose, [1, 2, 0.3])
    scene.set_guide_pose(None)
    assert scene.guide_pose is None


@pytest.mark.parametrize("factory", [
    lambda: VehicleConfig(length=1), lambda: VehicleConfig(max_steering=np.pi / 2),
    lambda: EnvironmentConfig(x_min=20), lambda: EnvironmentConfig(slot_center=(100, 0)),
    lambda: SimulationConfig(dt=0), lambda: SimulationConfig(initial_state=(1, 2)),
    lambda: ParkingEnvironment(EnvironmentConfig(slot_width=1)),
])
def test_invalid_configs(factory):
    with pytest.raises(ValueError):
        factory()
