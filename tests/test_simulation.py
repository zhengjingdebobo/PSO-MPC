import numpy as np

from pso_mpc_parking.config import SimulationConfig
from pso_mpc_parking.environment import CollisionChecker, ParkingEnvironment
from pso_mpc_parking.environment.parking_env import Obstacle
from pso_mpc_parking.models import VehicleModel
from pso_mpc_parking.simulation import Simulator


def test_demo_is_safe_and_records_correct_shapes():
    config = SimulationConfig()
    m = VehicleModel(initial_state=config.initial_state)
    result = Simulator(m, CollisionChecker(ParkingEnvironment()), config).run(np.zeros((120, 2)))
    assert result.states.shape == (121, 5)
    assert result.controls.shape == (120, 2)
    assert len(result.segment_checks) == 120
    assert result.times[-1] == 6.0
    assert result.states[-1, 2] < 0  # Reverse with positive steering.
    assert not result.stopped_on_collision
    assert all(c.safe for c in result.collisions)
    assert all(not c.collision and not c.safety_violation for c in result.segment_checks)
    np.testing.assert_array_equal(m.state, result.states[-1])


def test_stop_on_segment_collision_and_empty_run():
    env = ParkingEnvironment()
    env.obstacles = (Obstacle("small", (0, 2), 0.2, 1),)
    config = SimulationConfig(dt=8)
    sim = Simulator(VehicleModel(), CollisionChecker(env), config)
    result = sim.run(np.zeros((3, 2)), [-5, 2, 0, 1, 0])
    assert result.stopped_on_collision
    assert len(result.controls) == 1
    assert result.segment_checks[0].collision
    assert not result.collisions[-1].collision  # Crossed completely within a long step.
    empty = sim.run(np.empty((0, 2)), [-5, 2, 0, 0, 0])
    assert empty.states.shape == (1, 5)
    assert empty.controls.shape == (0, 2)


def test_initial_collision_stops_before_motion():
    env = ParkingEnvironment()
    goal = env.obstacles[0].center
    sim = Simulator(VehicleModel(), CollisionChecker(env))
    result = sim.run(np.zeros((5, 2)), [*goal, 0, 0, 0])
    assert result.stopped_on_collision
    assert result.controls.shape == (0, 2)
