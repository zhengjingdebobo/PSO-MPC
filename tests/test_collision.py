import numpy as np
import pytest

from pso_mpc_parking.config import EnvironmentConfig, VehicleConfig
from pso_mpc_parking.environment import CollisionChecker, ParkingEnvironment
from pso_mpc_parking.environment.geometry import (
    polygon_distance, polygons_intersect, rectangle_vertices, vehicle_vertices,
)
from pso_mpc_parking.environment.parking_env import Obstacle


@pytest.mark.parametrize("offset,expected", [(3, False), (2, True), (1.9, True), (0, True)])
def test_sat_separation_contact_overlap(offset, expected):
    a = rectangle_vertices([0, 0], 2, 2)
    b = rectangle_vertices([offset, 0], 2, 2)
    assert polygons_intersect(a, b) == expected
    assert polygons_intersect(b, a) == expected


def test_rotated_sat_and_aabb_false_positive():
    a = rectangle_vertices([0, 0], 4, 0.3, np.pi / 4)
    crossing = rectangle_vertices([0, 0], 4, 0.3, -np.pi / 4)
    parallel = rectangle_vertices([-0.5, 0.5], 4, 0.3, np.pi / 4)
    assert polygons_intersect(a, crossing)
    assert not polygons_intersect(a, parallel)  # Bounding boxes overlap.
    assert polygon_distance(a, parallel) > 0


def test_vehicle_rear_axle_geometry():
    c = VehicleConfig()
    vertices = vehicle_vertices(np.zeros(5), c)
    assert vertices[:, 0].min() == pytest.approx(-c.rear_overhang)
    assert vertices[:, 0].max() == pytest.approx(c.wheelbase + c.front_overhang)
    assert np.ptp(vertices[:, 1]) == pytest.approx(c.width)
    rotated = vehicle_vertices([2, 3, np.pi / 2, 0, 0], c)
    assert rotated[:, 1].min() == pytest.approx(3 - c.rear_overhang)
    assert rotated[:, 1].max() == pytest.approx(3 + c.wheelbase + c.front_overhang)
    assert np.ptp(rotated[:, 0]) == pytest.approx(c.width)


def test_corner_distance_is_euclidean():
    assert polygon_distance(rectangle_vertices([0, 0], 2, 2),
                            rectangle_vertices([3, 3], 2, 2)) == pytest.approx(np.sqrt(2))


def make_checker():
    env = ParkingEnvironment()
    env.obstacles = (Obstacle("box", (0, 0), 2, 2),)
    return CollisionChecker(env)


def test_safety_is_separate_from_collision():
    checker = make_checker()
    # Rear edge at x=1.1: 0.1 m separation from box's right edge x=1.
    result = checker.check_state([2, 0, 0, 0, 0], safety_distance=0.2)
    assert not result.collision
    assert result.safety_violation
    assert result.minimum_clearance == pytest.approx(0.1)
    assert not checker.check_state([2, 0, 0, 0, 0], safety_distance=0.1).safety_violation
    touching = checker.check_state([1.9, 0, 0, 0, 0], safety_distance=0)
    assert touching.collision
    assert not touching.safety_violation  # No safety threshold, still physical collision.
    assert not touching.safe


def test_rotated_boundary_and_contact():
    checker = CollisionChecker(ParkingEnvironment())
    # Rear axle remains inside but rotated front extends above y_max=6.
    assert checker.check_state([0, 3, np.pi / 2, 0, 0]).boundary_collision
    assert checker.check_state([0, 2.4, np.pi / 2, 0, 0]).boundary_collision
    assert not checker.check_state([0, 2.3, np.pi / 2, 0, 0]).boundary_collision


def test_interpolation_detects_endpoint_free_crossing():
    checker = make_checker()
    start, end = [-6, 0, 0, 0, 0], [5, 0, 0, 0, 0]
    assert not checker.check_state(start).collision
    assert not checker.check_state(end).collision
    result = checker.check_trajectory([start, end])
    assert result.collision
    assert result.checked_samples > 2
    assert result.first_collision_interval == 0
    assert 0 < result.first_collision_fraction < 1


def test_rotation_interpolation_detects_boundary_crossing():
    env = ParkingEnvironment()
    env.obstacles = ()
    checker = CollisionChecker(env)
    # Both horizontal bodies fit. Intermediate pi/2 pose exceeds the top edge.
    start, end = [0, 3, 0, 0, 0], [0, 3, np.pi - 0.01, 0, 0]
    assert not checker.check_state(start).collision
    assert not checker.check_state(end).collision
    assert checker.check_trajectory([start, end]).collision


def test_wrap_interpolation_uses_shortest_angle():
    env = ParkingEnvironment()
    env.obstacles = ()
    result = CollisionChecker(env).check_trajectory(
        [[0, 3, np.pi - 0.01, 0, 0], [0, 3, -np.pi + 0.01, 0, 0]])
    assert not result.collision
    assert result.checked_samples < 5


def test_invalid_collision_inputs():
    checker = make_checker()
    with pytest.raises(ValueError):
        checker.check_state(np.zeros(5), -1)
    with pytest.raises(ValueError):
        checker.check_trajectory([], resolution=0.1)
    with pytest.raises(ValueError):
        checker.check_trajectory([np.zeros(5)], resolution=0)
    with pytest.raises(ValueError):
        polygons_intersect([[0, 0], [0, 0], [1, 1]], rectangle_vertices([0, 0], 2, 2))
