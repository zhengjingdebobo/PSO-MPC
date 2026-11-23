"""Independent geometry queries; physical collisions and safety are distinct."""
from dataclasses import dataclass
from math import ceil

import numpy as np
from numpy.typing import ArrayLike

from pso_mpc_parking.config import VehicleConfig
from pso_mpc_parking.models.vehicle_model import wrap_angle
from .geometry import polygon_distance, polygons_intersect, vehicle_vertices
from .parking_env import ParkingEnvironment


@dataclass(frozen=True)
class CollisionResult:
    obstacle_names: tuple[str, ...]
    boundary_collision: bool
    minimum_clearance: float
    safety_violation: bool

    @property
    def collision(self) -> bool:
        return bool(self.obstacle_names) or self.boundary_collision

    @property
    def safe(self) -> bool:
        return not self.collision and not self.safety_violation


@dataclass(frozen=True)
class TrajectoryCollisionResult:
    collision: bool
    safety_violation: bool
    minimum_clearance: float
    checked_samples: int
    first_collision_interval: int | None
    first_collision_fraction: float | None


class CollisionChecker:
    def __init__(self, environment: ParkingEnvironment,
                 vehicle_config: VehicleConfig | None = None) -> None:
        self.environment = environment
        self.vehicle_config = vehicle_config or VehicleConfig()

    def check_state(self, state: ArrayLike,
                    safety_distance: float | None = None) -> CollisionResult:
        c = self.environment.config
        margin = c.safety_distance if safety_distance is None else safety_distance
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("safety_distance must be finite and nonnegative")
        vertices = vehicle_vertices(state, self.vehicle_config)
        # Contact with the enclosing boundary also counts as a collision.
        signed_boundary = float(min((vertices[:, 0] - c.x_min).min(),
                                    (c.x_max - vertices[:, 0]).min(),
                                    (vertices[:, 1] - c.y_min).min(),
                                    (c.y_max - vertices[:, 1]).min()))
        boundary_collision = signed_boundary <= 1e-9
        clearance = max(0.0, signed_boundary)
        contacts = []
        for obstacle in self.environment.obstacles:
            if polygons_intersect(vertices, obstacle.vertices):
                contacts.append(obstacle.name)
            clearance = min(clearance, polygon_distance(vertices, obstacle.vertices))
        return CollisionResult(tuple(contacts), boundary_collision, clearance,
                               bool(clearance < margin - 1e-9))

    def check_trajectory(self, states: ArrayLike, safety_distance: float | None = None,
                         resolution: float | None = None) -> TrajectoryCollisionResult:
        """Sample endpoints and interpolated poses, including shortest-angle turns.

        resolution bounds estimated corner travel between samples. This is a
        discrete check, not exact continuous swept-volume certification. Use fine
        model steps for highly curved motion; adjacent headings must differ < pi.
        Index 0 denotes the initial state or first interval, fraction in [0, 1].
        """
        trajectory = np.asarray(states, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] != 5 or not len(trajectory) or not np.all(np.isfinite(trajectory)):
            raise ValueError("states must have shape (N>=1, 5) and finite values")
        spacing = self.environment.config.collision_resolution if resolution is None else resolution
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("resolution must be finite and positive")
        c = self.vehicle_config
        radius = np.hypot(max(c.rear_overhang, c.wheelbase + c.front_overhang), c.width / 2)
        checked = 0
        any_collision = any_safety = False
        minimum = float("inf")
        first_interval = first_fraction = None

        def inspect(state: ArrayLike, interval: int, fraction: float) -> None:
            nonlocal checked, any_collision, any_safety, minimum, first_interval, first_fraction
            result = self.check_state(state, safety_distance)
            checked += 1
            minimum = min(minimum, result.minimum_clearance)
            any_safety |= result.safety_violation
            if result.collision:
                any_collision = True
                if first_interval is None:
                    first_interval, first_fraction = interval, fraction

        inspect(trajectory[0], 0, 0.0)
        for i, (start, end) in enumerate(zip(trajectory[:-1], trajectory[1:])):
            difference = end - start
            difference[2] = wrap_angle(difference[2])
            travel = np.linalg.norm(difference[:2]) + radius * abs(difference[2])
            count = max(1, ceil(travel / spacing))
            for j in range(1, count + 1):
                inspect(start + (j / count) * difference, i, j / count)
        return TrajectoryCollisionResult(any_collision, any_safety, minimum, checked,
                                         first_interval, first_fraction)
