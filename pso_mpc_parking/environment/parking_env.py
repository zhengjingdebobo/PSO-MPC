from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from pso_mpc_parking.config import EnvironmentConfig, VehicleConfig
from pso_mpc_parking.models.vehicle_model import FloatArray, finite_vector
from .geometry import rectangle_vertices


@dataclass(frozen=True)
class Obstacle:
    name: str
    center: tuple[float, float]
    length: float
    width: float
    heading: float = 0.0

    @property
    def vertices(self) -> FloatArray:
        return rectangle_vertices(self.center, self.length, self.width, self.heading)


class ParkingEnvironment:
    def __init__(self, config: EnvironmentConfig | None = None,
                 vehicle_config: VehicleConfig | None = None) -> None:
        self.config = config or EnvironmentConfig()
        vehicle = vehicle_config or VehicleConfig()
        c = self.config
        if c.slot_length < vehicle.length or c.slot_width < vehicle.width:
            raise ValueError("slot cannot contain the configured vehicle")
        cx, cy = c.slot_center
        offset = c.slot_length / 2 + c.obstacle_gap + c.obstacle_length / 2
        self.obstacles = (
            Obstacle("rear_car", (cx - offset, cy), c.obstacle_length, c.obstacle_width),
            Obstacle("front_car", (cx + offset, cy), c.obstacle_length, c.obstacle_width),
        )
        for obstacle in self.obstacles:
            if not self.contains(obstacle.vertices):
                raise ValueError("obstacle must be inside drivable bounds")
        self._guide_pose: FloatArray | None = None

    @property
    def slot_vertices(self) -> FloatArray:
        c = self.config
        return rectangle_vertices(c.slot_center, c.slot_length, c.slot_width)

    @property
    def boundary_vertices(self) -> FloatArray:
        c = self.config
        return rectangle_vertices(((c.x_min + c.x_max) / 2, (c.y_min + c.y_max) / 2),
                                  c.x_max - c.x_min, c.y_max - c.y_min)

    def contains(self, vertices: ArrayLike) -> bool:
        p = np.asarray(vertices)
        c = self.config
        return bool(np.all((p[:, 0] >= c.x_min) & (p[:, 0] <= c.x_max)
                           & (p[:, 1] >= c.y_min) & (p[:, 1] <= c.y_max)))

    def target_state(self, vehicle: VehicleConfig) -> FloatArray:
        """Geometric reference only; not a generated path or control policy."""
        offset = (vehicle.wheelbase + vehicle.front_overhang - vehicle.rear_overhang) / 2
        cx, cy = self.config.slot_center
        return np.array([cx - offset, cy, 0.0, 0.0, 0.0])

    def set_guide_pose(self, pose: ArrayLike | None) -> None:
        """Reserved external guide input (rear-axle x, y, heading); no planning."""
        self._guide_pose = None if pose is None else finite_vector(pose, 3, "guide pose")

    @property
    def guide_pose(self) -> FloatArray | None:
        return None if self._guide_pose is None else self._guide_pose.copy()
