from dataclasses import dataclass
from math import isclose, isfinite, pi


@dataclass(frozen=True)
class VehicleConfig:
    length: float = 4.5
    width: float = 1.8
    wheelbase: float = 2.7
    front_overhang: float = 0.9
    rear_overhang: float = 0.9
    max_steering: float = 35.0 * pi / 180.0
    max_steering_rate: float = 30.0 * pi / 180.0
    max_forward_speed: float = 2.0
    max_reverse_speed: float = 1.5
    max_acceleration: float = 1.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isclose(self.length, self.wheelbase + self.front_overhang + self.rear_overhang):
            raise ValueError("length must equal wheelbase + front/rear overhangs")
        if self.max_steering >= pi / 2:
            raise ValueError("max_steering must be below pi/2")


@dataclass(frozen=True)
class EnvironmentConfig:
    # A rectangular drivable domain includes the roadside parking strip.
    x_min: float = -12.0
    x_max: float = 18.0
    y_min: float = -3.2
    y_max: float = 6.0
    road_edge_y: float = 0.0
    slot_center: tuple[float, float] = (0.0, -1.25)
    slot_length: float = 7.0
    slot_width: float = 2.5
    obstacle_length: float = 4.5
    obstacle_width: float = 1.8
    obstacle_gap: float = 0.1
    safety_distance: float = 0.2
    # Maximum corner-motion estimate per interpolation interval, in metres.
    collision_resolution: float = 0.05

    def __post_init__(self) -> None:
        values = [v for k, v in vars(self).items() if k != "slot_center"]
        if not all(isfinite(v) for v in [*values, *self.slot_center]):
            raise ValueError("environment values must be finite")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("invalid drivable bounds")
        if min(self.slot_length, self.slot_width, self.obstacle_length,
               self.obstacle_width, self.collision_resolution) <= 0:
            raise ValueError("dimensions and collision resolution must be positive")
        if min(self.safety_distance, self.obstacle_gap) < 0:
            raise ValueError("clearances cannot be negative")
        cx, cy = self.slot_center
        if not (self.x_min <= cx - self.slot_length / 2
                and cx + self.slot_length / 2 <= self.x_max
                and self.y_min <= cy - self.slot_width / 2
                and cy + self.slot_width / 2 <= self.y_max):
            raise ValueError("parking slot must be inside drivable bounds")


@dataclass(frozen=True)
# 相关配置设置
class SimulationConfig:
    dt: float = 0.05 # 每步0.05秒
    duration: float = 6.0   # 总共6秒
    initial_state: tuple[float, float, float, float, float] = (
        6.0,     # 后轴中心 x，m
        2.0,     # 后轴中心 y，m
        0.0,     # 航向角 theta，rad
        -0.8,     # 速度 v，m/s：负值表示倒车
        20.0 * pi / 180.0  # 前轮转角 delta：20°
    )
    integration_method: str = "rk4"

    def __post_init__(self) -> None:
        if not isfinite(self.dt) or not isfinite(self.duration) or min(self.dt, self.duration) <= 0:
            raise ValueError("dt and duration must be finite and positive")
        if len(self.initial_state) != 5 or not all(isfinite(v) for v in self.initial_state):
            raise ValueError("initial_state must contain five finite values")
        if self.integration_method not in ("rk4", "euler"):
            raise ValueError("integration_method must be rk4 or euler")
