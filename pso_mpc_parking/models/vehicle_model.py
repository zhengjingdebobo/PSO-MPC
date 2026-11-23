"""Five-state rear-axle bicycle model; all angles use radians.

x points along the road, y to its left. Positive heading and steering are
counterclockwise/left. Positive speed is forward, negative speed is reverse.
This module imports neither environment, simulation nor visualization.
"""
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pso_mpc_parking.config import VehicleConfig

FloatArray = NDArray[np.float64]
IntegrationMethod = Literal["rk4", "euler"]


def finite_vector(value: ArrayLike, size: int, name: str) -> FloatArray:
    array = np.array(value, dtype=float, copy=True)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape ({size},) and finite values")
    return array


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


class VehicleModel:
    def __init__(self, config: VehicleConfig | None = None,
                 initial_state: ArrayLike | None = None) -> None:
        self.config = config or VehicleConfig()
        self.reset(np.zeros(5) if initial_state is None else initial_state)

    @property
    def state(self) -> FloatArray:
        """Return a copy: callers cannot mutate the live state."""
        return self._state.copy()

    def constrain_state(self, state: ArrayLike) -> FloatArray:
        result = finite_vector(state, 5, "state")
        c = self.config
        result[2] = wrap_angle(result[2])
        result[3] = np.clip(result[3], -c.max_reverse_speed, c.max_forward_speed)
        result[4] = np.clip(result[4], -c.max_steering, c.max_steering)
        return result

    def constrain_control(self, control: ArrayLike) -> FloatArray:
        result = finite_vector(control, 2, "control")
        c = self.config
        return np.clip(result, [-c.max_acceleration, -c.max_steering_rate],
                       [c.max_acceleration, c.max_steering_rate])

    def reset(self, state: ArrayLike) -> FloatArray:
        self._state = self.constrain_state(state)
        return self.state

    def derivative(self, state: ArrayLike, control: ArrayLike) -> FloatArray:
        """Constrained continuous dynamics, with outward rates zero at bounds."""
        s = self.constrain_state(state)
        a, omega = self.constrain_control(control)
        c = self.config
        if (s[3] >= c.max_forward_speed and a > 0) or (s[3] <= -c.max_reverse_speed and a < 0):
            a = 0.0
        if (s[4] >= c.max_steering and omega > 0) or (s[4] <= -c.max_steering and omega < 0):
            omega = 0.0
        return np.array([s[3] * np.cos(s[2]), s[3] * np.sin(s[2]),
                         s[3] / c.wheelbase * np.tan(s[4]), a, omega])

    def predict_step(self, initial_state: ArrayLike, control: ArrayLike,
                     dt: float, method: IntegrationMethod = "rk4") -> FloatArray:
        """Pure prediction for one zero-order-held control interval.

        Split at exact speed/steering saturation times. Integrate the pose using
        RK4 (default) or Euler; v/delta evolve exactly linearly on each segment.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if method not in ("rk4", "euler"):
            raise ValueError("method must be rk4 or euler")
        s = self.constrain_state(initial_state)
        u = self.constrain_control(control)
        c = self.config
        times = [0.0, float(dt)]
        for idx, rate, lower, upper in (
            (3, u[0], -c.max_reverse_speed, c.max_forward_speed),
            (4, u[1], -c.max_steering, c.max_steering),
        ):
            if rate != 0:
                hit = ((upper if rate > 0 else lower) - s[idx]) / rate
                if 0 < hit < dt:
                    times.append(float(hit))
        times = sorted(set(times))
        for start, end in zip(times[:-1], times[1:]):
            h = end - start
            # Keep constant rates throughout this segment, even at its endpoint.
            rates = self.derivative(s, u)[3:]

            def rhs(z: FloatArray) -> FloatArray:
                return np.array([z[3] * np.cos(z[2]), z[3] * np.sin(z[2]),
                                 z[3] / c.wheelbase * np.tan(z[4]), *rates])

            k1 = rhs(s)
            if method == "euler":
                s = s + h * k1
            else:
                k2 = rhs(s + h * k1 / 2)
                k3 = rhs(s + h * k2 / 2)
                k4 = rhs(s + h * k3)
                s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            s = self.constrain_state(s)
        return s

    def predict_trajectory(self, initial_state: ArrayLike, controls: ArrayLike,
                           dt: float, method: IntegrationMethod = "rk4") -> FloatArray:
        """Return (N+1, 5), including the initial state, without live-state edits.

        controls has shape (N, 2). Empty horizon: pass np.empty((0, 2)).
        No file access or drawing occurs here; suitable for repeated optimizers.
        """
        u = np.asarray(controls, dtype=float)
        if u.ndim != 2 or u.shape[1] != 2 or not np.all(np.isfinite(u)):
            raise ValueError("controls must have shape (N, 2) and finite values")
        if not np.isfinite(dt) or dt <= 0 or method not in ("rk4", "euler"):
            raise ValueError("invalid dt or integration method")
        states = np.empty((len(u) + 1, 5))
        states[0] = self.constrain_state(initial_state)
        for i, control in enumerate(u):
            states[i + 1] = self.predict_step(states[i], control, dt, method)
        return states

    def step(self, control: ArrayLike, dt: float,
             method: IntegrationMethod = "rk4") -> FloatArray:
        """Advance the live state; prediction methods above remain pure."""
        self._state = self.predict_step(self._state, control, dt, method)
        return self.state
