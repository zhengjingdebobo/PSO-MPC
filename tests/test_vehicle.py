import numpy as np
import pytest

from pso_mpc_parking.config import VehicleConfig
from pso_mpc_parking.models import VehicleModel


def test_stationary():
    m = VehicleModel()
    s = np.array([1, 2, 0.4, 0, 0.3])
    states = m.predict_trajectory(s, np.zeros((50, 2)), 0.05)
    np.testing.assert_allclose(states, np.tile(s, (51, 1)), atol=1e-12)


@pytest.mark.parametrize("speed", [1.0, -1.0])
def test_straight(speed):
    m = VehicleModel()
    s = np.array([1, 2, 0.4, speed, 0])
    end = m.predict_trajectory(s, np.zeros((40, 2)), 0.05)[-1]
    np.testing.assert_allclose(end[:2], s[:2] + 2 * speed * np.array([np.cos(0.4), np.sin(0.4)]), atol=1e-12)
    assert end[2] == pytest.approx(0.4)


@pytest.mark.parametrize("speed", [1.0, -1.0])
@pytest.mark.parametrize("steer", [0.3, -0.3])
def test_turn_direction_and_rate(speed, steer):
    m = VehicleModel()
    s = [0, 0, 0, speed, steer]
    expected_rate = speed / m.config.wheelbase * np.tan(steer)
    assert m.derivative(s, [0, 0])[2] == pytest.approx(expected_rate)
    end = m.predict_step(s, [0, 0], 0.05)
    assert end[2] == pytest.approx(expected_rate * 0.05)
    assert np.sign(end[2]) == np.sign(speed * steer)


def test_stationary_steering():
    m = VehicleModel()
    states = m.predict_trajectory([1, 2, 0.3, 0, 0], np.tile([0, 0.2], (20, 1)), 0.05)
    np.testing.assert_allclose(states[:, :3], np.tile([1, 2, 0.3], (21, 1)), atol=1e-12)
    assert states[-1, 4] == pytest.approx(0.2)


@pytest.mark.parametrize("sign", [1, -1])
def test_limits(sign):
    m = VehicleModel()
    c = m.config
    np.testing.assert_allclose(m.constrain_control([sign * 100, sign * 100]),
                               [sign * c.max_acceleration, sign * c.max_steering_rate])
    states = m.predict_trajectory([0, 0, 0, 0, 0], np.tile([sign * 100, sign * 100], (100, 1)), 0.1)
    assert np.max(states[:, 3]) <= c.max_forward_speed
    assert np.min(states[:, 3]) >= -c.max_reverse_speed
    assert np.max(np.abs(states[:, 4])) <= c.max_steering
    assert states[-1, 3] == pytest.approx(c.max_forward_speed if sign > 0 else -c.max_reverse_speed)
    assert states[-1, 4] == pytest.approx(sign * c.max_steering)
    reset = m.reset([0, 0, 9, 100 * sign, 100 * sign])
    assert -np.pi <= reset[2] < np.pi
    assert abs(reset[4]) <= c.max_steering


def test_speed_limit_event_has_correct_distance():
    m = VehicleModel()
    end = m.predict_step([0, 0, 0, 1.9, 0], [1, 0], 1)
    # 0.1 s accelerating, 0.9 s at 2 m/s.
    assert end[0] == pytest.approx(1.9 * 0.1 + 0.5 * 0.1**2 + 2 * 0.9)
    assert end[3] == pytest.approx(2)


@pytest.mark.parametrize("speed", [0.8, -0.8])
def test_circle_analytic_and_euler_comparison(speed):
    m = VehicleModel()
    s = np.array([1.0, 2.0, 0.4, speed, 0.3])
    rate = speed / m.config.wheelbase * np.tan(s[4])
    duration, dt = 2.0, 0.005
    controls = np.zeros((round(duration / dt), 2))
    rk = m.predict_trajectory(s, controls, dt, "rk4")[-1]
    eu = m.predict_trajectory(s, controls, dt, "euler")[-1]
    theta = s[2] + rate * duration
    analytic = [s[0] + speed / rate * (np.sin(theta) - np.sin(s[2])),
                s[1] - speed / rate * (np.cos(theta) - np.cos(s[2])), theta]
    np.testing.assert_allclose(rk[:3], analytic, atol=1e-10)
    np.testing.assert_allclose(eu[:3], rk[:3], atol=5e-4)


def test_rk4_more_accurate_than_euler():
    m = VehicleModel()
    state = [0, 0, 0, 1, 0.4]
    rate = np.tan(0.4) / m.config.wheelbase
    exact = np.array([np.sin(rate) / rate, (1 - np.cos(rate)) / rate])
    rk = m.predict_step(state, [0, 0], 1, "rk4")
    eu = m.predict_step(state, [0, 0], 1, "euler")
    assert np.linalg.norm(rk[:2] - exact) < np.linalg.norm(eu[:2] - exact) / 1000


def test_no_side_effects_and_copy():
    m = VehicleModel(initial_state=[1, 2, 0, 0, 0])
    original = m.state
    state = np.array([0, 0, 0, -1, 0.2], dtype=float)
    controls = np.zeros((5, 2))
    saved = state.copy()
    predicted = m.predict_trajectory(state, controls, 0.05)
    predicted[:] = 99
    copy = m.state
    copy[:] = 99
    np.testing.assert_array_equal(m.state, original)
    np.testing.assert_array_equal(state, saved)
    np.testing.assert_array_equal(controls, np.zeros((5, 2)))
    assert m.predict_trajectory(state, np.empty((0, 2)), 0.05).shape == (1, 5)
    m.step([1, 0], 0.1)
    assert m.state[3] > original[3]


@pytest.mark.parametrize("bad", [[1, 2], [0, 0, np.nan, 0, 0]])
def test_invalid_states(bad):
    with pytest.raises(ValueError):
        VehicleModel(initial_state=bad)


@pytest.mark.parametrize("dt", [0, -1, np.nan])
def test_invalid_dt(dt):
    with pytest.raises(ValueError):
        VehicleModel().predict_step(np.zeros(5), [0, 0], dt)


def test_invalid_method_and_controls():
    m = VehicleModel()
    with pytest.raises(ValueError):
        m.predict_step(np.zeros(5), [0, 0], 0.1, "bad")
    for controls in ([0, 0], [[np.inf, 0]]):
        with pytest.raises(ValueError):
            m.predict_trajectory(np.zeros(5), controls, 0.1)
