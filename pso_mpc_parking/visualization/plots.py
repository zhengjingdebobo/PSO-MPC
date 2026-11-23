"""Read-only Matplotlib views; no integration, control or collision decisions."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import numpy as np
from numpy.typing import ArrayLike

from pso_mpc_parking.config import VehicleConfig
from pso_mpc_parking.environment.geometry import vehicle_vertices
from pso_mpc_parking.environment.parking_env import ParkingEnvironment
from pso_mpc_parking.simulation import SimulationResult


def draw_scene(ax: Axes, environment: ParkingEnvironment) -> None:
    c = environment.config
    ax.add_patch(Polygon(environment.boundary_vertices, facecolor="#edf1f4", edgecolor="#253442", lw=2))
    # The dashed curb/road-edge reference is interrupted at the parking opening.
    left, right = c.slot_center[0] - c.slot_length / 2, c.slot_center[0] + c.slot_length / 2
    ax.plot([c.x_min, left], [c.road_edge_y] * 2, "--", color="#768591", lw=1)
    ax.plot([right, c.x_max], [c.road_edge_y] * 2, "--", color="#768591", lw=1)
    ax.add_patch(Polygon(environment.slot_vertices, facecolor="#d8efdf", edgecolor="#269459", lw=2,
                         linestyle="--", label=f"Target slot ({c.slot_length:.1f} x {c.slot_width:.1f} m)"))
    for i, obstacle in enumerate(environment.obstacles):
        ax.add_patch(Polygon(obstacle.vertices, facecolor="#8794a3", edgecolor="#344252",
                             label="Static vehicles" if i == 0 else None))
    guide = environment.guide_pose
    if guide is not None:
        ax.plot(*guide[:2], marker="x", color="purple", label="External guide pose")
        ax.arrow(*guide[:2], np.cos(guide[2]), np.sin(guide[2]), color="purple", width=0.025)
    ax.set(xlim=(c.x_min - 0.5, c.x_max + 0.5), ylim=(c.y_min - 0.5, c.y_max + 0.5),
           xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)


def draw_vehicle(ax: Axes, state: ArrayLike, vehicle: VehicleConfig,
                 color: str = "#1679ab", alpha: float = 1.0) -> None:
    state = np.asarray(state)
    x, y, theta, _, delta = state
    ax.add_patch(Polygon(vehicle_vertices(state, vehicle), facecolor=color, edgecolor="#163947",
                         alpha=alpha, lw=1.5))
    ax.plot(x, y, "o", color="white", markeredgecolor="#163947", ms=4)
    ax.arrow(x, y, 1.4 * np.cos(theta), 1.4 * np.sin(theta), width=0.035,
             head_width=0.22, color="#f5b333", length_includes_head=True, alpha=alpha)
    # Draw both axles' wheels. Front wheels use heading + actual steering angle.
    forward = np.array([np.cos(theta), np.sin(theta)])
    lateral = np.array([-np.sin(theta), np.cos(theta)])
    for longitudinal, angle in ((0, theta), (vehicle.wheelbase, theta + delta)):
        direction = np.array([np.cos(angle), np.sin(angle)])
        for side in (-1, 1):
            center = state[:2] + longitudinal * forward + side * vehicle.width * 0.43 * lateral
            endpoints = np.stack((center - 0.28 * direction, center + 0.28 * direction))
            ax.plot(*endpoints.T, color="#152a38", lw=3, alpha=alpha)


def plot_environment(environment: ParkingEnvironment, vehicle: VehicleConfig,
                     result: SimulationResult, index: int = -1) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    draw_scene(ax, environment)
    i = index % len(result.states)
    ax.plot(result.states[:i + 1, 0], result.states[:i + 1, 1], color="#147eaf", lw=2,
            label="Rear-axle history")
    if i > 0:
        outline = vehicle_vertices(result.states[0], vehicle)
        ax.add_patch(Polygon(outline, fill=False, edgecolor="#1679ab", linestyle=":", label="Initial body"))
    draw_vehicle(ax, result.states[i], vehicle)
    state = result.states[i]
    ax.set_title(f"Stage 1: prescribed reverse turn | t={result.times[i]:.2f} s\n"
                 f"heading={np.rad2deg(state[2]):.1f} deg, v={state[3]:.2f} m/s, "
                 f"steering={np.rad2deg(state[4]):.1f} deg")
    ax.legend(loc="lower right", fontsize=8)
    return fig


def plot_state_curves(result: SimulationResult) -> Figure:
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True, constrained_layout=True)
    t, states = result.times, result.states
    axes[0, 0].plot(t, states[:, 0], label="x (rear axle)")
    axes[0, 0].plot(t, states[:, 1], label="y (rear axle)")
    axes[0, 0].set_ylabel("Position [m]")
    axes[0, 0].legend()
    curves = [(axes[0, 1], np.rad2deg(np.unwrap(states[:, 2])), "Heading [deg]"),
              (axes[1, 0], states[:, 3], "Speed [m/s]"),
              (axes[1, 1], np.rad2deg(states[:, 4]), "Steering [deg]")]
    for ax, values, label in curves:
        ax.plot(t, values)
        ax.set_ylabel(label)
    for column, label, scale in ((0, "a [m/s²]", 1), (1, "Steering rate [deg/s]", 180 / np.pi)):
        if len(result.controls):
            values = result.controls[:, column] * scale
            axes[2, column].step(t, np.r_[values, values[-1]], where="post")
        axes[2, column].set_ylabel(label)
    for ax in axes.flat:
        ax.set_xlabel("Time [s]")
        ax.grid(alpha=0.3)
    fig.suptitle("Prescribed input response (angles displayed in degrees)")
    return fig


def animate_simulation(environment: ParkingEnvironment, vehicle: VehicleConfig,
                       result: SimulationResult, path: str | Path | None = None,
                       stride: int = 2) -> FuncAnimation:
    if stride < 1:
        raise ValueError("stride must be positive")
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    indices = list(range(0, len(result.states), stride))
    if indices[-1] != len(result.states) - 1:
        indices.append(len(result.states) - 1)

    def update(index: int) -> None:
        ax.clear()
        draw_scene(ax, environment)
        ax.plot(result.states[:index + 1, 0], result.states[:index + 1, 1], color="#1679ab", lw=2)
        draw_vehicle(ax, result.states[index], vehicle)
        s = result.states[index]
        ax.set_title(f"Prescribed reverse turn | t={result.times[index]:.2f} s | "
                     f"heading={np.rad2deg(s[2]):.1f} deg | steering={np.rad2deg(s[4]):.1f} deg")

    dt = result.times[1] - result.times[0] if len(result.times) > 1 else 0.05
    animation = FuncAnimation(fig, update, frames=indices, interval=dt * stride * 1000, repeat=True)
    if path is not None:
        path = Path(path)
        if path.suffix.lower() != ".gif":
            raise ValueError("stage 1 animation export supports .gif (Pillow), without FFmpeg")
        animation.save(str(path), writer=PillowWriter(fps=1 / (dt * stride)), dpi=100)
    return animation
