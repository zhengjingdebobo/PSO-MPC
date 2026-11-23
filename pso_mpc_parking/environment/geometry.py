"""Pure rectangle geometry; no plotting or simulation dependencies."""
import numpy as np
from numpy.typing import ArrayLike

from pso_mpc_parking.config import VehicleConfig
from pso_mpc_parking.models.vehicle_model import FloatArray, finite_vector


def rectangle_vertices(center: ArrayLike, length: float, width: float,
                       heading: float = 0.0) -> FloatArray:
    center = finite_vector(center, 2, "center")
    if not np.all(np.isfinite([length, width, heading])) or min(length, width) <= 0:
        raise ValueError("rectangle dimensions must be positive and finite")
    local = np.array([[length / 2, width / 2], [-length / 2, width / 2],
                      [-length / 2, -width / 2], [length / 2, -width / 2]])
    c, s = np.cos(heading), np.sin(heading)
    return local @ np.array([[c, s], [-s, c]]) + center


def vehicle_vertices(state: ArrayLike, config: VehicleConfig) -> FloatArray:
    """Four CCW body corners, correctly offset from the rear axle."""
    state = finite_vector(state, 5, "state")
    offset = (config.wheelbase + config.front_overhang - config.rear_overhang) / 2
    center = state[:2] + offset * np.array([np.cos(state[2]), np.sin(state[2])])
    return rectangle_vertices(center, config.length, config.width, state[2])


def validate_polygon(polygon: ArrayLike) -> FloatArray:
    p = np.asarray(polygon, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.all(np.isfinite(p)):
        raise ValueError("polygon must be finite with shape (N>=3, 2)")
    edges = np.roll(p, -1, axis=0) - p
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if np.any(np.linalg.norm(edges, axis=1) < 1e-12) or not (np.all(cross > 1e-12) or np.all(cross < -1e-12)):
        raise ValueError("polygon must be strictly convex and ordered")
    return p


def polygons_intersect(first: ArrayLike, second: ArrayLike, tolerance: float = 1e-9) -> bool:
    """SAT on convex polygons. Edge/corner contact counts as collision."""
    a, b = validate_polygon(first), validate_polygon(second)
    for polygon in (a, b):
        edges = np.roll(polygon, -1, axis=0) - polygon
        axes = np.column_stack((-edges[:, 1], edges[:, 0]))
        axes /= np.linalg.norm(axes, axis=1)[:, None]
        for axis in axes:
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() - tolerance or pb.max() < pa.min() - tolerance:
                return False
    return True


def polygon_distance(first: ArrayLike, second: ArrayLike) -> float:
    """Euclidean clearance, zero for touching or overlapping convex bodies."""
    a, b = validate_polygon(first), validate_polygon(second)
    if polygons_intersect(a, b):
        return 0.0
    minimum = float("inf")
    for points, edges in ((a, b), (b, a)):
        for start, end in zip(edges, np.roll(edges, -1, axis=0)):
            vector = end - start
            t = np.clip((points - start) @ vector / (vector @ vector), 0, 1)
            minimum = min(minimum, float(np.linalg.norm(points - (start + t[:, None] * vector), axis=1).min()))
    return minimum
