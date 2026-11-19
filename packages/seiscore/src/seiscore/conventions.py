"""Coordinate and component conventions used by every package and app.

Frame
    x = east, y = north, z = up, all in metres. Depth is ``-z``. Sources that
    store depth or elevation differently (PSET ``gelev``, Pulse ``z``) convert
    at the source boundary, never later.

Raw 3C components
    ``"E"``, ``"N"``, ``"Z"`` with Z positive **up**. A down-positive vertical
    channel is negated when it is read.

Ray frame (``P``, ``SH``, ``SV``)
    P  = unit vector from source to receiver (direction of propagation), so a
         compressional first motion is positive on P.
    SH = unit(z_up x P): horizontal, perpendicular to the ray.
    SV = SH x P.

    This matches qc-keepers (``gathers.py`` before the framework). event3CQC
    pointed P from receiver to source, so its P panel had the opposite
    polarity.
"""

from __future__ import annotations

import numpy as np

RAW_COMPONENTS: tuple[str, str, str] = ("E", "N", "Z")
RAY_COMPONENTS: tuple[str, str, str] = ("P", "SH", "SV")

#: Names that sources may use for raw components, mapped to the canonical ones.
COMPONENT_ALIASES: dict[str, str] = {
    "e": "E",
    "east": "E",
    "n": "N",
    "north": "N",
    "z": "Z",
    "up": "Z",
    "vertical": "Z",
}


def canonical_component(name: str) -> str | None:
    """Canonical raw component name for ``name``, or None if not E/N/Z."""
    return COMPONENT_ALIASES.get(str(name).strip().lower())


def ray_frames(source_xyz: np.ndarray, receivers_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation matrices from (E, N, Z) to (P, SH, SV) for each receiver.

    Returns ``(frames, valid)``. ``frames`` has shape ``(n, 3, 3)`` where
    ``frames[i] @ [e, n, z] == [p, sh, sv]``. ``valid`` is False where the ray
    is undefined: non-finite coordinates, zero length, or a vertical ray
    (no horizontal offset, so SH has no direction). Invalid rows are zero.
    """
    src = np.asarray(source_xyz, dtype=float).reshape(3)
    rec = np.asarray(receivers_xyz, dtype=float).reshape(-1, 3)
    ray = rec - src
    length = np.linalg.norm(ray, axis=1)
    horizontal = np.hypot(ray[:, 0], ray[:, 1])
    valid = (
        np.all(np.isfinite(rec), axis=1)
        & np.all(np.isfinite(src))
        & (length > 0.0)
        & (horizontal > 1e-9 * np.maximum(length, 1.0))
    )
    frames = np.zeros((rec.shape[0], 3, 3))
    if not np.any(valid):
        return frames, valid
    p = ray[valid] / length[valid, None]
    sh = np.column_stack((-ray[valid, 1], ray[valid, 0], np.zeros(int(valid.sum()))))
    sh /= horizontal[valid, None]
    sv = np.cross(sh, p)
    frames[valid, 0, :] = p
    frames[valid, 1, :] = sh
    frames[valid, 2, :] = sv
    return frames, valid
