"""Photoelectric-event dataset from events.h5 for (x,y,z) reconstruction.

Axis convention (set by Julia's column-major writes): maps arrive as (n, 8, 8)
with axis 1 = y (SiPM row j) and axis 2 = x (SiPM column i); pixel centres at
(i + 0.5) * 6 mm. xyz1 arrives as (n, 3).

D4 augmentation: the crystal + grid have the full symmetry of the square, so the
eight flips/rotations of the light map with the matching (x, y) transform are
exact new events. Run this file directly for a numerical self-test.
"""
from pathlib import Path

import h5py
import numpy as np

CONTAINED_KEV = 510.5
PITCH = 6.0


E0_KEV = 511.0


def _select(path, sel):
    with h5py.File(path, "r") as f:
        maps = f["maps"][()][sel].astype(np.float32)
        npe = f["npe"][()][sel].astype(np.float32)
        xyz = f["xyz1_mm"][()][sel].astype(np.float32)
        itype = f["int_type"][()][sel]
    return maps, npe, xyz, itype


def load_photo(path):
    """Truth selection: contained photoelectric events.
    Returns maps (n,8,8) float32, npe (n,), xyz1 (n,3), int_type (n,)."""
    with h5py.File(path, "r") as f:
        sel = (f["int_type"][()] == 0) & (f["edep_kev"][()] > CONTAINED_KEV)
    return _select(path, sel)


def load_window(path, fwhm, nsigma=2.0, seed=2026):
    """Realistic selection: measured energy inside 511 +- nsigma * sigma.

    The measured energy is the true deposited energy smeared with the FULL
    detector resolution (fractional FWHM at 511 keV given by fwhm), sigma
    scaling as sqrt(E) below the peak. Fully contained events sit at exactly
    511 keV before the smear, so the window keeps 95.4% of them (nsigma = 2)
    and admits the partial-containment leakage from below. Seeded: the
    selection is reproducible.
    """
    with h5py.File(path, "r") as f:
        edep = f["edep_kev"][()].astype(np.float64)
    sigma511 = fwhm / 2.355 * E0_KEV
    sigma = sigma511 * np.sqrt(np.maximum(edep, 0.0) / E0_KEV)
    rng = np.random.default_rng(seed)
    e_meas = edep + rng.normal(0.0, 1.0, len(edep)) * sigma
    sel = np.abs(e_meas - E0_KEV) < nsigma * sigma511
    return _select(path, sel)


def d4_ops(w_mm):
    """The 8 (map transform, xyz transform) pairs of the square symmetry."""
    ops = []
    for transpose in (False, True):
        for fx in (False, True):
            for fy in (False, True):
                def m_op(m, fx=fx, fy=fy, transpose=transpose):
                    if fx:
                        m = np.flip(m, axis=-1)   # i (x) axis
                    if fy:
                        m = np.flip(m, axis=-2)   # j (y) axis
                    if transpose:
                        m = np.swapaxes(m, -1, -2)
                    return m

                def t_op(xyz, fx=fx, fy=fy, transpose=transpose, w=w_mm):
                    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
                    if fx:
                        x = w - x
                    if fy:
                        y = w - y
                    if transpose:
                        x, y = y, x
                    return np.stack([x, y, z], axis=-1)

                ops.append((m_op, t_op))
    return ops


def d4_expand(maps, npe, xyz, w_mm=48.0):
    """Static 8x expansion of (maps, npe, xyz) over the D4 group."""
    ms, ns, ts = [], [], []
    for m_op, t_op in d4_ops(w_mm):
        ms.append(np.ascontiguousarray(m_op(maps)))
        ns.append(npe)
        ts.append(t_op(xyz))
    return np.concatenate(ms), np.concatenate(ns), np.concatenate(ts)


def _selftest():
    rng = np.random.default_rng(0)
    maps = np.zeros((5, 8, 8), np.float32)
    xyz = np.zeros((5, 3), np.float32)
    for e in range(5):
        i, j = rng.integers(0, 8, 2)
        maps[e, j, i] = 1.0
        xyz[e] = ((i + 0.5) * PITCH, (j + 0.5) * PITCH, rng.uniform(0, 37))
    for m_op, t_op in d4_ops(48.0):
        m2, t2 = m_op(maps), t_op(xyz)
        for e in range(5):
            j2, i2 = np.argwhere(m2[e] > 0)[0]
            assert np.allclose(((i2 + 0.5) * PITCH, (j2 + 0.5) * PITCH),
                               t2[e, :2]), "D4 map/target mismatch"
            assert t2[e, 2] == xyz[e, 2]
    print("D4 self-test passed: all 8 ops map pixel centres onto transformed targets")


if __name__ == "__main__":
    _selftest()
