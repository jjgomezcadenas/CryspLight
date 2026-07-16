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
        return {"maps": f["maps"][()][sel].astype(np.float32),
                "npe": f["npe"][()][sel].astype(np.float32),
                "xyz1": f["xyz1_mm"][()][sel].astype(np.float32),
                "xyz2": f["xyz2_mm"][()][sel].astype(np.float32),
                "e1": f["e1_kev"][()][sel],
                "edep": f["edep_kev"][()][sel],
                "n_int": f["n_int"][()][sel],
                "itype": f["int_type"][()][sel],
                "row": np.where(sel)[0]}     # original events.h5 row: join key


def load_photo(path):
    """Truth selection: contained photoelectric events. Returns a dict with
    maps (n,8,8) float32, npe, xyz1 (n,3), xyz2, n_int, itype."""
    with h5py.File(path, "r") as f:
        sel = (f["int_type"][()] == 0) & (f["edep_kev"][()] > CONTAINED_KEV)
    return _select(path, sel)


def two_site_targets(d):
    """(n, 6) depth-ordered targets: columns 0-2 the shallower site, 3-5 the
    deeper one; both set to the single site when only one deposit exists
    (the coincident convention: 'single site' = 'the two points coincide')."""
    a, b = d["xyz1"], d["xyz2"]
    multi = (d["n_int"] >= 2)[:, None]
    first_shallow = (a[:, 2] <= b[:, 2])[:, None]
    shallow = np.where(multi & ~first_shallow, b, a)
    deep = np.where(multi & first_shallow, b, a)
    return np.concatenate([shallow, deep], axis=1).astype(np.float32)


def window_mask(path, fwhm, nsigma=2.0, seed=2026):
    """Boolean event mask of the energy-window selection (see load_window).
    Shared by training and truth studies so the selection can never drift."""
    with h5py.File(path, "r") as f:
        edep = f["edep_kev"][()].astype(np.float64)
    sigma511 = fwhm / 2.355 * E0_KEV
    sigma = sigma511 * np.sqrt(np.maximum(edep, 0.0) / E0_KEV)
    rng = np.random.default_rng(seed)
    e_meas = edep + rng.normal(0.0, 1.0, len(edep)) * sigma
    return np.abs(e_meas - E0_KEV) < nsigma * sigma511


def load_window(path, fwhm, nsigma=2.0, seed=2026):
    """Realistic selection: measured energy inside 511 +- nsigma * sigma.

    The measured energy is the true deposited energy smeared with the FULL
    detector resolution (fractional FWHM at 511 keV given by fwhm), sigma
    scaling as sqrt(E) below the peak. Fully contained events sit at exactly
    511 keV before the smear, so the window keeps 95.4% of them (nsigma = 2)
    and admits the partial-containment leakage from below. Seeded: the
    selection is reproducible.
    """
    return _select(path, window_mask(path, fwhm, nsigma=nsigma, seed=seed))


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
    """Static 8x expansion of (maps, npe, xyz) over the D4 group. xyz may
    carry one target triplet (n, 3) or several (n, 3k): each triplet is
    transformed identically, so depth ordering is preserved."""
    ms, ns, ts = [], [], []
    for m_op, t_op in d4_ops(w_mm):
        ms.append(np.ascontiguousarray(m_op(maps)))
        ns.append(npe)
        ts.append(np.concatenate([t_op(xyz[:, c:c + 3])
                                  for c in range(0, xyz.shape[1], 3)], axis=1))
    return np.concatenate(ms), np.concatenate(ns), np.concatenate(ts)


def save_test_events(outpath, cols):
    """One dataset per column (plain h5py; no pytables dependency).
    cols: dict of equal-length 1-D arrays."""
    with h5py.File(outpath, "w") as f:
        for k, v in cols.items():
            f[k] = np.asarray(v)


def load_test_events(tag_or_path):
    """Per-event test-split dataframe of a run: truth + every estimator's
    prediction. Accepts a run tag (cnn_csitl_win) or a path to test_events.h5.
        df = load_test_events("cnn_csitl_win")"""
    import pandas as pd
    p = Path(tag_or_path)
    if p.suffix != ".h5":
        p = Path(__file__).resolve().parent.parent / "results" / "cnn" \
            / str(tag_or_path) / "test_events.h5"
    with h5py.File(p, "r") as f:
        return pd.DataFrame({k: f[k][()] for k in sorted(f)})


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
