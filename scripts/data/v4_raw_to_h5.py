#!/usr/bin/env python
"""
Convert v4 raw 3D-PDR grid output (COL128.*) into a v3-style HDF5 dataset of
Av-sorted sightlines ("models"), each a (n_points, 39) array with columns
["visual_extinction", "tgas", "tdust", "etype", "density", "radfield"] + 33 species.

Unlike v1/v2/v3, v4's raw data is a single 128^3 = 2,097,152-point 3D grid (one
turbulent GMC snapshot), not pre-cut sightlines. Two kinds of models are built:

  * axis-aligned models (+x/-x/+y/-y): built by directly slicing/reversing grid
    columns and reading the corresponding column from COL128.rayAV.fin. Exactly
    monotonic in Av by construction (3D-PDR reuses one physical ray for the whole
    grid-aligned column) - no smoothing needed.

  * diagonal models (the other 8 of the 12 HEALPix ray directions): COL128.rayAV.fin
    values for these directions are NOT self-consistent when stitched across
    neighbouring lattice points (each point's ray is cast independently), so we
    re-derive Av ourselves: a 3D DDA/voxel-traversal ray marches a true continuous
    line through the grid, trilinear-interpolating density at each traversed
    cell's entry/mid/exit point and integrating exactly via Simpson's rule (exact
    for the cubic polynomial trilinear density reduces to along a line). This is
    monotonic by construction (cumulative sum of a non-negative integrand) and is
    calibrated against the axis-aligned case's N_H/Av conversion constant.

    Entry points are sampled from all 3 "upstream" faces of the cube per
    direction (the 3 faces whose outward normal has a positive dot product with
    the ray direction, i.e. where Av~=0) at 128x128 cell-center resolution per
    face, matching the axis-aligned case's areal sampling density. A single face
    only reaches ~42% of the cube's volume for a given direction (verified by an
    exact back-projection calculation); 3-face entry reaches effectively 100%,
    with some interior cells covered redundantly by more than one face - a
    genuine bonus (independent, differently-angled sightlines through the same
    region), not wasted computation.

Usage:
    python scripts/data/v4_raw_to_h5.py data/v4 data/processed/3dpdr_dataset_v4.h5
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

N = 128
SPACING = 0.125  # pc
DOMAIN = N * SPACING  # 16.0 pc
PC_TO_CM = 3.0857e18
DEFAULT_CONVERSION_CONSTANT = 1.588e21  # cm^-2 mag^-1, calibrated against rayAV.fin
BUFFER_SIZE = 512
MIN_POINTS = 2  # drop degenerate rays with fewer than this many crossing points

SPECIES = [
    "H3+",
    "He+",
    "Mg",
    "H2+",
    "O2",
    "CH5+",
    "CH4+",
    "O+",
    "OH+",
    "Mg+",
    "C+",
    "CH4",
    "H2O+",
    "H3O+",
    "CO+",
    "O2+",
    "CH2",
    "H2O",
    "H+",
    "CH3+",
    "CH",
    "CH3",
    "HCO+",
    "CH2+",
    "C",
    "He",
    "CH+",
    "CO",
    "OH",
    "O",
    "H2",
    "H",
    "e-",
]

# field_all column layout (38 cols): tgas, tdust, etype, density, radfield, 33 species
FIELD_TGAS, FIELD_TDUST, FIELD_ETYPE, FIELD_DENSITY, FIELD_RADFIELD = range(5)
FIELD_SPECIES_START = 5
N_FIELDS = 5 + len(SPECIES)

PLANES = np.arange(N + 1) * SPACING  # 129 grid-line positions, 0..16.0
FACE_LABEL = {0: "x", 1: "y", 2: "z"}


# --------------------------------------------------------------------------
# Loading raw data
# --------------------------------------------------------------------------


def load_pdr_field_all(data_path: Path) -> np.ndarray:
    """Load COL128.pdr.fin into a (N,N,N,38) grid of [tgas,tdust,etype,density,radfield]+species."""
    arr = pd.read_csv(
        data_path / "COL128.pdr.fin", sep=r"\s+", header=None, dtype=np.float64
    ).values
    idx = arr[:, 0].astype(np.int64) - 1
    assert idx.min() == 0 and idx.max() == N**3 - 1, "unexpected index range in pdr.fin"
    ix = idx // (N * N)
    iy = (idx // N) % N
    iz = idx % N
    # sanity check: z is the fastest-varying axis (index=ix*N*N+iy*N+iz)
    assert np.array_equal(idx, ix * N * N + iy * N + iz)

    cols = [4, 5, 6, 7, 8] + list(range(9, 9 + len(SPECIES)))
    field_all = np.empty((N, N, N, N_FIELDS), dtype=np.float32)
    field_all[ix, iy, iz, :] = arr[:, cols].astype(np.float32)
    return field_all


def load_rayav_grid(data_path: Path) -> np.ndarray:
    """Load COL128.rayAV.fin (rows already in sequential grid-index order) -> (N,N,N,12)."""
    arr = pd.read_csv(
        data_path / "COL128.rayAV.fin", sep=r"\s+", header=None, dtype=np.float32
    ).values
    assert arr.shape == (N**3, 12)
    return arr.reshape(N, N, N, 12)


def load_ray_directions(data_path: Path):
    """Return unit direction vectors (12,3) and axis-aligned/diagonal index sets."""
    raydir = np.loadtxt(data_path / "COL128.rayDir.fin")
    theta, phi = raydir[:, 0], raydir[:, 1]
    dirs = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        axis=1,
    )

    axis_aligned_mask = np.isclose(theta, np.pi / 2, atol=1e-3)
    assert axis_aligned_mask.sum() == 4, "expected exactly 4 axis-aligned ray directions"
    assert (~axis_aligned_mask).sum() == 8, "expected exactly 8 diagonal ray directions"

    axis_map = {}
    for i in np.where(axis_aligned_mask)[0]:
        d = dirs[i]
        if d[0] > 0.9:
            axis_map["+x"] = i
        elif d[0] < -0.9:
            axis_map["-x"] = i
        elif d[1] > 0.9:
            axis_map["+y"] = i
        elif d[1] < -0.9:
            axis_map["-y"] = i
    assert set(axis_map) == {"+x", "-x", "+y", "-y"}, f"could not identify axis rays: {axis_map}"

    diagonal_indices = np.where(~axis_aligned_mask)[0].tolist()
    return dirs, axis_map, diagonal_indices


# --------------------------------------------------------------------------
# Axis-aligned branch (trivial: slice + reverse, Av taken straight from rayAV)
# --------------------------------------------------------------------------


def iter_axis_models(field_all: np.ndarray, rayav: np.ndarray, axis_map: dict):
    """Yield (name, (128,39) float32 array) for the 65,536 axis-aligned models."""
    for direction, ray_idx in axis_map.items():
        walk_axis = 0 if direction[-1] == "x" else 1
        other_axes = [a for a in range(3) if a != walk_axis]
        # +x/+y: ray points toward the far (index N-1) face -> Av=0 there, walk descending.
        # -x/-y: ray points toward the near (index 0) face -> Av=0 there, walk ascending.
        descending = direction[0] == "+"
        order = np.arange(N - 1, -1, -1) if descending else np.arange(N)

        for i in range(N):
            for j in range(N):
                idx = [None, None, None]
                idx[walk_axis] = slice(None)
                idx[other_axes[0]] = i
                idx[other_axes[1]] = j
                col_fields = field_all[tuple(idx)][order]  # (128, 38)
                col_av = rayav[tuple(idx)][order, ray_idx]  # (128,)

                assert (np.diff(col_av) >= -1e-4).all(), (
                    f"axis-aligned Av not monotonic for {direction} at ({i},{j})"
                )

                model = np.empty((N, 1 + N_FIELDS), dtype=np.float32)
                model[:, 0] = col_av
                model[:, 1:] = col_fields
                yield f"model_axis_{direction}_{i:03d}_{j:03d}", model


# --------------------------------------------------------------------------
# Diagonal branch: DDA ray-marching + trilinear interpolation + Simpson integration
# --------------------------------------------------------------------------


def trilinear_interp(coords: np.ndarray, field_all: np.ndarray) -> np.ndarray:
    """Trilinearly interpolate field_all (N,N,N,K) at continuous pc coords (M,3) -> (M,K).

    Edge-clamped: coordinates outside the domain are clamped to the nearest cell.
    """
    g = coords / SPACING - 0.5
    g = np.clip(g, 0.0, N - 1 - 1e-6)
    i0 = np.floor(g).astype(np.int64)
    frac = g - i0
    i1 = np.minimum(i0 + 1, N - 1)

    ix0, iy0, iz0 = i0[:, 0], i0[:, 1], i0[:, 2]
    ix1, iy1, iz1 = i1[:, 0], i1[:, 1], i1[:, 2]
    fx, fy, fz = frac[:, 0:1], frac[:, 1:2], frac[:, 2:3]

    c000 = field_all[ix0, iy0, iz0]
    c100 = field_all[ix1, iy0, iz0]
    c010 = field_all[ix0, iy1, iz0]
    c110 = field_all[ix1, iy1, iz0]
    c001 = field_all[ix0, iy0, iz1]
    c101 = field_all[ix1, iy0, iz1]
    c011 = field_all[ix0, iy1, iz1]
    c111 = field_all[ix1, iy1, iz1]

    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def march_ray(p0: np.ndarray, m: np.ndarray, t_exit: float, field_all: np.ndarray, conv_const: float):
    """March a single ray from entry point p0 in direction m until t_exit.

    Returns a (n_points, 39) float32 array [visual_extinction, tgas, tdust, etype,
    density, radfield, *species] ordered by strictly non-decreasing Av, or None if
    the ray is degenerate (too few crossing points).
    """
    if t_exit <= 0:
        return None

    t_candidates = [np.array([0.0, t_exit])]
    for axis in range(3):
        m_axis = m[axis]
        t_axis = (PLANES - p0[axis]) / m_axis
        t_candidates.append(t_axis)
    t_all = np.concatenate(t_candidates)
    t_all = np.clip(t_all, 0.0, t_exit)
    t_all = np.unique(t_all)  # sorted + exact-deduplicated
    if t_all.size >= 2:
        # merge near-duplicate crossings (e.g. a ray passing near a grid edge/corner
        # can produce two crossing times that differ only by float roundoff)
        keep = np.concatenate([[True], np.diff(t_all) > 1e-6])
        t_all = t_all[keep]

    if t_all.size < MIN_POINTS:
        return None

    pts = p0[None, :] + t_all[:, None] * m[None, :]
    t_mid = 0.5 * (t_all[:-1] + t_all[1:])
    pts_mid = p0[None, :] + t_mid[:, None] * m[None, :]

    vals = trilinear_interp(pts, field_all)  # (n_points, 38)
    vals_mid = trilinear_interp(pts_mid, field_all)  # (n_points-1, 38)

    dens = vals[:, FIELD_DENSITY]
    dens_mid = vals_mid[:, FIELD_DENSITY]
    seg_len_pc = np.diff(t_all)
    seg_integral = seg_len_pc / 6.0 * (dens[:-1] + 4 * dens_mid + dens[1:]) * PC_TO_CM
    coldens = np.concatenate([[0.0], np.cumsum(seg_integral)])
    av = coldens / conv_const

    if not (np.diff(av) >= -1e-6).all():
        return None  # numerical edge case (near-zero-length segment); drop defensively

    row = np.empty((t_all.size, 1 + N_FIELDS), dtype=np.float32)
    row[:, 0] = av
    row[:, 1:] = vals
    return row


def compute_t_exit(p0: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Slab-method distance to domain boundary for a batch of rays sharing direction m."""
    t_max = np.full(p0.shape[0], np.inf)
    for axis in range(3):
        m_axis = m[axis]
        boundary = DOMAIN if m_axis > 0 else 0.0
        t_axis = (boundary - p0[:, axis]) / m_axis
        t_max = np.minimum(t_max, t_axis)
    return t_max


def entry_points_for_face(face_axis: int, face_value: float, other_axes: list) -> np.ndarray:
    coords = (np.arange(N) + 0.5) * SPACING
    A, B = np.meshgrid(coords, coords, indexing="ij")
    p0 = np.empty((N * N, 3), dtype=np.float64)
    p0[:, other_axes[0]] = A.ravel()
    p0[:, other_axes[1]] = B.ravel()
    p0[:, face_axis] = face_value
    return p0


def _march_batch(indices, p0_batch, m, t_exit_batch, field_all, conv_const):
    results = []
    for k, p0, t_exit in zip(indices, p0_batch, t_exit_batch):
        row = march_ray(p0, m, t_exit, field_all, conv_const)
        if row is not None:
            results.append((k, row))
    return results


def iter_diagonal_models(
    field_all: np.ndarray,
    dirs: np.ndarray,
    diagonal_indices: list,
    conv_const: float,
    n_jobs: int,
    limit_entries: int | None = None,
):
    """Yield (name, (n_points,39) float32 array) for the diagonal-ray models."""
    for dir_idx in diagonal_indices:
        d = dirs[dir_idx]
        m = -d  # march from Av~=0 (upstream face) into the box

        for axis in range(3):
            face_value = DOMAIN if d[axis] > 0 else 0.0
            other_axes = [a for a in range(3) if a != axis]
            p0_all = entry_points_for_face(axis, face_value, other_axes)
            if limit_entries is not None:
                p0_all = p0_all[:limit_entries]
            t_exit_all = compute_t_exit(p0_all, m)

            n_rays = p0_all.shape[0]
            chunk_size = max(1, min(2048, n_rays))
            chunks = [
                (list(range(s, min(s + chunk_size, n_rays))),)
                for s in range(0, n_rays, chunk_size)
            ]

            batch_results = Parallel(n_jobs=n_jobs)(
                delayed(_march_batch)(
                    idxs, p0_all[idxs], m, t_exit_all[idxs], field_all, conv_const
                )
                for (idxs,) in chunks
            )

            for batch in batch_results:
                for k, row in batch:
                    i, j = divmod(k, N)
                    name = (
                        f"model_diag_{dir_idx:02d}_{FACE_LABEL[axis]}_{i:03d}_{j:03d}"
                    )
                    yield name, row


# --------------------------------------------------------------------------
# CLI / orchestration
# --------------------------------------------------------------------------


def write_models(model_iter, store_path: Path, total: int | None = None):
    buffer = {}
    n_written = 0
    for name, model in tqdm(model_iter, total=total, desc="writing models"):
        buffer[name] = model
        if len(buffer) >= BUFFER_SIZE:
            with h5py.File(store_path, "a") as fh:
                for model_name, data in buffer.items():
                    fh.create_dataset(
                        name=f"{model_name}/pdr",
                        data=data,
                        dtype="float32",
                        compression="gzip",
                    )
            n_written += len(buffer)
            buffer = {}
    if buffer:
        with h5py.File(store_path, "a") as fh:
            for model_name, data in buffer.items():
                fh.create_dataset(
                    name=f"{model_name}/pdr",
                    data=data,
                    dtype="float32",
                    compression="gzip",
                )
        n_written += len(buffer)
    return n_written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=str, help="Directory containing COL128.* raw files")
    parser.add_argument("output_file", type=str, help="Path to output HDF5 file")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--conversion-constant", type=float, default=DEFAULT_CONVERSION_CONSTANT)
    parser.add_argument(
        "--limit-entries",
        type=int,
        default=None,
        help="Cap entry points per (direction,face) group - for quick testing only",
    )
    parser.add_argument(
        "--skip-axis", action="store_true", help="Skip axis-aligned models (testing only)"
    )
    parser.add_argument(
        "--skip-diagonal", action="store_true", help="Skip diagonal models (testing only)"
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        output_file.unlink()

    print("Loading raw grid data...")
    field_all = load_pdr_field_all(data_path)
    rayav = load_rayav_grid(data_path)
    dirs, axis_map, diagonal_indices = load_ray_directions(data_path)
    print(f"  axis-aligned ray indices: {axis_map}")
    print(f"  diagonal ray indices: {diagonal_indices}")

    params_path = data_path / "COL128.params"
    if params_path.exists():
        g0_ext, zeta, metallicity = np.loadtxt(params_path)
        with h5py.File(output_file, "a") as fh:
            fh.attrs["G0_ext"] = float(g0_ext)
            fh.attrs["zeta"] = float(zeta)
            fh.attrs["metallicity"] = float(metallicity)

    if not args.skip_axis:
        print("Building axis-aligned models...")
        n_axis = write_models(
            iter_axis_models(field_all, rayav, axis_map), output_file, total=4 * N * N
        )
        print(f"  wrote {n_axis} axis-aligned models")

    if not args.skip_diagonal:
        print("Building diagonal models (ray-marching)...")
        total_diag = (
            len(diagonal_indices)
            * 3
            * (args.limit_entries if args.limit_entries else N * N)
        )
        n_diag = write_models(
            iter_diagonal_models(
                field_all,
                dirs,
                diagonal_indices,
                args.conversion_constant,
                args.n_jobs,
                limit_entries=args.limit_entries,
            ),
            output_file,
            total=total_diag,
        )
        print(f"  wrote {n_diag} diagonal models")

    print(f"\nDone: {output_file}")


if __name__ == "__main__":
    main()
