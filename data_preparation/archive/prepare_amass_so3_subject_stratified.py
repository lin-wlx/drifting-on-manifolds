#!/usr/bin/env python
"""
prepare_amass_so3.py

Generic AMASS -> SO(3) preprocessing script for the dissertation's
exact-manifold vs Euclidean drifting experiments.

Expected input
--------------
An EXTRACTED AMASS dataset folder containing files ending in:
    *_poses.npz

The folder structure may be nested arbitrarily; the script searches recursively.

For each motion sequence it:
1. Loads poses[:, :3], the AMASS root/global orientation in axis-angle form.
2. Converts it to a 3x3 rotation matrix R in SO(3).
3. Downsamples to approximately target_fps (default 10 Hz).
4. Keeps absolute R_t and relative R_0^T R_t versions.
5. Splits WHOLE motion sequences into train/validation/test.
6. Saves flattened 3x3 matrices as 9-vectors for the drifting MLP.
7. Performs numerical SO(3) checks and writes a JSON summary.

Example
-------
python prepare_amass_so3.py path/to/AMASS/DanceDB --output path/to/prepared

Dependencies
------------
numpy, scipy, matplotlib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


def discover_pose_files(dataset_root: Path) -> list[Path]:
    """Find every AMASS motion file recursively."""
    pose_files = sorted(dataset_root.rglob("*_poses.npz"))
    if not pose_files:
        raise FileNotFoundError(
            f"No '*_poses.npz' files were found under:\n{dataset_root}\n\n"
            "Make sure you point the script at the EXTRACTED AMASS dataset folder."
        )
    return pose_files


def infer_dataset_name(dataset_root: Path) -> str:
    return dataset_root.name.replace(" ", "_")


def infer_subject_id(path: Path, dataset_root: Path) -> str:
    """
    Use the first directory below the dataset root as a subject/session ID.

    Examples:
      DanceDB/20120731_StefanosTheodorou/foo_poses.npz -> 20120731_StefanosTheodorou
      CMU/01/foo_poses.npz                            -> 01
    """
    rel = path.relative_to(dataset_root)
    if len(rel.parts) >= 2:
        return rel.parts[0]
    return "unknown"


def split_sequences_stratified(
    pose_files: list[Path],
    dataset_root: Path,
    seed: int,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> dict[str, str]:
    """
    Fixed train/validation/test split at the WHOLE-SEQUENCE level.

    We stratify by the first directory below the dataset root when possible.
    Frames from one motion sequence can never leak across splits.
    """
    rng = np.random.default_rng(seed)

    by_subject: dict[str, list[Path]] = {}
    for path in pose_files:
        subject = infer_subject_id(path, dataset_root)
        by_subject.setdefault(subject, []).append(path)

    assignments: dict[str, str] = {}

    for subject in sorted(by_subject):
        seqs = np.array(sorted(by_subject[subject]), dtype=object)
        seqs = seqs[rng.permutation(len(seqs))]
        n = len(seqs)

        if n == 1:
            n_train, n_val = 1, 0
        elif n == 2:
            n_train, n_val = 1, 0
        else:
            n_val = max(1, int(round(val_fraction * n)))
            n_test = max(1, int(round((1.0 - train_fraction - val_fraction) * n)))
            n_train = n - n_val - n_test
            if n_train < 1:
                n_train = 1
                if n_val > n_test:
                    n_val -= 1
                else:
                    n_test -= 1

        for p in seqs[:n_train]:
            assignments[str(p)] = "train"
        for p in seqs[n_train:n_train + n_val]:
            assignments[str(p)] = "val"
        for p in seqs[n_train + n_val:]:
            assignments[str(p)] = "test"

    return assignments


def choose_frame_indices(num_frames: int, source_fps: float, target_fps: float) -> np.ndarray:
    """
    Downsample approximately to target_fps using time-based sampling.

    Examples:
      120 Hz -> about every 12th frame
       60 Hz -> about every 6th frame
       30 Hz -> about every 3rd frame
    """
    if source_fps <= 0:
        raise ValueError(f"Invalid source frame rate: {source_fps}")
    if target_fps <= 0:
        raise ValueError(f"Invalid target frame rate: {target_fps}")
    if target_fps >= source_fps:
        return np.arange(num_frames, dtype=np.int32)

    duration = (num_frames - 1) / source_fps
    sample_times = np.arange(0.0, duration + 1e-12, 1.0 / target_fps)
    indices = np.rint(sample_times * source_fps).astype(np.int32)
    indices = np.clip(indices, 0, num_frames - 1)
    return np.unique(indices)


def so3_checks(R: np.ndarray) -> dict[str, float]:
    """Check R^T R = I and det(R) = +1 numerically."""
    I = np.eye(3, dtype=R.dtype)
    RtR = np.swapaxes(R, -1, -2) @ R
    orth_error = np.linalg.norm(RtR - I, axis=(-2, -1))
    det = np.linalg.det(R)

    return {
        "max_orthogonality_error": float(np.max(orth_error)),
        "mean_orthogonality_error": float(np.mean(orth_error)),
        "max_abs_det_minus_1": float(np.max(np.abs(det - 1.0))),
        "min_determinant": float(np.min(det)),
        "max_determinant": float(np.max(det)),
    }


def angle_from_identity(R: np.ndarray) -> np.ndarray:
    """Principal SO(3) distance from I, in radians."""
    tr = np.trace(R, axis1=-2, axis2=-1)
    c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(c)


def prepare_dataset(
    dataset_root: Path,
    output_dir: Path,
    target_fps: float = 10.0,
    split_seed: int = 42,
    make_plot: bool = True,
) -> None:

    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = infer_dataset_name(dataset_root)

    print("=" * 72)
    print(f"Dataset: {dataset_name}")
    print(f"Input folder: {dataset_root}")
    print(f"Output folder: {output_dir}")
    print("=" * 72)

    pose_files = discover_pose_files(dataset_root)
    print(f"Found {len(pose_files):,} '*_poses.npz' motion files.")

    subjects = sorted({infer_subject_id(p, dataset_root) for p in pose_files})
    print(f"Found {len(subjects):,} subject/session folders.")

    sequence_split = split_sequences_stratified(
        pose_files=pose_files,
        dataset_root=dataset_root,
        seed=split_seed,
    )

    R_abs_chunks = []
    R_rel_chunks = []
    root_rotvec_chunks = []
    subject_chunks = []
    sequence_chunks = []
    split_chunks = []
    frame_index_chunks = []
    time_sec_chunks = []

    total_raw_frames = 0
    fps_values = []
    skipped_files = []

    for i, path in enumerate(pose_files, start=1):
        try:
            with np.load(path, allow_pickle=True) as data:
                if "poses" not in data:
                    raise KeyError("missing 'poses' field")
                if "mocap_framerate" not in data:
                    raise KeyError("missing 'mocap_framerate' field")

                poses = np.asarray(data["poses"])
                source_fps = float(np.asarray(data["mocap_framerate"]).item())

        except Exception as exc:
            skipped_files.append({"file": str(path), "reason": repr(exc)})
            print(f"[WARNING] Skipping {path.name}: {exc}")
            continue

        if poses.ndim != 2 or poses.shape[1] < 3:
            skipped_files.append({
                "file": str(path),
                "reason": f"unexpected poses shape {poses.shape}",
            })
            print(f"[WARNING] Skipping {path.name}: unexpected poses shape {poses.shape}")
            continue

        total_raw_frames += len(poses)
        fps_values.append(source_fps)

        # AMASS convention: first 3 pose entries are root/global orientation.
        root_rotvec_full = poses[:, :3].astype(np.float64)

        indices = choose_frame_indices(
            num_frames=len(poses),
            source_fps=source_fps,
            target_fps=target_fps,
        )

        root_rotvec = root_rotvec_full[indices]

        # Axis-angle -> 3x3 rotation matrix in SO(3).
        R_abs = Rotation.from_rotvec(root_rotvec).as_matrix()

        # Relative orientation R_0^T R_t.
        R0 = Rotation.from_rotvec(root_rotvec_full[0]).as_matrix()
        R_rel = np.einsum("ij,njk->nik", R0.T, R_abs)

        n_keep = len(indices)
        subject = infer_subject_id(path, dataset_root)
        sequence = str(path.relative_to(dataset_root).with_suffix(""))
        split = sequence_split[str(path)]

        R_abs_chunks.append(R_abs.astype(np.float32))
        R_rel_chunks.append(R_rel.astype(np.float32))
        root_rotvec_chunks.append(root_rotvec.astype(np.float32))
        subject_chunks.append(np.full(n_keep, subject))
        sequence_chunks.append(np.full(n_keep, sequence))
        split_chunks.append(np.full(n_keep, split))
        frame_index_chunks.append(indices.astype(np.int32))
        time_sec_chunks.append((indices / source_fps).astype(np.float32))

        if i % 100 == 0 or i == len(pose_files):
            print(f"Processed {i:,}/{len(pose_files):,} motion files...")

    if not R_abs_chunks:
        raise RuntimeError("No motion files were successfully processed.")

    R_abs = np.concatenate(R_abs_chunks, axis=0)
    R_rel = np.concatenate(R_rel_chunks, axis=0)
    root_rotvec = np.concatenate(root_rotvec_chunks, axis=0)
    subject = np.concatenate(subject_chunks)
    sequence = np.concatenate(sequence_chunks)
    split = np.concatenate(split_chunks)
    frame_index = np.concatenate(frame_index_chunks)
    time_sec = np.concatenate(time_sec_chunks)

    # Flatten each 3x3 matrix to a 9-vector for the neural network.
    X_abs = R_abs.reshape(-1, 9)
    X_rel = R_rel.reshape(-1, 9)

    abs_check = so3_checks(R_abs)
    rel_check = so3_checks(R_rel)

    print("\nSO(3) checks for absolute rotations:")
    print(json.dumps(abs_check, indent=2))
    print("\nSO(3) checks for relative rotations:")
    print(json.dumps(rel_check, indent=2))

    if abs_check["max_orthogonality_error"] > 1e-4:
        raise RuntimeError("Absolute rotations failed the orthogonality check.")
    if abs_check["max_abs_det_minus_1"] > 1e-4:
        raise RuntimeError("Absolute rotations failed the determinant check.")

    ready = {}
    sample_counts = {}
    sequence_counts = {}

    for split_name in ("train", "val", "test"):
        mask = split == split_name
        ready[f"{split_name}_abs"] = X_abs[mask]
        ready[f"{split_name}_rel"] = X_rel[mask]
        ready[f"{split_name}_subject"] = subject[mask]
        ready[f"{split_name}_sequence"] = sequence[mask]
        sample_counts[split_name] = int(mask.sum())
        sequence_counts[split_name] = int(len(np.unique(sequence[mask])))

    ready_path = output_dir / f"{dataset_name}_SO3_ready_{target_fps:g}hz.npz"
    np.savez_compressed(
        ready_path,
        **ready,
        representation_note=np.array(
            "Each sample is a row-major flattened 3x3 rotation matrix. "
            "abs = absolute AMASS root orientation R_t. "
            "rel = R_0^T R_t within the same motion sequence."
        ),
        split_note=np.array(
            "Fixed train/validation/test split over whole motion sequences, "
            "stratified by the first folder below the dataset root. "
            "Frames from one motion sequence never appear in multiple splits."
        ),
        split_seed=np.array(split_seed, dtype=np.int32),
        target_fps=np.array(target_fps, dtype=np.float32),
    )

    master_path = output_dir / f"{dataset_name}_SO3_master_{target_fps:g}hz.npz"
    np.savez_compressed(
        master_path,
        R_abs=R_abs,
        R_rel=R_rel,
        X_abs=X_abs,
        X_rel=X_rel,
        root_rotvec=root_rotvec,
        subject=subject,
        sequence=sequence,
        split=split,
        frame_index=frame_index,
        time_sec=time_sec,
        target_fps=np.array(target_fps, dtype=np.float32),
        split_seed=np.array(split_seed, dtype=np.int32),
    )

    abs_angles = angle_from_identity(R_abs)
    rel_angles = angle_from_identity(R_rel)

    summary = {
        "dataset_name": dataset_name,
        "input_folder": str(dataset_root),
        "pose_files_found": len(pose_files),
        "pose_files_processed": len(R_abs_chunks),
        "pose_files_skipped": len(skipped_files),
        "subject_session_groups": len(subjects),
        "raw_frames": int(total_raw_frames),
        "retained_samples": int(len(X_abs)),
        "source_fps_min": float(np.min(fps_values)),
        "source_fps_median": float(np.median(fps_values)),
        "source_fps_max": float(np.max(fps_values)),
        "target_fps": float(target_fps),
        "split_seed": int(split_seed),
        "sample_counts": sample_counts,
        "sequence_counts": sequence_counts,
        "absolute_so3_checks": abs_check,
        "relative_so3_checks": rel_check,
        "absolute_angle_from_identity_radians": {
            "median": float(np.median(abs_angles)),
            "p90": float(np.quantile(abs_angles, 0.90)),
            "max": float(np.max(abs_angles)),
        },
        "relative_angle_from_sequence_start_radians": {
            "median": float(np.median(rel_angles)),
            "p90": float(np.quantile(rel_angles, 0.90)),
            "max": float(np.max(rel_angles)),
        },
        "skipped_files": skipped_files,
    }

    summary_path = output_dir / f"{dataset_name}_SO3_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_path = None
    if make_plot:
        rng = np.random.default_rng(123)
        n_plot = min(8000, len(R_abs))
        plot_idx = rng.choice(len(R_abs), size=n_plot, replace=False)

        abs_rotvec = Rotation.from_matrix(R_abs[plot_idx]).as_rotvec()
        rel_rotvec = Rotation.from_matrix(R_rel[plot_idx]).as_rotvec()

        fig = plt.figure(figsize=(14, 6))

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.scatter(abs_rotvec[:, 0], abs_rotvec[:, 1], abs_rotvec[:, 2], s=3, alpha=0.25)
        ax1.set_title(f"{dataset_name}: absolute root orientation")
        ax1.set_xlabel("rotation-vector x")
        ax1.set_ylabel("rotation-vector y")
        ax1.set_zlabel("rotation-vector z")

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.scatter(rel_rotvec[:, 0], rel_rotvec[:, 1], rel_rotvec[:, 2], s=3, alpha=0.25)
        ax2.set_title(f"{dataset_name}: relative to sequence start")
        ax2.set_xlabel("rotation-vector x")
        ax2.set_ylabel("rotation-vector y")
        ax2.set_zlabel("rotation-vector z")

        fig.suptitle("Principal axis-angle coordinates — visualisation only", fontsize=12)
        plt.tight_layout()

        plot_path = output_dir / f"{dataset_name}_SO3_axis_angle.png"
        plt.savefig(plot_path, dpi=160)
        plt.close(fig)

    print("\n" + "=" * 72)
    print("FINISHED")
    print("=" * 72)
    print(f"Raw frames:       {total_raw_frames:,}")
    print(f"Retained samples: {len(X_abs):,}\n")

    print("Samples by split:")
    for k, v in sample_counts.items():
        print(f"  {k:5s}: {v:,}")

    print("\nSequences by split:")
    for k, v in sequence_counts.items():
        print(f"  {k:5s}: {v:,}")

    print("\nCreated:")
    print(f"  READY:   {ready_path}")
    print(f"  MASTER:  {master_path}")
    print(f"  SUMMARY: {summary_path}")
    if plot_path is not None:
        print(f"  PLOT:    {plot_path}")

    print("\nFor the Colab experiment, upload the READY file:")
    print(f"  {ready_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare an extracted AMASS dataset for SO(3) drifting."
    )

    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the EXTRACTED AMASS dataset folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prepared_amass_so3"),
        help="Folder in which processed files will be saved.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=10.0,
        help="Approximate temporal sampling rate after downsampling (default: 10 Hz).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed train/validation/test split seed (default: 42).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the diagnostic axis-angle plot.",
    )

    args = parser.parse_args()

    prepare_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output,
        target_fps=args.target_fps,
        split_seed=args.seed,
        make_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
