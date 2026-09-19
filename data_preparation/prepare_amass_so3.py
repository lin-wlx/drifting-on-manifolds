#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


def discover_pose_files(dataset_root: Path) -> list[Path]:
    pose_files = sorted(dataset_root.rglob("*_poses.npz"))
    if not pose_files:
        raise FileNotFoundError(
            f"No '*_poses.npz' files found under {dataset_root}. "
            "Point the script at the EXTRACTED AMASS dataset folder."
        )
    return pose_files


def infer_dataset_name(dataset_root: Path) -> str:
    return dataset_root.name.replace(" ", "_")


def global_sequence_split(
    pose_files: list[Path],
    seed: int = 42,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    seqs = np.array(sorted(pose_files), dtype=object)
    seqs = seqs[rng.permutation(len(seqs))]

    n = len(seqs)
    n_train = int(np.floor(train_fraction * n))
    n_val = int(np.floor(val_fraction * n))

    assignments = {}
    for p in seqs[:n_train]:
        assignments[str(p)] = "train"
    for p in seqs[n_train:n_train + n_val]:
        assignments[str(p)] = "val"
    for p in seqs[n_train + n_val:]:
        assignments[str(p)] = "test"
    return assignments


def choose_frame_indices(num_frames: int, source_fps: float, target_fps: float) -> np.ndarray:
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

    sequence_split = global_sequence_split(pose_files, seed=split_seed)

    R_abs_chunks = []
    R_rel_chunks = []
    root_rotvec_chunks = []
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

        root_rotvec_full = poses[:, :3].astype(np.float64)

        indices = choose_frame_indices(
            num_frames=len(poses),
            source_fps=source_fps,
            target_fps=target_fps,
        )

        root_rotvec = root_rotvec_full[indices]
        R_abs = Rotation.from_rotvec(root_rotvec).as_matrix()

        R0 = Rotation.from_rotvec(root_rotvec_full[0]).as_matrix()
        R_rel = np.einsum("ij,njk->nik", R0.T, R_abs)

        n_keep = len(indices)
        sequence = str(path.relative_to(dataset_root).with_suffix(""))
        split = sequence_split[str(path)]

        R_abs_chunks.append(R_abs.astype(np.float32))
        R_rel_chunks.append(R_rel.astype(np.float32))
        root_rotvec_chunks.append(root_rotvec.astype(np.float32))

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
    sequence = np.concatenate(sequence_chunks)
    split = np.concatenate(split_chunks)
    frame_index = np.concatenate(frame_index_chunks)
    time_sec = np.concatenate(time_sec_chunks)

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
            "Fixed global 80/10/10 split over complete motion sequences. "
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
        "split_method": "global sequence-level 80/10/10",
        "split_seed": int(split_seed),
        "pose_files_found": len(pose_files),
        "pose_files_processed": len(R_abs_chunks),
        "pose_files_skipped": len(skipped_files),
        "raw_frames": int(total_raw_frames),
        "retained_samples": int(len(X_abs)),
        "source_fps_min": float(np.min(fps_values)),
        "source_fps_median": float(np.median(fps_values)),
        "source_fps_max": float(np.max(fps_values)),
        "target_fps": float(target_fps),
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
    print(f"Retained samples: {len(X_abs):,}")

    print("\nSamples by split:")
    for k, v in sample_counts.items():
        print(f"  {k:5s}: {v:,}")

    print("\nSequences by split:")
    for k, v in sequence_counts.items():
        pct = 100.0 * v / len(pose_files)
        print(f"  {k:5s}: {v:,} ({pct:.1f}%)")

    print("\nCreated:")
    print(f"  READY:   {ready_path}")
    print(f"  MASTER:  {master_path}")
    print(f"  SUMMARY: {summary_path}")
    if plot_path is not None:
        print(f"  PLOT:    {plot_path}")

    print("\nFor the Colab experiment, upload ONLY the READY file:")
    print(f"  {ready_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare AMASS data for SO(3) drifting with a global 80/10/10 sequence split."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prepared_amass_so3"),
    )
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")

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
