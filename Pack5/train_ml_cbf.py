import argparse
import csv
from pathlib import Path

import numpy as np

import ml_cbf
from ml_cbf_experiment import run_episode


BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treina a camada ML-CBF do Pack5 com simulacoes em v,w."
    )
    parser.add_argument("--episodes-per-candidate", type=int, default=1)
    parser.add_argument("--horizon", type=float, default=8.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--record-stride", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--model-out", type=Path, default=ml_cbf.DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--report-out",
        type=Path,
        default=BASE_DIR / "ml_cbf_training_report.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    candidates = list(ml_cbf.DEFAULT_CANDIDATES)
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    contexts = []
    candidate_ids = []
    scores = []
    report_rows = []

    for candidate_id, params in enumerate(candidates):
        episode_scores = []

        for episode_idx in range(args.episodes_per_candidate):
            variation_id = candidate_id * args.episodes_per_candidate + episode_idx
            metrics = run_episode(
                params=params,
                variation_id=variation_id,
                horizon=args.horizon,
                dt=args.dt,
                record_stride=args.record_stride,
                record_history=False,
            )

            episode_score = float(metrics["score"])
            episode_scores.append(episode_score)

            episode_contexts = metrics["contexts"]
            if episode_contexts.size == 0:
                episode_contexts = np.zeros((1, len(ml_cbf.FEATURE_NAMES)), dtype=float)

            contexts.append(episode_contexts)
            candidate_ids.append(
                np.full(len(episode_contexts), candidate_id, dtype=int)
            )
            scores.append(
                np.full(len(episode_contexts), episode_score, dtype=float)
            )

            report_rows.append(
                {
                    "candidate_id": candidate_id,
                    "episode": episode_idx,
                    "variation_id": variation_id,
                    "score": episode_score,
                    "completed": metrics["completed"],
                    "collided": metrics["collided"],
                    "progress_ratio": metrics["progress_ratio"],
                    "qp_failures": metrics["qp_failures"],
                    "mean_abs_cte": metrics["mean_abs_cte"],
                    "min_obstacle_clearance": metrics["min_obstacle_clearance"],
                    "min_barrier_clearance": metrics["min_barrier_clearance"],
                    **params,
                }
            )

        print(
            "candidate "
            f"{candidate_id:02d}: mean_score={np.mean(episode_scores):.2f} "
            f"params={params}"
        )

    contexts = np.vstack(contexts)
    candidate_ids = np.concatenate(candidate_ids)
    scores = np.concatenate(scores)

    model = ml_cbf.MLCBFParameterModel.fit(
        contexts=contexts,
        candidate_ids=candidate_ids,
        scores=scores,
        candidates=candidates,
    )
    model.save(args.model_out)

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with args.report_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"modelo guardado em: {args.model_out}")
    print(f"relatorio guardado em: {args.report_out}")
    print(f"amostras de treino: {len(contexts)}")


if __name__ == "__main__":
    main()
