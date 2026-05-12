import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import ml_cbf
from ml_cbf_experiment import DEFAULT_BASELINE_PARAMS, run_episode


BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compara Pack5 baseline CBF-QP contra ML-CBF em v,w."
    )
    parser.add_argument("--model", type=Path, default=ml_cbf.DEFAULT_MODEL_PATH)
    parser.add_argument("--horizon", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--variation-id", type=int, default=0)
    parser.add_argument(
        "--pdf-out",
        type=Path,
        default=BASE_DIR / "ml_cbf_comparison.pdf",
    )
    return parser.parse_args()


def print_metrics(name, metrics):
    print(
        f"{name}: score={metrics['score']:.2f}, "
        f"progress={metrics['progress_ratio']:.3f}, "
        f"completed={metrics['completed']}, "
        f"collided={metrics['collided']}, "
        f"qp_failures={metrics['qp_failures']}, "
        f"mean_abs_cte={metrics['mean_abs_cte']:.3f}, "
        f"min_obs={metrics['min_obstacle_clearance']:.3f}, "
        f"min_bar={metrics['min_barrier_clearance']:.3f}"
    )


def plot_comparison(baseline, learned, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(baseline["px"], baseline["py"], "--", linewidth=1, label="spline")
    ax.plot(
        baseline["inner_bar"][:, 0],
        baseline["inner_bar"][:, 1],
        "-",
        linewidth=2,
        label="barreira interna",
    )
    ax.plot(
        baseline["outer_bar"][:, 0],
        baseline["outer_bar"][:, 1],
        "-",
        linewidth=2,
        label="barreira externa",
    )

    for obs in baseline["obstacles"]:
        ax.add_patch(Circle((obs["x"], obs["y"]), obs["r"], fill=False))

    ax.plot(
        baseline["history"]["x"],
        baseline["history"]["y"],
        "-",
        linewidth=2,
        label="baseline CBF-QP",
    )
    ax.plot(
        learned["history"]["x"],
        learned["history"]["y"],
        "-",
        linewidth=2,
        label="ML-CBF + QP",
    )

    ax.set_aspect("equal", "box")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Pack5: baseline CBF-QP vs ML-CBF")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado: {args.model}. "
            "Corre primeiro: python Pack5/train_ml_cbf.py"
        )

    model = ml_cbf.MLCBFParameterModel.load(args.model)

    baseline = run_episode(
        params=DEFAULT_BASELINE_PARAMS,
        variation_id=args.variation_id,
        horizon=args.horizon,
        dt=args.dt,
        record_history=True,
    )
    learned = run_episode(
        params=DEFAULT_BASELINE_PARAMS,
        ml_cbf_model=model,
        variation_id=args.variation_id,
        horizon=args.horizon,
        dt=args.dt,
        record_history=True,
    )

    print_metrics("baseline", baseline)
    print_metrics("ml_cbf", learned)

    selected = Counter(learned["selected_candidates"])
    if selected:
        print("candidatos ML-CBF usados:", dict(sorted(selected.items())))

    plot_comparison(baseline, learned, args.pdf_out)
    print(f"comparacao guardada em: {args.pdf_out}")


if __name__ == "__main__":
    main()
