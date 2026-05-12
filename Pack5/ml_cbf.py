from pathlib import Path

import numpy as np


PARAM_NAMES = (
    "alpha",
    "margin",
    "lookahead_l",
    "barrier_lookahead_l",
    "Wv",
    "Ww",
    "p_slack",
)

FEATURE_NAMES = (
    "obs_clearance",
    "abs_ey",
    "abs_epsi",
    "abs_kappa",
    "abs_v_nom",
    "abs_w_nom",
    "obstacle_count",
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "ml_cbf_model.npz"


DEFAULT_CANDIDATES = [
    {
        "alpha": 2.0,
        "margin": 0.035,
        "lookahead_l": 0.08,
        "barrier_lookahead_l": 0.08,
        "Wv": 50000.0,
        "Ww": 1.0,
        "p_slack": 50.0,
    },
    {
        "alpha": 5.0,
        "margin": 0.05,
        "lookahead_l": 0.10,
        "barrier_lookahead_l": 0.10,
        "Wv": 100000.0,
        "Ww": 1.0,
        "p_slack": 50.0,
    },
    {
        "alpha": 8.0,
        "margin": 0.06,
        "lookahead_l": 0.12,
        "barrier_lookahead_l": 0.12,
        "Wv": 50000.0,
        "Ww": 1.0,
        "p_slack": 100.0,
    },
    {
        "alpha": 12.0,
        "margin": 0.08,
        "lookahead_l": 0.16,
        "barrier_lookahead_l": 0.14,
        "Wv": 30000.0,
        "Ww": 1.0,
        "p_slack": 150.0,
    },
    {
        "alpha": 4.0,
        "margin": 0.04,
        "lookahead_l": 0.14,
        "barrier_lookahead_l": 0.10,
        "Wv": 30000.0,
        "Ww": 2.0,
        "p_slack": 75.0,
    },
    {
        "alpha": 7.0,
        "margin": 0.05,
        "lookahead_l": 0.18,
        "barrier_lookahead_l": 0.12,
        "Wv": 20000.0,
        "Ww": 2.0,
        "p_slack": 100.0,
    },
    {
        "alpha": 10.0,
        "margin": 0.07,
        "lookahead_l": 0.10,
        "barrier_lookahead_l": 0.16,
        "Wv": 15000.0,
        "Ww": 1.5,
        "p_slack": 200.0,
    },
    {
        "alpha": 3.0,
        "margin": 0.06,
        "lookahead_l": 0.08,
        "barrier_lookahead_l": 0.12,
        "Wv": 100000.0,
        "Ww": 1.0,
        "p_slack": 100.0,
    },
    {
        "alpha": 6.0,
        "margin": 0.03,
        "lookahead_l": 0.16,
        "barrier_lookahead_l": 0.08,
        "Wv": 50000.0,
        "Ww": 3.0,
        "p_slack": 50.0,
    },
    {
        "alpha": 9.0,
        "margin": 0.05,
        "lookahead_l": 0.12,
        "barrier_lookahead_l": 0.18,
        "Wv": 20000.0,
        "Ww": 2.0,
        "p_slack": 250.0,
    },
    {
        "alpha": 14.0,
        "margin": 0.10,
        "lookahead_l": 0.18,
        "barrier_lookahead_l": 0.18,
        "Wv": 10000.0,
        "Ww": 1.0,
        "p_slack": 300.0,
    },
    {
        "alpha": 5.0,
        "margin": 0.08,
        "lookahead_l": 0.10,
        "barrier_lookahead_l": 0.14,
        "Wv": 75000.0,
        "Ww": 1.0,
        "p_slack": 200.0,
    },
]


def candidates_to_matrix(candidates=None):
    candidates = DEFAULT_CANDIDATES if candidates is None else candidates
    return np.array(
        [[float(candidate[name]) for name in PARAM_NAMES] for candidate in candidates],
        dtype=float,
    )


def vector_to_params(vector):
    return {name: float(value) for name, value in zip(PARAM_NAMES, vector)}


def make_context_features(
    robot_state,
    u_nom,
    obstacles,
    clf_info=None,
    ellipse_ab=(0.30, 0.20),
):
    x, y, _ = robot_state
    v_nom, w_nom = u_nom
    clf_info = clf_info or {}

    robot_radius = max(float(ellipse_ab[0]), float(ellipse_ab[1]))
    clearances = []

    for obs in obstacles:
        ox = float(obs["x"])
        oy = float(obs["y"])
        radius = float(obs.get("r", 0.0))
        clearances.append(np.hypot(x - ox, y - oy) - radius - robot_radius)

    obs_clearance = min(clearances) if clearances else 10.0

    features = np.array(
        [
            np.clip(obs_clearance, -2.0, 10.0),
            abs(float(clf_info.get("ey", 0.0))),
            abs(float(clf_info.get("epsi", 0.0))),
            abs(float(clf_info.get("kappa_r", 0.0))),
            abs(float(v_nom)),
            abs(float(w_nom)),
            min(float(len(obstacles)), 10.0),
        ],
        dtype=float,
    )

    return np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-2.0)


class MLCBFParameterModel:
    """
    Dependency-free learner for CBF parameter selection.

    The model stores simulation samples and predicts the score of each candidate
    parameter set with a local weighted average in the current context.
    """

    def __init__(
        self,
        candidate_matrix,
        contexts,
        candidate_ids,
        scores,
        context_scale=None,
        k_neighbors=20,
    ):
        self.candidate_matrix = np.asarray(candidate_matrix, dtype=float)
        self.contexts = np.asarray(contexts, dtype=float)
        self.candidate_ids = np.asarray(candidate_ids, dtype=int)
        self.scores = np.asarray(scores, dtype=float)
        self.k_neighbors = int(k_neighbors)

        if context_scale is None:
            if self.contexts.size == 0:
                context_scale = np.ones(len(FEATURE_NAMES), dtype=float)
            else:
                context_scale = np.std(self.contexts, axis=0)

        self.context_scale = np.maximum(np.asarray(context_scale, dtype=float), 1e-3)

    @classmethod
    def fit(cls, contexts, candidate_ids, scores, candidates=None, k_neighbors=20):
        return cls(
            candidate_matrix=candidates_to_matrix(candidates),
            contexts=contexts,
            candidate_ids=candidate_ids,
            scores=scores,
            k_neighbors=k_neighbors,
        )

    def save(self, path=DEFAULT_MODEL_PATH):
        path = Path(path)
        np.savez(
            path,
            candidate_matrix=self.candidate_matrix,
            contexts=self.contexts,
            candidate_ids=self.candidate_ids,
            scores=self.scores,
            context_scale=self.context_scale,
            k_neighbors=np.array([self.k_neighbors], dtype=int),
            param_names=np.array(PARAM_NAMES),
            feature_names=np.array(FEATURE_NAMES),
        )

    @classmethod
    def load(cls, path=DEFAULT_MODEL_PATH):
        data = np.load(Path(path), allow_pickle=False)
        k_neighbors = int(data["k_neighbors"][0]) if "k_neighbors" in data else 20
        return cls(
            candidate_matrix=data["candidate_matrix"],
            contexts=data["contexts"],
            candidate_ids=data["candidate_ids"],
            scores=data["scores"],
            context_scale=data["context_scale"],
            k_neighbors=k_neighbors,
        )

    def _predict_candidate_score(self, context, candidate_id):
        mask = self.candidate_ids == int(candidate_id)
        if not np.any(mask):
            return -np.inf

        candidate_contexts = self.contexts[mask]
        candidate_scores = self.scores[mask]
        diff = (candidate_contexts - context) / self.context_scale
        dist = np.linalg.norm(diff, axis=1)

        k = min(self.k_neighbors, len(dist))
        idx = np.argpartition(dist, k - 1)[:k] if k > 0 else np.arange(len(dist))
        local_dist = dist[idx]
        local_scores = candidate_scores[idx]

        weights = np.exp(-0.5 * local_dist * local_dist)
        weight_sum = float(np.sum(weights))
        if weight_sum < 1e-9:
            return float(np.mean(local_scores))

        return float(np.dot(weights, local_scores) / weight_sum)

    def select_params(
        self,
        robot_state,
        u_nom,
        obstacles,
        clf_info=None,
        ellipse_ab=(0.30, 0.20),
    ):
        context = make_context_features(
            robot_state=robot_state,
            u_nom=u_nom,
            obstacles=obstacles,
            clf_info=clf_info,
            ellipse_ab=ellipse_ab,
        )

        predicted_scores = np.array(
            [
                self._predict_candidate_score(context, candidate_id)
                for candidate_id in range(len(self.candidate_matrix))
            ],
            dtype=float,
        )

        best_id = int(np.argmax(predicted_scores))
        params = vector_to_params(self.candidate_matrix[best_id])
        params["_ml_cbf_candidate"] = best_id
        params["_ml_cbf_score"] = float(predicted_scores[best_id])
        return params


def load_model_if_available(path=DEFAULT_MODEL_PATH):
    path = Path(path)
    if not path.exists():
        return None
    return MLCBFParameterModel.load(path)
