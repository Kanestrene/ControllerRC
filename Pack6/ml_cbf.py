from pathlib import Path

import numpy as np
from sklearn.svm import SVC


PARAM_NAMES = (
    "alpha",
    "margin",
    "lookahead_l",
    "barrier_lookahead_l",
    "Wv",
    "Ww",
    "p_slack",
    "eps_clf",
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

# Continuous search ranges for the FGO/SVM version. Wide-gain parameters use a
# log transform so the optimizer explores orders of magnitude evenly.
PARAM_BOUNDS = np.array(
    [
        [1.0, 8.0],       # alpha
        [0.02, 0.12],      # margin
        [0.01, 0.25],      # obstacle lookahead
        [0.01, 0.22],      # barrier lookahead
        [8_000.0, 120_000.0],
        [0.5, 4.0],
        [20.0, 500.0],
        [0.3, 3.0],        # eps_clf
    ],
    dtype=float,
)
PARAM_LOG_SCALE = np.array([False, False, False, False, True, False, True, False])

DEFAULT_REFERENCE_PARAMS = {
    "alpha": 3.0,
    "margin": 0.05,
    "lookahead_l": 0.20,
    "barrier_lookahead_l": 0.10,
    "Wv": 100000.0,
    "Ww": 1.0,
    "p_slack": 50.0,
    "eps_clf": 1.0,
}

# Kept as human-readable anchors, but the SVM/FGO training no longer chooses
# only from this finite list.
DEFAULT_CANDIDATES = [
    {**DEFAULT_REFERENCE_PARAMS, "alpha": 2.0, "margin": 0.035, "eps_clf": 0.6},
    {**DEFAULT_REFERENCE_PARAMS, "alpha": 5.0, "lookahead_l": 0.10, "eps_clf": 1.0},
    {**DEFAULT_REFERENCE_PARAMS, "alpha": 8.0, "margin": 0.06, "eps_clf": 1.4},
    {**DEFAULT_REFERENCE_PARAMS, "alpha": 12.0, "margin": 0.08, "eps_clf": 0.8},
]


def normalize_param_matrix(param_matrix):
    params = np.asarray(param_matrix, dtype=float)
    low = PARAM_BOUNDS[:, 0]
    high = PARAM_BOUNDS[:, 1]
    out = np.empty_like(params, dtype=float)

    linear = ~PARAM_LOG_SCALE
    out[..., linear] = (params[..., linear] - low[linear]) / (high[linear] - low[linear])

    log_low = np.log(low[PARAM_LOG_SCALE])
    log_high = np.log(high[PARAM_LOG_SCALE])
    out[..., PARAM_LOG_SCALE] = (
        np.log(params[..., PARAM_LOG_SCALE]) - log_low
    ) / (log_high - log_low)

    return np.clip(out, 0.0, 1.0)


def denormalize_param_matrix(normal_matrix):
    normal = np.asarray(normal_matrix, dtype=float)
    low = PARAM_BOUNDS[:, 0]
    high = PARAM_BOUNDS[:, 1]
    out = np.empty_like(normal, dtype=float)

    linear = ~PARAM_LOG_SCALE
    out[..., linear] = low[linear] + normal[..., linear] * (high[linear] - low[linear])

    log_low = np.log(low[PARAM_LOG_SCALE])
    log_high = np.log(high[PARAM_LOG_SCALE])
    out[..., PARAM_LOG_SCALE] = np.exp(
        log_low + normal[..., PARAM_LOG_SCALE] * (log_high - log_low)
    )

    return out


def sample_parameter_matrix(n_samples, seed=0):
    rng = np.random.default_rng(seed)
    normal = rng.random((int(n_samples), len(PARAM_NAMES)))
    return denormalize_param_matrix(normal)


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


def _poly_kernel(x, z, c1=0.8, c2=0.5, degree=7):
    return (c1 + c2 * np.asarray(x) @ np.asarray(z).T) ** degree


class PolynomialSVM:
    """Polynomial-kernel SVM backed by scikit-learn."""

    def __init__(self, degree=7, c1=0.8, c2=0.5, C=10.0):
        self.degree = int(degree)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.C = float(C)
        self.support_vectors = np.zeros((0, len(PARAM_NAMES)), dtype=float)
        self.support_weights = np.zeros((0,), dtype=float)
        self.bias = 0.0
        self.constant_label = 0.0

    def fit(self, x, y, solver_preference=("quadprog", "daqp")):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        y = np.where(y > 0, 1.0, -1.0)

        if len(np.unique(y)) < 2:
            self.constant_label = float(y[0]) if len(y) else -1.0
            return self

        svc = SVC(
            C=self.C,
            kernel="poly",
            degree=self.degree,
            gamma=self.c2,
            coef0=self.c1,
        )
        svc.fit(x, y)

        self.support_vectors = np.asarray(svc.support_vectors_, dtype=float)
        self.support_weights = np.asarray(svc.dual_coef_[0], dtype=float)
        self.bias = float(svc.intercept_[0])
        self.constant_label = 0.0
        return self

    def decision_function(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if self.constant_label != 0.0:
            return np.full(x.shape[0], self.constant_label, dtype=float)

        if len(self.support_vectors) == 0:
            return -np.ones(x.shape[0], dtype=float)

        kernel = _poly_kernel(
            x,
            self.support_vectors,
            c1=self.c1,
            c2=self.c2,
            degree=self.degree,
        )
        return kernel @ self.support_weights + self.bias


class MLCBFParameterModel:
    """
    Paper-style ML-CBF parameter learner.

    It learns a polynomial-kernel SVM feasibility boundary in continuous
    parameter space, then runs a feasibility-guided search (FGO-style) to pick
    continuous CBF/CLF-QP parameters.
    """

    def __init__(
        self,
        train_params,
        scores,
        labels,
        svm,
        best_normal_params,
        fgo_trace=None,
        score_bandwidth=0.22,
    ):
        self.train_params = np.asarray(train_params, dtype=float)
        self.train_normal_params = normalize_param_matrix(self.train_params)
        self.scores = np.asarray(scores, dtype=float)
        self.labels = np.asarray(labels, dtype=int)
        self.svm = svm
        self.best_normal_params = np.asarray(best_normal_params, dtype=float)
        self.fgo_trace = (
            np.asarray(fgo_trace, dtype=float)
            if fgo_trace is not None
            else np.zeros((0, 3), dtype=float)
        )
        self.score_bandwidth = float(score_bandwidth)

    @classmethod
    def fit(
        cls,
        param_matrix,
        scores,
        labels,
        svm_degree=7,
        svm_c1=0.8,
        svm_c2=0.5,
        svm_C=10.0,
        seed=0,
        fgo_random_samples=2500,
        fgo_iterations=10,
        fgo_population=80,
        score_bandwidth=0.22,
        solver_preference=("quadprog", "daqp"),
    ):
        param_matrix = np.asarray(param_matrix, dtype=float)
        normal_params = normalize_param_matrix(param_matrix)
        labels = np.asarray(labels, dtype=int)
        scores = np.asarray(scores, dtype=float)

        svm = PolynomialSVM(degree=svm_degree, c1=svm_c1, c2=svm_c2, C=svm_C)
        svm.fit(normal_params, labels, solver_preference=solver_preference)

        model = cls(
            train_params=param_matrix,
            scores=scores,
            labels=labels,
            svm=svm,
            best_normal_params=np.full(len(PARAM_NAMES), 0.5, dtype=float),
            score_bandwidth=score_bandwidth,
        )
        best, trace = model._fgo_search(
            seed=seed,
            random_samples=fgo_random_samples,
            iterations=fgo_iterations,
            population=fgo_population,
        )
        model.best_normal_params = best
        model.fgo_trace = trace
        return model

    @property
    def best_params_vector(self):
        return denormalize_param_matrix(self.best_normal_params.reshape(1, -1))[0]

    @property
    def best_params(self):
        return vector_to_params(self.best_params_vector)

    def _predict_score(self, normal_params):
        normal_params = np.asarray(normal_params, dtype=float)
        if normal_params.ndim == 1:
            normal_params = normal_params.reshape(1, -1)

        diff = normal_params[:, None, :] - self.train_normal_params[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        weights = np.exp(-0.5 * dist2 / max(self.score_bandwidth, 1e-6) ** 2)
        denom = np.sum(weights, axis=1)
        fallback = float(np.mean(self.scores)) if len(self.scores) else 0.0
        pred = np.full(normal_params.shape[0], fallback, dtype=float)
        good = denom > 1e-12
        pred[good] = weights[good] @ self.scores / denom[good]
        return pred

    def _objective(self, normal_params):
        score = self._predict_score(normal_params)
        feasibility = self.svm.decision_function(normal_params)
        feasible_bonus = 8.0 * np.tanh(feasibility)
        infeasible_penalty = 60.0 * np.maximum(0.0, -feasibility)
        return score + feasible_bonus - infeasible_penalty

    def _fgo_search(self, seed=0, random_samples=2500, iterations=10, population=80):
        rng = np.random.default_rng(seed)
        dim = len(PARAM_NAMES)

        observed = self.train_normal_params
        random_points = rng.random((int(random_samples), dim))
        pool = np.vstack([observed, random_points])

        values = self._objective(pool)
        best_idx = int(np.argmax(values))
        best = pool[best_idx].copy()
        trace = [[0.0, float(values[best_idx]), float(self.svm.decision_function(best)[0])]]

        elite_count = min(8, len(pool))
        elite = pool[np.argsort(values)[-elite_count:]]

        for iteration in range(1, int(iterations) + 1):
            sigma = 0.22 * (0.72 ** (iteration - 1))
            local = []
            per_elite = max(1, int(population) // len(elite))

            for center in elite:
                noise = rng.normal(0.0, sigma, size=(per_elite, dim))
                local.append(np.clip(center + noise, 0.0, 1.0))

            local = np.vstack(local)
            candidates = np.vstack([best.reshape(1, -1), elite, local])
            cand_values = self._objective(candidates)
            cand_best_idx = int(np.argmax(cand_values))

            if cand_values[cand_best_idx] > values[best_idx]:
                best = candidates[cand_best_idx].copy()
                best_idx = 0
                values = np.array([cand_values[cand_best_idx]])

            merged = np.vstack([elite, candidates])
            merged_values = self._objective(merged)
            elite = merged[np.argsort(merged_values)[-elite_count:]]

            trace.append(
                [
                    float(iteration),
                    float(self._objective(best)[0]),
                    float(self.svm.decision_function(best)[0]),
                ]
            )

        return best, np.asarray(trace, dtype=float)

    def save(self, path=DEFAULT_MODEL_PATH):
        path = Path(path)
        np.savez(
            path,
            model_version=np.array([2], dtype=int),
            train_params=self.train_params,
            train_normal_params=self.train_normal_params,
            scores=self.scores,
            labels=self.labels,
            best_normal_params=self.best_normal_params,
            best_params=self.best_params_vector,
            fgo_trace=self.fgo_trace,
            score_bandwidth=np.array([self.score_bandwidth], dtype=float),
            param_names=np.array(PARAM_NAMES),
            param_bounds=PARAM_BOUNDS,
            param_log_scale=PARAM_LOG_SCALE,
            svm_support_vectors=self.svm.support_vectors,
            svm_support_weights=self.svm.support_weights,
            svm_bias=np.array([self.svm.bias], dtype=float),
            svm_constant_label=np.array([self.svm.constant_label], dtype=float),
            svm_degree=np.array([self.svm.degree], dtype=int),
            svm_c1=np.array([self.svm.c1], dtype=float),
            svm_c2=np.array([self.svm.c2], dtype=float),
            svm_C=np.array([self.svm.C], dtype=float),
        )

    @classmethod
    def load(cls, path=DEFAULT_MODEL_PATH):
        data = np.load(Path(path), allow_pickle=False)

        if "model_version" not in data:
            # Backward compatibility for the earlier discrete-candidate model.
            candidate_matrix = data["candidate_matrix"]
            scores = data["scores"]
            candidate_ids = data["candidate_ids"]
            mean_scores = []
            for cid in range(len(candidate_matrix)):
                mask = candidate_ids == cid
                mean_scores.append(np.mean(scores[mask]) if np.any(mask) else -np.inf)
            best_id = int(np.argmax(mean_scores))
            labels = np.ones(len(candidate_matrix), dtype=int)
            svm = PolynomialSVM()
            svm.constant_label = 1.0
            return cls(
                train_params=candidate_matrix,
                scores=np.asarray(mean_scores, dtype=float),
                labels=labels,
                svm=svm,
                best_normal_params=normalize_param_matrix(candidate_matrix[best_id]),
            )

        svm = PolynomialSVM(
            degree=int(data["svm_degree"][0]),
            c1=float(data["svm_c1"][0]),
            c2=float(data["svm_c2"][0]),
            C=float(data["svm_C"][0]),
        )
        svm.support_vectors = data["svm_support_vectors"]
        svm.support_weights = data["svm_support_weights"]
        svm.bias = float(data["svm_bias"][0])
        svm.constant_label = float(data["svm_constant_label"][0])

        return cls(
            train_params=data["train_params"],
            scores=data["scores"],
            labels=data["labels"],
            svm=svm,
            best_normal_params=data["best_normal_params"],
            fgo_trace=data["fgo_trace"],
            score_bandwidth=float(data["score_bandwidth"][0]),
        )

    def select_params(
        self,
        robot_state=None,
        u_nom=None,
        obstacles=None,
        clf_info=None,
        ellipse_ab=(0.30, 0.20),
    ):
        params = self.best_params
        params["_ml_cbf_candidate"] = -1
        params["_ml_cbf_score"] = float(self._predict_score(self.best_normal_params)[0])
        params["_ml_cbf_feasibility"] = float(
            self.svm.decision_function(self.best_normal_params)[0]
        )
        return params


def load_model_if_available(path=DEFAULT_MODEL_PATH):
    path = Path(path)
    if not path.exists():
        return None
    return MLCBFParameterModel.load(path)
