import numpy as np
from pathlib import Path
from qpsolvers import solve_qp
import cbf 
import clf


BASE_DIR = Path(__file__).resolve().parent
_BARRIER_CACHE = None


def _load_barriers():
    global _BARRIER_CACHE
    if _BARRIER_CACHE is None:
        inner = np.loadtxt(BASE_DIR / "barreira_suavizada_interna.txt")
        outer = np.loadtxt(BASE_DIR / "barreira_suavizada_externa.txt")
        _BARRIER_CACHE = inner, outer
    return _BARRIER_CACHE


def _apply_ml_cbf_params(
    ml_cbf_selector,
    robot_state,
    u_nom,
    obstacles,
    clf_info,
    ellipse_ab,
    alpha,
    margin,
    lookahead_l,
    barrier_lookahead_l,
    W,
    p_slack,
    eps_clf,
):
    selected = {}

    if ml_cbf_selector is not None:
        selected = ml_cbf_selector.select_params(
            robot_state=robot_state,
            u_nom=u_nom,
            obstacles=obstacles,
            clf_info=clf_info,
            ellipse_ab=ellipse_ab,
        )

    alpha = float(selected.get("alpha", alpha))
    margin = float(selected.get("margin", margin))
    lookahead_l = float(selected.get("lookahead_l", lookahead_l))
    barrier_lookahead_l = float(selected.get("barrier_lookahead_l", barrier_lookahead_l))
    W = (
        float(selected.get("Wv", W[0])),
        float(selected.get("Ww", W[1])),
    )
    p_slack = float(selected.get("p_slack", p_slack))
    eps_clf = float(selected.get("eps_clf", eps_clf))

    active = {
        "alpha": alpha,
        "margin": margin,
        "lookahead_l": lookahead_l,
        "barrier_lookahead_l": barrier_lookahead_l,
        "W": W,
        "p_slack": p_slack,
        "eps_clf": eps_clf,
    }

    if selected:
        active["ml_cbf_candidate"] = int(selected.get("_ml_cbf_candidate", -1))
        active["ml_cbf_score"] = float(selected.get("_ml_cbf_score", np.nan))
        active["ml_cbf_feasibility"] = float(
            selected.get("_ml_cbf_feasibility", np.nan)
        )

    return alpha, margin, lookahead_l, barrier_lookahead_l, W, p_slack, eps_clf, active


def cbf_qp_filter(u_nom, robot_state, obstacles,
                  ellipse_ab=(0.30, 0.20),
                  margin=0.05, lookahead_l=0.35, alpha=2.0,
                  barrier_lookahead_l=0.1,
                  W=(20.0, 1.0),
                  v_bounds=(0.0, 1.5), w_bounds=(-2.5, 2.5),
                  solver_preference=("quadprog", "daqp")):
    """
    Resolve:
      min (u-u_nom)^T W (u-u_nom)
      s.t. G u <= h   (CBF + bounds)
    u = [v, w]
    """
    v_nom, w_nom = u_nom
    x, y, th = robot_state

    # custo: (u-u_nom)^T W (u-u_nom)  -> 1/2 u^T P u + q^T u
    Wv, Ww = W
    P = 2.0 * np.diag([Wv, Ww])
    q = -2.0 * np.array([Wv * v_nom, Ww * w_nom], dtype=float)

    # CBF constraints
    G_obs, h_obs = cbf.cbf_rows_for_circle_obstacles(
        x, y, th, obstacles,
        ellipse_ab=ellipse_ab, margin=margin,
        lookahead_l=lookahead_l, alpha=alpha
    )

    inner, outer = _load_barriers()
    
    G_barrier, h_barrier = cbf.cbf_rows_for_barriers(
        x, y, th,
        barrier_inner=inner,
        barrier_outer=outer,
        ellipse_ab=ellipse_ab,
        margin=margin,
        lookahead_l=barrier_lookahead_l,
        alpha=alpha,
        max_segments=10
    )
    
    # bounds (box) em G u <= h
    vmin, vmax = v_bounds
    wmin, wmax = w_bounds
    G_box = np.array([
        [ 1.0,  0.0],   #  v <= vmax
        [-1.0,  0.0],   # -v <= -vmin  -> v >= vmin
        [ 0.0,  1.0],   #  w <= wmax
        [ 0.0, -1.0],   # -w <= -wmin  -> w >= wmin
    ])
    h_box = np.array([vmax, -vmin, wmax, -wmin], dtype=float)

    # juntar tudo
    G = np.vstack([G_obs, G_barrier, G_box])
    h = np.concatenate([h_obs, h_barrier, h_box])

    # resolver QP
    u = None
    for s in solver_preference:
        try:
            u = solve_qp(P, q, G, h, solver=s)
            if u is not None:
                break
        except Exception:
            pass

    # fallback se falhar
    if u is None or np.any(np.isnan(u)):
        return np.array([vmin, 0.0], dtype=float)

    return u

def cbf_clf_qp_filter(
    u_nom,
    robot_state,
    obstacles,
    px, py, pyaw, s,
    last_path_idx=0,
    ellipse_ab=(0.30, 0.20),
    margin=0.05,
    lookahead_l=0.35,
    alpha=2.0,               # CBF
    eps_clf=1.0,             # CLF
    q_clf=(1.0, 4.0, 2.0),   # qx, qy, qpsi
    W=(20.0, 1.0),           # pesos para v,w
    p_slack=1000.0,          # peso da slack delta
    v_ref=1.5,
    v_bounds=(0.0, 1.5),
    w_bounds=(-2.5, 2.5),
    barrier_lookahead_l=0.1,
    ml_cbf_selector=None,
    solver_preference=("quadprog", "daqp")
):
    """
    Resolve:
      min (u-u_nom)^T W (u-u_nom) + p_slack * delta^2

    sujeito a:
      CBFs
      CLF: Vdot + eps_clf V <= delta
      bounds em v,w
      delta >= 0
    """
    v_nom, w_nom = u_nom
    x, y, th = robot_state

    # ----------------------------------
    # CLF preliminar para dar contexto ao seletor ML-CBF
    # ----------------------------------
    _, _, clf_info = clf.clf_row_path_tracking(
        px, py, pyaw, s,
        robot_state=(x, y, th),
        last_idx=last_path_idx,
        v_ref=v_ref,
        qx=q_clf[0],
        qy=q_clf[1],
        qpsi=q_clf[2],
        eps=eps_clf
    )

    (
        alpha,
        margin,
        lookahead_l,
        barrier_lookahead_l,
        W,
        p_slack,
        eps_clf,
        active_params,
    ) = _apply_ml_cbf_params(
        ml_cbf_selector=ml_cbf_selector,
        robot_state=(x, y, th),
        u_nom=(v_nom, w_nom),
        obstacles=obstacles,
        clf_info=clf_info,
        ellipse_ab=ellipse_ab,
        alpha=alpha,
        margin=margin,
        lookahead_l=lookahead_l,
        barrier_lookahead_l=barrier_lookahead_l,
        W=W,
        p_slack=p_slack,
        eps_clf=eps_clf,
    )

    # ----------------------------------
    # CLF final com o eps_clf selecionado
    # ----------------------------------
    G_clf, h_clf, clf_info = clf.clf_row_path_tracking(
        px, py, pyaw, s,
        robot_state=(x, y, th),
        last_idx=last_path_idx,
        v_ref=v_ref,
        qx=q_clf[0],
        qy=q_clf[1],
        qpsi=q_clf[2],
        eps=eps_clf
    )
    clf_info["active_cbf_params"] = active_params

    # ----------------------------------
    # custo em z = [v, w, delta]
    # ----------------------------------
    Wv, Ww = W
    P = 2.0 * np.diag([Wv, Ww, p_slack])
    q = -2.0 * np.array([Wv * v_nom, Ww * w_nom, 0.0], dtype=float)

    # ----------------------------------
    # CBF obstáculos circulares
    # retorna G u <= h com u=[v,w]
    # ----------------------------------
    G_obs_2, h_obs = cbf.cbf_rows_for_circle_obstacles(
        x, y, th, obstacles,
        ellipse_ab=ellipse_ab,
        margin=margin,
        lookahead_l=lookahead_l,
        alpha=alpha
    )

    if G_obs_2.size == 0:
        G_obs = np.zeros((0, 3))
        h_obs = np.zeros((0,))
    else:
        G_obs = np.hstack([G_obs_2, np.zeros((G_obs_2.shape[0], 1))])

    # ----------------------------------
    # CBF barreiras
    # ----------------------------------
    inner, outer = _load_barriers()

    G_bar_2, h_bar = cbf.cbf_rows_for_barriers(
        x, y, th,
        barrier_inner=inner,
        barrier_outer=outer,
        ellipse_ab=ellipse_ab,
        margin=margin,
        lookahead_l=barrier_lookahead_l,
        alpha=alpha,
        max_segments=10
    )

    if G_bar_2.size == 0:
        G_bar = np.zeros((0, 3))
        h_bar = np.zeros((0,))
    else:
        G_bar = np.hstack([G_bar_2, np.zeros((G_bar_2.shape[0], 1))])

    # ----------------------------------
    # bounds
    # ----------------------------------
    vmin, vmax = v_bounds
    wmin, wmax = w_bounds

    G_box = np.array([
        [ 1.0,  0.0,  0.0],   # v <= vmax
        [-1.0,  0.0,  0.0],   # v >= vmin
        [ 0.0,  1.0,  0.0],   # w <= wmax
        [ 0.0, -1.0,  0.0],   # w >= wmin
        [ 0.0,  0.0, -1.0],   # delta >= 0
    ], dtype=float)

    h_box = np.array([
        vmax,
        -vmin,
        wmax,
        -wmin,
        0.0
    ], dtype=float)

    # ----------------------------------
    # juntar tudo
    # ----------------------------------
    G = np.vstack([G_obs, G_bar, G_clf, G_box])
    h = np.concatenate([h_obs, h_bar, h_clf, h_box])
    #G = np.vstack([G_obs, G_clf, G_box])
    #h = np.concatenate([h_obs, h_clf, h_box])

    z = None
    for sname in solver_preference:
        try:
            z = solve_qp(P, q, G, h, solver=sname)
            if z is not None:
                break
        except Exception:
            pass

    if z is None or np.any(np.isnan(z)):
        clf_info["qp_failed"] = True
        clf_info["solver"] = None
        clf_info["slack"] = np.nan
        return np.array([vmin, 0.0], dtype=float), clf_info

    clf_info["qp_failed"] = False
    clf_info["solver"] = sname
    clf_info["slack"] = float(z[2])
    return z[:2], clf_info
