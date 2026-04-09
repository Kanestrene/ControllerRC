import numpy as np
from qpsolvers import solve_qp
import cbf
import clf


def cbf_qp_filter_acc(
    u_nom, robot_state, obstacles,
    px, py, pyaw, s, last_idx,
    v_ref=1.0,
    ellipse_ab=(0.30, 0.20),
    margin=0.05, lookahead_l=0.35,
    lambda1=2.0, lambda2=2.0,
    W=(20.0, 1.0),          # pesos de a, alpha
    p_slack=1000.0,         # peso do slack delta
    a_bounds=(-1.0, 1.5), alpha_bounds=(-2.5, 2.5),
    solver_preference=("quadprog", "daqp"),
):
    """
    Resolve:
      min (u-u_nom)^T W (u-u_nom) + p_slack * delta^2

    sujeito a:
      HOCLF + CBF + bounds + delta >= 0

    variável de decisão:
      z = [a, alpha, delta]

    retorno:
      [a, alpha], clf_info
    """
    a_nom, alpha_nom = u_nom
    x, y, th, v, w = robot_state

    # ==================================
    # custo em z = [a, alpha, delta]
    # ==================================
    Wa, Walpha = W
    P = 2.0 * np.diag([Wa, Walpha, p_slack])
    q = -2.0 * np.array([Wa * a_nom, Walpha * alpha_nom, 0.0], dtype=float)

    # =========================
    # HOCLF de tracking
    # =========================
    G_clf, h_clf, clf_info = clf.hoclf_row_path_tracking_acc(
        px, py, pyaw, s,
        robot_state=robot_state,
        last_idx=last_idx,
        v_ref=v_ref,
        qx=1.0,
        qy=4.0,
        qpsi=1.0,
        lambda1=1.5,
        lambda2=1.5,
        lookahead_l=0.6,
        with_slack=True,
    )

    # =========================
    # CBF obstáculos circulares
    # retorna restrições em [a, alpha]
    # precisamos expandir para [a, alpha, delta]
    # =========================
    G_obs_2, h_obs = cbf.cbf_rows_for_circle_obstacles_acc(
        x, y, th, v, w, obstacles,
        ellipse_ab=ellipse_ab,
        margin=margin,
        lookahead_l=lookahead_l,
        lambda1=lambda1,
        lambda2=lambda2,
    )

    if G_obs_2 is None or G_obs_2.size == 0:
        G_obs = np.zeros((0, 3))
        h_obs = np.zeros((0,), dtype=float)
    else:
        G_obs = np.hstack([G_obs_2, np.zeros((G_obs_2.shape[0], 1))])

    # =========================
    # CBF barreiras laterais / pista
    # também expandir para [a, alpha, delta]
    # =========================
    inner = np.loadtxt("barreira_suavizada_interna.txt")
    outer = np.loadtxt("barreira_suavizada_externa.txt")

    G_barrier_2, h_barrier = cbf.cbf_rows_for_barriers_acc(
        x, y, th, v, w,
        barrier_inner=inner,
        barrier_outer=outer,
        ellipse_ab=ellipse_ab,
        margin=0.25,
        lookahead_l=0.2,
        lambda1=5,
        lambda2=5,
        max_segments=10,
    )

    if G_barrier_2 is None or G_barrier_2.size == 0:
        G_barrier = np.zeros((0, 3))
        h_barrier = np.zeros((0,), dtype=float)
    else:
        G_barrier = np.hstack([G_barrier_2, np.zeros((G_barrier_2.shape[0], 1))])

    # =========================
    # bounds em aceleração + delta >= 0
    # =========================
    amin, amax = a_bounds
    alphamin, alphamax = alpha_bounds

    G_box = np.array([
        [ 1.0,  0.0,  0.0],   # a <= amax
        [-1.0,  0.0,  0.0],   # a >= amin
        [ 0.0,  1.0,  0.0],   # alpha <= alphamax
        [ 0.0, -1.0,  0.0],   # alpha >= alphamin
        [ 0.0,  0.0, -1.0],   # delta >= 0
    ], dtype=float)

    h_box = np.array([amax, -amin, alphamax, -alphamin, 0.0], dtype=float)

    # =========================
    # juntar restrições
    # =========================
    G = np.vstack([G_clf, G_obs, G_barrier, G_box])
    h = np.concatenate([h_clf, h_obs, h_barrier, h_box])

    # =========================
    # resolver QP
    # =========================
    z = None
    for sname in solver_preference:
        try:
            z = solve_qp(P, q, G, h, solver=sname)
            if z is not None:
                break
        except Exception:
            pass

    # fallback se falhar
    if z is None or np.any(np.isnan(z)):
        clf_info["delta"] = np.nan
        return np.array([0.0, 0.0], dtype=float), clf_info

    clf_info["delta"] = z[2]

    return z[:2], clf_info