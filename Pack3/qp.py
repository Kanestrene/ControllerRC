import numpy as np
from qpsolvers import solve_qp
import cbf


def cbf_qp_filter_acc(
    u_nom, robot_state, obstacles,
    ellipse_ab=(0.30, 0.20),
    margin=0.05, lookahead_l=0.35,
    lambda1=2.0, lambda2=2.0,
    W=(20.0, 1.0),
    a_bounds=(-1.0, 1.5), alpha_bounds=(-2.5, 2.5),
    solver_preference=("quadprog", "daqp"),
):
    """
    Resolve:
      min (u-u_nom)^T W (u-u_nom)
      s.t. G u <= h   (CBF + bounds)

    u = [a, alpha]
    """
    a_nom, alpha_nom = u_nom
    x, y, th, v, w = robot_state

    # custo: (u-u_nom)^T W (u-u_nom) = 1/2 u^T P u + q^T u
    Wa, Walpha = W
    P = 2.0 * np.diag([Wa, Walpha])
    q = -2.0 * np.array([Wa * a_nom, Walpha * alpha_nom], dtype=float)

    # CBF obstáculos circulares (em aceleração)
    G_obs, h_obs = cbf.cbf_rows_for_circle_obstacles_acc(
        x, y, th, v, w, obstacles,
        ellipse_ab=ellipse_ab,
        margin=margin,
        lookahead_l=lookahead_l,
        lambda1=lambda1,
        lambda2=lambda2,
    )

    # Barreiras laterais / pista (em aceleração)
    inner = np.loadtxt("barreira_suavizada_interna.txt")
    outer = np.loadtxt("barreira_suavizada_externa.txt")

    G_barrier, h_barrier = cbf.cbf_rows_for_barriers_acc(
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

    # bounds em aceleração: u = [a, alpha]
    amin, amax = a_bounds
    alphamin, alphamax = alpha_bounds

    G_box = np.array([
        [ 1.0,  0.0],   #  a <= amax
        [-1.0,  0.0],   # -a <= -amin  -> a >= amin
        [ 0.0,  1.0],   #  alpha <= alphamax
        [ 0.0, -1.0],   # -alpha <= -alphamin -> alpha >= alphamin
    ], dtype=float)

    h_box = np.array([amax, -amin, alphamax, -alphamin], dtype=float)

    # juntar todas as restrições existentes
    G_list = []
    h_list = []

    if G_obs is not None and G_obs.size > 0:
        G_list.append(G_obs)
        h_list.append(h_obs)

    if G_barrier is not None and G_barrier.size > 0:
        G_list.append(G_barrier)
        h_list.append(h_barrier)

    G_list.append(G_box)
    h_list.append(h_box)

    G = np.vstack(G_list)
    h = np.concatenate(h_list)

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
        return np.array([0.0, 0.0], dtype=float)

    return u