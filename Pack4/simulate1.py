import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import qp

from controller import (
    wrap_to_pi,
    build_spline_path,
    omega_to_delta,
    rate_limit,
)


def simulate():
    waypoints = [
        (3.0, 3.0),
        (2.6, 3.5),
        (2.2, 4.2),
        (2.0, 5.0),
        (2.0, 6.2),
        (2.0, 7.4),
        (2.0, 8.8),
        (2.0, 10.2),
        (2.0, 11.6),
        (2.0, 13.0),
        (2.2, 13.6),
        (2.6, 14.5),
        (3.1, 14.8),
        (3.7, 15.0),
        (4.2, 15.0),
        (4.8, 14.9),
        (5.3, 14.6),
        (5.6, 14.1),
        (5.7, 13.5),
        (5.6, 12.6),
        (5.5, 11.6),
        (5.5, 10.6),
        (5.5, 9.8),
        (5.7, 9.2),
        (6.0, 8.7),
        (6.6, 8.4),
        (7.4, 8.4),
        (8.2, 8.5),
        (8.8, 8.9),
        (9.1, 9.6),
        (9.3, 10.4),
        (9.5, 11.6),
        (9.7, 12.6),
        (9.9, 13.4),
        (10.2, 14.0),
        (10.8, 14.6),
        (11.6, 15.0),
        (12.6, 15.0),
        (13.6, 15.0),
        (14.6, 14.8),
        (15.4, 14.4),
        (16.0, 13.6),
        (16.0, 12.4),
        (16.0, 11.2),
        (16.0, 10.0),
        (16.0, 8.8),
        (16.0, 7.6),
        (16.0, 6.4),
        (16.0, 5.0),
        (15.5, 3.6),
        (14.2, 3.3),
        (12.0, 3.3),
        (10.8, 3.5),
        (9.6, 3.5),
        (8.4, 3.5),
        (7.2, 3.3),
        (6.0, 3.3),
        (4.0, 2.5),
        (3.0, 3.0),
    ]

    px, py, pyaw, s = build_spline_path(waypoints, ds=0.01)

    n_obs = 5
    idxs = np.linspace(0, len(px) - 1, n_obs + 2, dtype=int)[1:-1]

    obstacles = []
    
    for k, idx in enumerate(idxs[:-1]):
        x_path = px[idx]
        y_path = py[idx]
        yaw_path = pyaw[idx]

        nx = -np.sin(yaw_path)
        ny = np.cos(yaw_path)

        side = (-1) ** k
        offset = 0.3

        ox = x_path + side * offset * nx
        oy = y_path + side * offset * ny

        obstacles.append({
            "x": ox,
            "y": oy,
            "r": 0.35
        })
    
    # Estado inicial
    x, y, yaw, v, w = 2, 6, np.deg2rad(90), 0.0, 0.0

    dt = 0.02
    T = 60.0
    steps = int(T / dt)

    v_ref = 2.0

    L0 = 0.1
    kv = 0.5

    w_max = 2.5
    a_max = 1.0

    a_ell, b_ell = 0.35, 0.25
    margin = 0.01

    last_near = 0
    hx, hy, ctes = [], [], []

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 5))

    delta = 0.0

    L = 0.26
    delta_max = np.deg2rad(25)
    delta_rate_max = np.deg2rad(300)

    # carregar barreiras uma vez só
    inner_bar = np.loadtxt("barreira_suavizada_interna.txt")
    outer_bar = np.loadtxt("barreira_suavizada_externa.txt")

    inner_x, inner_y = inner_bar[:, 0], inner_bar[:, 1]
    outer_x, outer_y = outer_bar[:, 0], outer_bar[:, 1]

    for k in range(steps):
        Ld = L0 + kv * abs(v)

        # nominal simples: velocidade desejada + amortecimento angular
        a_cmd = 1.5 * (v_ref - v)
        alpha_cmd = -2.0 * w

        u_safe, clf_info = qp.cbf_qp_filter_acc(
            u_nom=(a_cmd, alpha_cmd),
            robot_state=(x, y, yaw, v, w),
            obstacles=obstacles,
            px=px, py=py, pyaw=pyaw, s=s,
            last_idx=last_near,
            v_ref=v_ref,
            ellipse_ab=(a_ell, b_ell),
            margin=0.01,
            lookahead_l=0.6,
            lambda1=3.0,
            lambda2=3.0,
            W=(25000.0, 1.0),
            p_slack=500.0, 
            a_bounds=(-a_max, a_max),
            alpha_bounds=(-4.0, 4.0),
        )

        a_safe, alpha_safe = u_safe

        last_near = clf_info["idx"]
        cte = clf_info["ey"]
        target_idx = clf_info["idx"]

        v += a_safe * dt
        w += alpha_safe * dt

        v = np.clip(v, 0.0, 2.0)

        kappa_max = np.tan(delta_max) / L
        w_max_speed = abs(v) * kappa_max
        w = np.clip(w, -w_max_speed, w_max_speed)

        delta_cmd = omega_to_delta(w, v, L, v_min=0.2)
        delta_cmd = np.clip(delta_cmd, -delta_max, delta_max)

        delta = rate_limit(delta_cmd, delta, du_max=delta_rate_max * dt)

        w = (v / L) * np.tan(delta)

        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt
        yaw = wrap_to_pi(yaw + w * dt)

        hx.append(x)
        hy.append(y)
        ctes.append(cte)

        if k % 5 == 0:
            ax.clear()

            ax.plot(px, py, "--", label="Spline (referência)")
            ax.plot(hx, hy, "-", label="Trajetória robô (HOCLF + CBF)")

            ax.plot(inner_x, inner_y, "-", linewidth=2, label="Barreira interna")
            ax.plot(outer_x, outer_y, "-", linewidth=2, label="Barreira externa")

            for obs in obstacles:
                ax.add_patch(Circle((obs["x"], obs["y"]), obs["r"], fill=False))

            ell = Ellipse((x, y), width=2 * a_ell, height=2 * b_ell,
                          angle=np.degrees(yaw), fill=False)
            ax.add_patch(ell)

            ell_safe = Ellipse((x, y), width=2 * (a_ell + margin), height=2 * (b_ell + margin),
                               angle=np.degrees(yaw), fill=False)
            ax.add_patch(ell_safe)

            ax.plot(x, y, "o", label="Robô")
            ax.arrow(x, y, 0.4 * np.cos(yaw), 0.4 * np.sin(yaw), head_width=0.15)

            ax.plot(px[target_idx], py[target_idx], "x", markersize=10, label="Ref HOCLF")
            ax.add_patch(Circle((x, y), Ld, fill=False))

            ax.set_aspect("equal", "box")
            ax.grid(True)
            ax.set_title(
                f"HOCLF + CBF-QP(acc→bicycle) | "
                f"v={v:.2f} | w={w:.2f} | a={a_safe:.2f} | alpha={alpha_safe:.2f} | ey={cte:.3f}"
            )
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
            plt.pause(0.001)

    plt.ioff()

    fig2, ax2 = plt.subplots()
    ax2.plot(ctes)
    ax2.set_title("Erro lateral ey - HOCLF")
    ax2.set_xlabel("Passo")
    ax2.set_ylabel("ey [m]")
    ax2.grid(True)
    plt.show()


if __name__ == "__main__":
    simulate()