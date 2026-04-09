import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Arc
from scipy.interpolate import splprep, splev


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    # =========================================================
    # 1. Trajetória spline
    # =========================================================
    ctrl = np.array([
        [0.2, 0.5],
        [0.9, 0.9],
        [1.8, 1.2],
        [2.8, 1.0],
        [3.6, 0.6],
        [4.4, 0.8],
        [5.0, 1.3]
    ])

    tck, _ = splprep([ctrl[:, 0], ctrl[:, 1]], s=0.0, k=3)
    uu = np.linspace(0, 1, 500)
    px, py = splev(uu, tck)

    dxdu, dydu = splev(uu, tck, der=1)
    pyaw = np.arctan2(dydu, dxdu)

    # =========================================================
    # 2. Escolher ponto de referência na trajetória
    # =========================================================
    idx = 220
    xr, yr = px[idx], py[idx]
    psi_r = pyaw[idx]

    # Base local da trajetória
    t_hat = np.array([np.cos(psi_r), np.sin(psi_r)])
    n_hat = np.array([-np.sin(psi_r), np.cos(psi_r)])

    # =========================================================
    # 3. Colocar o robô ao lado do ponto de referência
    # =========================================================
    ex = 0.20
    ey = 0.45
    epsi = np.deg2rad(20)

    robot_pos = np.array([xr, yr]) + ex * t_hat + ey * n_hat
    x, y = robot_pos
    theta = psi_r + epsi

    # dimensões do robô
    a_car = 0.22
    b_car = 0.12

    # ponto intermédio para decompor erro
    p_ex = np.array([xr, yr]) + ex * t_hat

    # =========================================================
    # 4. Plot
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    # trajetória
    ax.plot(px, py, color="black", linewidth=2.5)
    #ax.plot(ctrl[:, 0], ctrl[:, 1], "o", color="gray", alpha=0.35, markersize=5)

    # ponto de referência
    #ax.plot(xr, yr, "bo", markersize=7)
    #ax.text(xr + 0.05, yr - 0.10, "referência", color="blue", fontsize=11)

            

    # robô
    ax.add_patch(Ellipse(
        (x, y),
        width=2 * a_car,
        height=2 * b_car,
        angle=np.degrees(theta),
        fill=False,
        linewidth=2.5,
        edgecolor="black"
    ))

    ax.plot(x, y, "ko", markersize=5)
    #ax.text(x + 0.05, y + 0.05, "robô", fontsize=11)

    # orientação do robô
    car_scale = 0.35
    ax.arrow(
        x, y,
        car_scale * np.cos(theta), car_scale * np.sin(theta),
        head_width=0.04, head_length=0.06,
        fc="black", ec="black", linewidth=2, length_includes_head=True
    )

    # orientação de referência transportada para o robô
    ref_scale = 0.28
    ax.arrow(
        x, y,
        ref_scale * np.cos(psi_r), ref_scale * np.sin(psi_r),
        head_width=0.035, head_length=0.05,
        fc="gray", ec="gray", linewidth=1.8, alpha=0.8, length_includes_head=True
    )

    # vetor erro global
    ax.plot([xr, x], [yr, y], "--", color="gray", linewidth=1.6)

    # erro ex tracejado
    ax.plot(
        [xr, p_ex[0]], [yr, p_ex[1]],
        color="red", linewidth=2.5, linestyle="--"
    )
    ax.plot(p_ex[0], p_ex[1], "ro", markersize=5)
    ax.text(
        0.5 * (xr + p_ex[0]) - 0.0,
        0.5 * (yr + p_ex[1]) - 0.1,
        r"$e_x$",
        color="red",
        fontsize=13
    )

    # erro ey tracejado
    ax.plot(
        [p_ex[0], x], [p_ex[1], y],
        color="blue", linewidth=2.5, linestyle="--"
    )
    ax.text(
        0.5 * (p_ex[0] + x) + 0.03,
        0.5 * (p_ex[1] + y) - 0.05,
        r"$e_y$",
        color="blue",
        fontsize=13
    )

    # arco do erro angular tracejado
    r_arc = 0.30
    theta1 = np.degrees(psi_r)
    theta2 = np.degrees(theta)

    arc = Arc(
        (x, y),
        width=2 * r_arc,
        height=2 * r_arc,
        angle=0,
        theta1=theta1,
        theta2=theta2,
        linestyle="--",
        color="black",
        linewidth=1.5
    )
    ax.add_patch(arc)

    mid = 0.5 * (psi_r + theta)
    ax.text(
        x + (r_arc + 0.05) * np.cos(mid),
        y + (r_arc + 0.05) * np.sin(mid),
        r"$e_\psi$",
        fontsize=13
    )

    # =========================================================
    # 5. Estilo
    # =========================================================
    ax.set_aspect("equal", "box")
    ax.set_xlim(0.0, 5.3)
    ax.set_ylim(0.1, 1.9)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()