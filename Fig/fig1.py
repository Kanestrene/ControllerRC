import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

def ellipse_radius_in_direction(a, b, ux, uy):
    """
    Raio da elipse na direção unitária u = (ux, uy).
    """
    denom = (ux / max(1e-9, a))**2 + (uy / max(1e-9, b))**2
    return 1.0 / np.sqrt(max(1e-12, denom))

def main():
    # -------------------------------------------------
    # Parâmetros
    # -------------------------------------------------
    # obstáculo
    ox, oy = 0.0, 0.0
    r_o = 0.35

    # robô
    x, y = 1.05, 0.62
    yaw = np.deg2rad(30)

    # elipse do robô
    a_ell = 0.30
    b_ell = 0.20

    # margem
    margin = 0.08

    # -------------------------------------------------
    # Direção entre centros
    # -------------------------------------------------
    dx = x - ox
    dy = y - oy
    dist_centers = np.hypot(dx, dy)

    if dist_centers < 1e-12:
        raise ValueError("Os centros do robô e do obstáculo não podem coincidir.")

    ux = dx / dist_centers
    uy = dy / dist_centers

    # raio efetivo do robô nessa direção
    r_robot = ellipse_radius_in_direction(a_ell, b_ell, ux, uy)

    # pontos ao longo da reta entre centros
    p_obs_center = np.array([ox, oy])
    p_obs_surface = p_obs_center + r_o * np.array([ux, uy])
    p_robot_surface = np.array([x, y]) - r_robot * np.array([ux, uy])*1.15
    p_margin_end = p_obs_surface + margin * np.array([ux, uy])

    r_margin = r_o + margin

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))

    # obstáculo
    ax.add_patch(Circle(
        (ox, oy), r_o,
        fill=False, linewidth=2.2
    ))

    # robô
    ax.add_patch(Ellipse(
        (x, y),
        width=2 * a_ell,
        height=2 * b_ell,
        angle=np.degrees(yaw),
        fill=False,
        linewidth=2.2
    ))

    # centros
    ax.plot(ox, oy, "ko", markersize=4)
    ax.plot(x, y, "ko", markersize=4)

    # reta completa entre centros
    # cores (podes ajustar ao teu gosto)
    c_ro = "tab:blue"
    c_margin = "tab:orange"
    c_rrobot = "tab:green"

    # r_obs (centro -> superfície obstáculo)
    ax.plot(
        [p_obs_center[0], p_obs_surface[0]],
        [p_obs_center[1], p_obs_surface[1]],
        color=c_ro, linewidth=3
    )

    # margin (superfície obstáculo -> fim da margem)
    ax.plot(
        [p_obs_surface[0], p_margin_end[0]],
        [p_obs_surface[1], p_margin_end[1]],
        color=c_margin, linewidth=3
    )

    # r_robot (superfície robô -> centro robô)
    ax.plot(
        [p_robot_surface[0], x],
        [p_robot_surface[1], y],
        color=c_rrobot, linewidth=3
    )

    ax.add_patch(Circle(
        (ox, oy),
        r_margin,
        fill=False,
        linestyle="--",   # ou ":" se quiseres mais picotado
        linewidth=1.8,
        color=c_margin,
        label=r"$r_o + margin$"
    ))

    # marcar pontos relevantes na reta
    pts = np.vstack([p_obs_surface, p_robot_surface])
    ax.plot(pts[:, 0], pts[:, 1], "ko", markersize=3)

    # -------------------------------------------------
    # Anotações dos segmentos
    # -------------------------------------------------
    # vetor normal só para deslocar texto visualmente
    nx, ny = -uy, ux
    txt_off = 0.06

    # r_o : centro do obstáculo até à superfície
    mid_ro = 0.5 * (p_obs_center + p_obs_surface)
    ax.text(
        mid_ro[0] + txt_off * nx - 0.06,
        mid_ro[1] + txt_off * ny,
        r"$r_{obs}$",
        fontsize=13
    )

    # margin : superfície do obstáculo até fim da margem
    mid_margin = 0.5 * (p_obs_surface + p_margin_end)
    ax.text(
        mid_margin[0] + txt_off * nx +0.05,
        mid_margin[1] + txt_off * ny -0.1,
        r"$margin$",
        fontsize=11
    )

    # r_robot : superfície do robô até ao centro do robô
    mid_rrobot = 0.5 * (p_robot_surface + np.array([x, y]))
    ax.text(
        mid_rrobot[0] + txt_off * nx - 0.05,
        mid_rrobot[1] + txt_off * ny + 0.03,
        r"$r_{robot}$",
        fontsize=13
    )

    # d : distância entre centros
    # offset da linha paralela
    offset_d = 0

    # pontos deslocados
    ox_d = ox + offset_d * nx
    oy_d = oy + offset_d * ny

    x_d = x + offset_d * nx
    y_d = y + offset_d * ny

    # linha paralela (distância d)
    ax.plot(
        [ox_d, x_d],
        [oy_d, y_d],
        "k--",
        linewidth=1.5
    )

    mid_dx = 0.5 * (ox_d + x_d)
    mid_dy = 0.5 * (oy_d + y_d)

    ax.text(
        mid_dx - 0.05,
        mid_dy + 0.05,
        r"$dist$",
        fontsize=11
    )

    # labels dos centros
    #ax.text(ox - 0.08, oy - 0.08, r"$o$", fontsize=13)
    #ax.text(x + 0.03, y + 0.03, r"$p$", fontsize=13)

    # opcional: fórmula em baixo
    ax.text(
        -0, -0.55,
        r"$r_{safe} \;=\; r_{obs} + r_{robot} + margin$",
        fontsize=14
    )

    # -------------------------------------------------
    # Estilo
    # -------------------------------------------------
    ax.set_aspect("equal", "box")
    ax.set_xlim(-0.8, 1.6)
    ax.set_ylim(-0.8, 1.5)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()