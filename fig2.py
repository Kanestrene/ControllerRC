import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Arc

def main():
    # ----------------------------------------
    # Parâmetros do carro
    # ----------------------------------------
    x, y = 1.0, 0.8
    yaw = np.deg2rad(30)

    a_car = 0.35
    b_car = 0.20

    # ----------------------------------------
    # Lookahead do carro original
    # ----------------------------------------
    lookahead_dist = 0.7
    lx = x + lookahead_dist * np.cos(yaw)
    ly = y + lookahead_dist * np.sin(yaw)

    # ----------------------------------------
    # Carro sombreado / rodado
    # ----------------------------------------
    theta_overlay = np.deg2rad(25)
    yaw_overlay = yaw + theta_overlay

    lx_overlay = x + lookahead_dist * np.cos(yaw_overlay)
    ly_overlay = y + lookahead_dist * np.sin(yaw_overlay)

    # ----------------------------------------
    # Plot
    # ----------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))

    # carro sombreado
    ax.add_patch(Ellipse(
        (x, y),
        width=2 * a_car,
        height=2 * b_car,
        angle=np.degrees(yaw_overlay),
        facecolor="gray",
        edgecolor="gray",
        alpha=0.25,
        linewidth=2
    ))

    # carro original
    ax.add_patch(Ellipse(
        (x, y),
        width=2 * a_car,
        height=2 * b_car,
        angle=np.degrees(yaw),
        fill=False,
        linewidth=2.5,
        edgecolor="black"
    ))

    # centro
    ax.plot(x, y, "ko", markersize=5)
    ax.text(x + 0.03, y + 0.03, "centro", fontsize=11)

    # lookahead original
    ax.plot(lx, ly, "ro", markersize=6)
    ax.plot([x, lx], [y, ly], color="red", linewidth=2)
    ax.text(lx + 0.03, ly + 0.03, "lookahead", fontsize=11, color="red")

    # lookahead sombreado
    ax.plot(lx_overlay, ly_overlay, "o", color="gray", markersize=6)
    ax.plot([x, lx_overlay], [y, ly_overlay], color="gray", linewidth=2)
    ax.text(
        lx_overlay + 0.03,
        ly_overlay + 0.03,
        "lookahead",
        fontsize=11,
        color="gray"
    )

    # ----------------------------------------
    # Arco picotado entre os dois lookaheads
    # ----------------------------------------
    theta1 = np.degrees(yaw)
    theta2 = np.degrees(yaw_overlay)

    arc = Arc(
        (x, y),
        width=2 * lookahead_dist,
        height=2 * lookahead_dist,
        angle=0,
        theta1=theta1,
        theta2=theta2,
        linestyle="--",
        color="black",
        linewidth=1.5
    )
    ax.add_patch(arc)

    # label do theta no meio do arco
    theta_mid = 0.5 * (yaw + yaw_overlay)
    tx = x + (lookahead_dist + 0.06) * np.cos(theta_mid)
    ty = y + (lookahead_dist + 0.06) * np.sin(theta_mid)

    ax.text(tx, ty, r"$\theta$", fontsize=12)

    # ----------------------------------------
    # Estilo
    # ----------------------------------------
    ax.set_aspect("equal", "box")
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 1.8)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()