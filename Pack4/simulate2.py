# simulate.py
import os
import yaml
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(BASE_DIR, "paths.yaml")

with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

cars_config = config["cars"]

cars = []

for name, data in cars_config.items():
    waypoints = data["waypoints"]
    v_ref_car = data.get("v_ref", 2.0)

    px, py, pyaw, s = build_spline_path(waypoints, ds=0.01)

    start = data.get("start", [waypoints[0][0], waypoints[0][1], 0.0])

    cars.append({
        "name": name,
        "x": float(start[0]),
        "y": float(start[1]),
        "yaw": float(start[2]),
        "v": 0.0,
        "w": 0.0,
        "delta": 0.0,
        "last_near": 0,
        "px": px,
        "py": py,
        "pyaw": pyaw,
        "s": s,
        "v_ref": v_ref_car,
        "color": data.get("color", None),
    })


def simulate():
    obstacles = []

    dt = 0.02
    T = 60.0
    steps = int(T / dt)

    # só para visualização
    L0 = 0.2
    kv_lookahead = 0.2

    # limites
    w_max = 2.5
    a_max = 1.0

    # elipse do robô
    a_ell, b_ell = 0.35, 0.25
    margin = 0.01

    # parâmetros bicycle/servo
    L = 0.26
    delta_max = np.deg2rad(25)
    delta_rate_max = np.deg2rad(300)

    inner_bar = np.loadtxt(os.path.join(BASE_DIR, "barreira_suavizada_interna.txt"))
    outer_bar = np.loadtxt(os.path.join(BASE_DIR, "barreira_suavizada_externa.txt"))
    inner_x, inner_y = inner_bar[:, 0], inner_bar[:, 1]
    outer_x, outer_y = outer_bar[:, 0], outer_bar[:, 1]

    for car in cars:
        car["hx"], car["hy"], car["ctes"], car["deltaslack"] = [], [], [], []

    def car_as_obstacle(car, r=0.35):
        return {"x": car["x"], "y": car["y"], "r": r}

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    for k in range(steps):
        for i, car in enumerate(cars):
            # lookahead só para desenho
            Ld = L0 + kv_lookahead * abs(car["v"])

            # nominal simples: regula v e amortece w
            a_cmd = 1.5 * (car["v_ref"] - car["v"])
            alpha_cmd = -2.0 * car["w"]

            # obstáculos fixos + outros carros
            obs_all = list(obstacles)
            for j, other in enumerate(cars):
                if j != i:
                    obs_all.append(car_as_obstacle(other, r=0.35))

            # QP com HOCLF + CBF
            u_safe, clf_info = qp.cbf_qp_filter_acc(
                u_nom=(a_cmd, alpha_cmd),
                robot_state=(car["x"], car["y"], car["yaw"], car["v"], car["w"]),
                obstacles=obs_all,
                px=car["px"],
                py=car["py"],
                pyaw=car["pyaw"],
                s=car["s"],
                last_idx=car["last_near"],
                v_ref=car["v_ref"],
                ellipse_ab=(a_ell, b_ell),
                margin=0.4,
                lookahead_l=0.6,
                lambda1=4.0,
                lambda2=4.0,
                W=(20.0, 1.0),
                p_slack=1000.0,
                a_bounds=(-a_max, a_max),
                alpha_bounds=(-4.0, 4.0),
            )

            a_safe, alpha_safe = u_safe

            # info do HOCLF
            car["last_near"] = clf_info["idx"]
            target_idx = clf_info["idx"]
            cte = clf_info["ey"]
            delta_slack = clf_info.get("delta", 0.0)

            # integra velocidades
            car["v"] += a_safe * dt
            car["w"] += alpha_safe * dt

            # limite velocidade linear
            car["v"] = np.clip(car["v"], 0.0, 2.0)

            # limite de yaw-rate compatível com steering
            kappa_max = np.tan(delta_max) / L
            w_max_speed = abs(car["v"]) * kappa_max
            w_lim = min(w_max, w_max_speed)
            car["w"] = np.clip(car["w"], -w_lim, w_lim)

            # converte yaw-rate desejado para steering
            delta_cmd = omega_to_delta(car["w"], car["v"], L, v_min=0.2)
            delta_cmd = np.clip(delta_cmd, -delta_max, delta_max)

            # limita taxa do servo
            car["delta"] = rate_limit(
                delta_cmd,
                car["delta"],
                du_max=delta_rate_max * dt
            )

            # yaw-rate real imposto pelo bicycle
            car["w"] = (car["v"] / L) * np.tan(car["delta"])

            # integra bicycle
            car["x"] += car["v"] * np.cos(car["yaw"]) * dt
            car["y"] += car["v"] * np.sin(car["yaw"]) * dt
            car["yaw"] = wrap_to_pi(car["yaw"] + car["w"] * dt)

            # histórico
            car["hx"].append(car["x"])
            car["hy"].append(car["y"])
            car["ctes"].append(cte)
            car["deltaslack"].append(delta_slack)

            car["_plot"] = {
                "Ld": Ld,
                "target_idx": target_idx,
                "a_safe": a_safe,
                "alpha_safe": alpha_safe,
                "cte": cte,
                "delta_slack": delta_slack,
            }

        if k % 5 == 0:
            ax.clear()

            ax.plot(inner_x, inner_y, "-", linewidth=2, label="Barreira interna")
            ax.plot(outer_x, outer_y, "-", linewidth=2, label="Barreira externa")

            for obs in obstacles:
                ax.add_patch(Circle((obs["x"], obs["y"]), obs["r"], fill=False))

            for car in cars:
                px, py = car["px"], car["py"]

                ax.plot(px, py, "--", label=f"Spline {car['name']}")
                ax.plot(car["hx"], car["hy"], "-", label=f"Trajetória {car['name']}")

                ell = Ellipse(
                    (car["x"], car["y"]),
                    width=2 * a_ell,
                    height=2 * b_ell,
                    angle=np.degrees(car["yaw"]),
                    fill=False
                )
                ax.add_patch(ell)

                ell_safe = Ellipse(
                    (car["x"], car["y"]),
                    width=2 * (a_ell + margin),
                    height=2 * (b_ell + margin),
                    angle=np.degrees(car["yaw"]),
                    fill=False
                )
                ax.add_patch(ell_safe)

                ax.plot(car["x"], car["y"], "o", label=f"Robô {car['name']}")
                ax.arrow(
                    car["x"], car["y"],
                    0.4 * np.cos(car["yaw"]),
                    0.4 * np.sin(car["yaw"]),
                    head_width=0.15
                )

                ti = car["_plot"]["target_idx"]
                ax.plot(px[ti], py[ti], "x", markersize=10, label=f"Ref {car['name']}")
                ax.add_patch(Circle((car["x"], car["y"]), car["_plot"]["Ld"], fill=False))

            ax.set_aspect("equal", "box")
            ax.grid(True)
            ax.set_title("Multi-carro: HOCLF + CBF-QP")
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
            plt.pause(0.001)

    plt.ioff()

    fig2, ax2 = plt.subplots()
    for car in cars:
        ax2.plot(car["ctes"], label=car["name"])
    ax2.set_title("Erro lateral ey - HOCLF")
    ax2.set_xlabel("Passo")
    ax2.set_ylabel("ey [m]")
    ax2.grid(True)
    ax2.legend()

    fig3, ax3 = plt.subplots()
    for car in cars:
        ax3.plot(car["deltaslack"], label=car["name"])
    ax3.set_title("Slack do HOCLF")
    ax3.set_xlabel("Passo")
    ax3.set_ylabel("delta")
    ax3.grid(True)
    ax3.legend()

    plt.show()


if __name__ == "__main__":
    simulate()