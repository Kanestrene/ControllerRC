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
        "color": data.get("color", "black"),
    })


def rel_in_car_frame(car, other):
    dx = other["x"] - car["x"]
    dy = other["y"] - car["y"]
    forward = dx * np.cos(car["yaw"]) + dy * np.sin(car["yaw"])
    lateral = -dx * np.sin(car["yaw"]) + dy * np.cos(car["yaw"])
    dist = np.hypot(dx, dy)
    return dist, forward, lateral


def should_consider_other(car, other, d_act=3.0):
    dist, forward, lateral = rel_in_car_frame(car, other)
    if dist > d_act:
        return False
    if forward < 0.0:
        return False
    if abs(lateral) > 1.2:
        return False
    return True


QP_PARAMS = {
    "leader": {
        "W": (100000.0, 10.0),
        "lambda1": 2.0,
        "lambda2": 2.0,
        "lookahead_l": 0.60,
        "p_slack": 500.0,
    },
    "follower": {
        "W": (100.0, 10.0),
        "lambda1": 2.0,
        "lambda2": 2.0,
        "lookahead_l": 0.40,
        "p_slack": 500.0,
    },
}


def simulate():
    obstacles = []

    dt = 0.02
    T = 50.0
    steps = int(T / dt)

    # só para desenho
    L0 = 0.2
    kv_lookahead = 0.2

    # limites
    w_max = 2.5
    a_max = 2.0
    alpha_max = 4.0

    # elipse do robô
    a_ell, b_ell = 0.60, 0.50
    margin = 0.05

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

    def car_as_obstacle(car, r=0.45):
        return {"x": car["x"], "y": car["y"], "r": r}

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 5))

    for k in range(steps):
        for i, car in enumerate(cars):
            Ld = L0 + kv_lookahead * abs(car["v"])

            # obstáculos fixos + outros carros
            role = "leader"
            nearest_front_dist = 1e9
            obs_all = list(obstacles)

            for j, other in enumerate(cars):
                if j == i:
                    continue

                if not should_consider_other(car, other, d_act=3.0):
                    continue

                dist, forward, lateral = rel_in_car_frame(car, other)
                obs_all.append(car_as_obstacle(other, r=0.25))

                if forward > 0.0 and dist < nearest_front_dist:
                    nearest_front_dist = dist
                    role = "follower"

            p = QP_PARAMS[role]

            # nominal simples: regula velocidade e amortece yaw-rate
            a_cmd = 1.5 * (car["v_ref"] - car["v"])
            alpha_cmd = -2.0 * car["w"]

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
                margin=margin,
                lookahead_l=p["lookahead_l"],
                lambda1=p["lambda1"],
                lambda2=p["lambda2"],
                W=p["W"],
                p_slack=p["p_slack"],
                a_bounds=(-a_max, a_max),
                alpha_bounds=(-alpha_max, alpha_max),
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

            # limites velocidade linear
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
                "role": role,
                "delta_slack": delta_slack,
            }

        if k % 5 == 0:
            ax.clear()

            ax.plot(inner_x, inner_y, "-", linewidth=2, label="Barreira interna")
            ax.plot(outer_x, outer_y, "-", linewidth=2, label="Barreira externa")

            for car in cars:
                px, py = car["px"], car["py"]
                ax.plot(px, py, "--", linewidth=1, label=f"Spline {car['name']}")

                role = car["_plot"].get("role", "?")
                ax.plot(
                    car["hx"],
                    car["hy"],
                    "-",
                    linewidth=2,
                    label=f"Traj {car['name']} ({role})"
                )

                for obs in obstacles:
                    ax.add_patch(Circle((obs["x"], obs["y"]), obs["r"], fill=False))

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

                ax.plot(car["x"], car["y"], "o")
                ax.arrow(
                    car["x"], car["y"],
                    0.4 * np.cos(car["yaw"]),
                    0.4 * np.sin(car["yaw"]),
                    head_width=0.15
                )

                ti = car["_plot"]["target_idx"]
                ax.plot(px[ti], py[ti], "x", markersize=8)
                ax.add_patch(Circle((car["x"], car["y"]), car["_plot"]["Ld"], fill=False))

            ax.set_aspect("equal", "box")
            ax.grid(True)
            ax.set_title("Multi-carro: HOCLF + CBF-QP(acc→bicycle)")
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