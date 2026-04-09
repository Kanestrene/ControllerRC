import numpy as np


def ellipse_radius_in_direction(a, b, ux, uy):
    denom = (ux / max(1e-9, a))**2 + (uy / max(1e-9, b))**2
    return 1.0 / np.sqrt(max(1e-12, denom))


def _lookahead_kinematics(x, y, th, v, w, l):
    """
    Ponto lookahead p = [px, py] e suas derivadas.

    Modelo:
        x_dot  = v cos(th)
        y_dot  = v sin(th)
        th_dot = w
        v_dot  = a
        w_dot  = alpha_w

    Aqui devolvemos:
      - px, py
      - px_dot, py_dot
      - coeficientes lineares de a e alpha_w em px_ddot, py_ddot
      - parte drift de px_ddot, py_ddot
    """
    c = np.cos(th)
    s = np.sin(th)

    px = x + l * c
    py = y + l * s

    px_dot = v * c - l * w * s
    py_dot = v * s + l * w * c

    # px_ddot = a*c - v*w*s - l*alpha_w*s - l*w^2*c
    # py_ddot = a*s + v*w*c + l*alpha_w*c - l*w^2*s

    # coeficientes do controlo u = [a, alpha_w]
    px_ddot_a = c
    px_ddot_alpha = -l * s

    py_ddot_a = s
    py_ddot_alpha = l * c

    # drift
    px_ddot_drift = -v * w * s - l * w * w * c
    py_ddot_drift =  v * w * c - l * w * w * s

    return (
        px, py,
        px_dot, py_dot,
        px_ddot_a, px_ddot_alpha,
        py_ddot_a, py_ddot_alpha,
        px_ddot_drift, py_ddot_drift,
    )


def cbf_rows_for_circle_obstacles_acc(
    x, y, th, v, w, obstacles,
    ellipse_ab=(0.30, 0.20),
    margin=0.05,
    lookahead_l=0.35,
    lambda1=2.0,
    lambda2=2.0,
):
    """
    HOCBF ordem 2 para obstáculos circulares fixos.

    Retorna G, h para:
        G u <= h,  com u = [a, alpha_w]

    Impõe:
        h_ddot + (lambda1 + lambda2) h_dot + lambda1*lambda2*h >= 0
    """
    a_ell, b_ell = ellipse_ab
    l = lookahead_l

    (
        px, py,
        px_dot, py_dot,
        px_ddot_a, px_ddot_alpha,
        py_ddot_a, py_ddot_alpha,
        px_ddot_drift, py_ddot_drift,
    ) = _lookahead_kinematics(x, y, th, v, w, l)

    G_list = []
    h_list = []

    for obs in obstacles:
        ox, oy, ro = obs["x"], obs["y"], obs["r"]

        # inflação pela elipse do robô, usando direção centro->obstáculo
        dx_c = x - ox
        dy_c = y - oy
        dist_c = np.hypot(dx_c, dy_c)
        if dist_c < 1e-9:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = dx_c / dist_c, dy_c / dist_c

        r_robot = ellipse_radius_in_direction(a_ell, b_ell, ux, uy)
        r_safe = ro + r_robot + margin

        dx = px - ox
        dy = py - oy

        h_val = dx * dx + dy * dy - r_safe * r_safe
        h_dot = 2.0 * (dx * px_dot + dy * py_dot)

        # h_ddot = 2(px_dot^2 + py_dot^2) + 2(dx px_ddot + dy py_ddot)
        #        = drift + c_a * a + c_alpha * alpha_w

        c_a = 2.0 * (dx * px_ddot_a + dy * py_ddot_a)
        c_alpha = 2.0 * (dx * px_ddot_alpha + dy * py_ddot_alpha)

        h_ddot_drift = (
            2.0 * (px_dot * px_dot + py_dot * py_dot)
            + 2.0 * (dx * px_ddot_drift + dy * py_ddot_drift)
        )

        rhs = h_ddot_drift + (lambda1 + lambda2) * h_dot + (lambda1 * lambda2) * h_val

        # c_a * a + c_alpha * alpha_w + rhs >= 0
        # => -(c_a * a + c_alpha * alpha_w) <= rhs
        G_list.append([-c_a, -c_alpha])
        h_list.append(rhs)

    if len(G_list) == 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

    return np.array(G_list, dtype=float), np.array(h_list, dtype=float)


def _closest_point_on_segment(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    denom = abx * abx + aby * aby
    if denom < 1e-12:
        return ax, ay, 0.0

    t = (apx * abx + apy * aby) / denom
    t = np.clip(t, 0.0, 1.0)

    qx = ax + t * abx
    qy = ay + t * aby
    return qx, qy, t


def cbf_rows_for_barriers_acc(
    x, y, th, v, w,
    barrier_inner, barrier_outer,
    ellipse_ab=(0.30, 0.20),
    margin=0.05,
    lookahead_l=0.35,
    lambda1=2.0,
    lambda2=2.0,
    max_segments=40,
):
    """
    HOCBF ordem 2 para barreiras poligonais.

    Retorna G, h para:
        G u <= h,  com u = [a, alpha_w]

    Aproximação usada:
      o ponto mais próximo q em cada segmento é congelado no instante atual.
    Isso evita derivar dq/dt.
    """
    a_ell, b_ell = ellipse_ab
    l = lookahead_l

    (
        px, py,
        px_dot, py_dot,
        px_ddot_a, px_ddot_alpha,
        py_ddot_a, py_ddot_alpha,
        px_ddot_drift, py_ddot_drift,
    ) = _lookahead_kinematics(x, y, th, v, w, l)

    def add_constraints_from_poly(poly):
        G_list = []
        h_list = []

        poly = np.asarray(poly, dtype=float)
        if poly.shape[0] < 2:
            return G_list, h_list

        if np.hypot(poly[0, 0] - poly[-1, 0], poly[0, 1] - poly[-1, 1]) > 1e-9:
            poly2 = np.vstack([poly, poly[0]])
        else:
            poly2 = poly

        A = poly2[:-1]
        B = poly2[1:]

        d2 = np.empty(len(A), dtype=float)
        q_cache = np.empty((len(A), 2), dtype=float)

        for i, ((ax, ay), (bx, by)) in enumerate(zip(A, B)):
            qx, qy, _ = _closest_point_on_segment(px, py, ax, ay, bx, by)
            q_cache[i, 0] = qx
            q_cache[i, 1] = qy

            dx = px - qx
            dy = py - qy
            d2[i] = dx * dx + dy * dy

        m = min(max_segments, len(A))
        idxs = np.argpartition(d2, m - 1)[:m]

        for i in idxs:
            qx, qy = q_cache[i, 0], q_cache[i, 1]
            dx = px - qx
            dy = py - qy
            dist = np.hypot(dx, dy)

            if dist < 1e-9:
                ux, uy = np.cos(th), np.sin(th)
            else:
                ux, uy = dx / dist, dy / dist

            r_robot = ellipse_radius_in_direction(a_ell, b_ell, ux, uy)
            d_safe = r_robot + margin

            h_val = dx * dx + dy * dy - d_safe * d_safe
            h_dot = 2.0 * (dx * px_dot + dy * py_dot)

            c_a = 2.0 * (dx * px_ddot_a + dy * py_ddot_a)
            c_alpha = 2.0 * (dx * px_ddot_alpha + dy * py_ddot_alpha)

            h_ddot_drift = (
                2.0 * (px_dot * px_dot + py_dot * py_dot)
                + 2.0 * (dx * px_ddot_drift + dy * py_ddot_drift)
            )

            rhs = (
                h_ddot_drift
                + (lambda1 + lambda2) * h_dot
                + (lambda1 * lambda2) * h_val
            )

            G_list.append([-c_a, -c_alpha])
            h_list.append(rhs)

        return G_list, h_list

    G_all = []
    h_all = []

    for poly in (barrier_inner, barrier_outer):
        Gi, hi = add_constraints_from_poly(poly)
        G_all.extend(Gi)
        h_all.extend(hi)

    if len(G_all) == 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

    return np.array(G_all, dtype=float), np.array(h_all, dtype=float)