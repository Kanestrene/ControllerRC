import numpy as np


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def nearest_path_point_closed(px, py, x, y, last_idx=0, window=300, back=20):
    n = len(px)
    if n == 0:
        raise ValueError("Trajetória vazia.")

    offsets = np.arange(-back, window)
    idxs = (last_idx + offsets) % n

    dx = px[idxs] - x
    dy = py[idxs] - y
    d2 = dx * dx + dy * dy

    j = np.argmin(d2)
    idx = idxs[j]
    return idx, np.sqrt(d2[j])


def path_curvature_closed(pyaw, s, idx):
    n = len(pyaw)
    im1 = (idx - 1) % n
    ip1 = (idx + 1) % n

    dyaw = wrap_to_pi(pyaw[ip1] - pyaw[im1])

    ds = s[ip1] - s[im1]
    L = s[-1]

    if ds < -0.5 * L:
        ds += L
    elif ds > 0.5 * L:
        ds -= L

    if abs(ds) < 1e-8:
        return 0.0

    return dyaw / ds


def _tracking_errors_fixed_ref(
    x, y, theta,
    xr, yr, psi_r,
    lookahead_l=0.6,
):
    """
    Erros de tracking usando ponto lookahead e referência congelada.
    """
    l = lookahead_l

    xl = x + l * np.cos(theta)
    yl = y + l * np.sin(theta)

    dx = xl - xr
    dy = yl - yr

    ex =  np.cos(psi_r) * dx + np.sin(psi_r) * dy
    ey = -np.sin(psi_r) * dx + np.cos(psi_r) * dy
    epsi = wrap_to_pi(theta - psi_r)

    return ex, ey, epsi, xl, yl


def _V_phi_fixed_ref(
    x, y, theta, v, w,
    xr, yr, psi_r, kappa_r,
    v_ref=1.0,
    qx=1.0, qy=4.0, qpsi=2.0,
    lookahead_l=0.6,
):
    """
    Calcula:
      psi0 = V
      dV/dt = phi

    com a referência congelada.
    """
    ex, ey, epsi, xl, yl = _tracking_errors_fixed_ref(
        x, y, theta,
        xr, yr, psi_r,
        lookahead_l=lookahead_l,
    )

    l = lookahead_l
    w_ref = v_ref * kappa_r

    V = 0.5 * (qx * ex**2 + qy * ey**2 + qpsi * epsi**2)

    # dV/dt = av * v + aw * w + c
    av = qx * ex * np.cos(epsi) + qy * ey * np.sin(epsi)

    aw = (
        -qx * ex * l * np.sin(epsi)
        + qy * ey * l * np.cos(epsi)
        + qpsi * epsi
    )

    c = (
        -qx * ex * v_ref
        + w_ref * ex * ey * (qx - qy)
        - qpsi * epsi * w_ref
    )

    phi = av * v + aw * w + c

    info = {
        "V": V,
        "phi": phi,
        "ex": ex,
        "ey": ey,
        "epsi": epsi,
        "xr": xr,
        "yr": yr,
        "psi_r": psi_r,
        "kappa_r": kappa_r,
        "w_ref": w_ref,
        "xl": xl,
        "yl": yl,
        "av": av,
        "aw": aw,
        "c": c,
    }

    return V, phi, info


def hoclf_row_path_tracking_acc(
    px, py, pyaw, s, robot_state,
    last_idx=0,
    v_ref=1.0,
    qx=1.0, qy=4.0, qpsi=2.0,
    lambda1=2.0, lambda2=2.0,
    lookahead_l=0.6,
    fd_eps=1e-5,
    with_slack=False,
):
    """
    HOCLF canónico de ordem 2 para u = [a, alpha].

    Cadeia:
        psi0 = V
        psi1 = dV/dt + lambda1 * V
        psi2 = d(psi1)/dt + lambda2 * psi1 <= 0

    Retorna:
        sem slack:
            G_clf shape (1,2), h_clf shape (1,)
        com slack:
            G_clf shape (1,3), h_clf shape (1,)
            variável = [a, alpha, delta], delta >= 0

    Convenção QP:
        G u <= h
    """
    x, y, theta, v, w = robot_state

    # Escolher referência usando o lookahead atual
    l = lookahead_l
    xl = x + l * np.cos(theta)
    yl = y + l * np.sin(theta)

    idx, _ = nearest_path_point_closed(px, py, xl, yl, last_idx=last_idx)

    xr = px[idx]
    yr = py[idx]
    psi_r = pyaw[idx]
    kappa_r = path_curvature_closed(pyaw, s, idx)

    # psi0 = V, e dV/dt
    V, phi, info = _V_phi_fixed_ref(
        x, y, theta, v, w,
        xr, yr, psi_r, kappa_r,
        v_ref=v_ref,
        qx=qx, qy=qy, qpsi=qpsi,
        lookahead_l=lookahead_l,
    )

    # psi1 = dV/dt + lambda1 * V
    psi1 = phi + lambda1 * V

    # Funções auxiliares com referência congelada
    def V_of_state(x_, y_, th_, v_, w_):
        V_, _, _ = _V_phi_fixed_ref(
            x_, y_, th_, v_, w_,
            xr, yr, psi_r, kappa_r,
            v_ref=v_ref,
            qx=qx, qy=qy, qpsi=qpsi,
            lookahead_l=lookahead_l,
        )
        return V_

    def phi_of_state(x_, y_, th_, v_, w_):
        _, phi_, _ = _V_phi_fixed_ref(
            x_, y_, th_, v_, w_,
            xr, yr, psi_r, kappa_r,
            v_ref=v_ref,
            qx=qx, qy=qy, qpsi=qpsi,
            lookahead_l=lookahead_l,
        )
        return phi_

    def psi1_of_state(x_, y_, th_, v_, w_):
        V_ = V_of_state(x_, y_, th_, v_, w_)
        phi_ = phi_of_state(x_, y_, th_, v_, w_)
        return phi_ + lambda1 * V_

    # Derivadas de psi1
    dpsi1_dx = (
        psi1_of_state(x + fd_eps, y, theta, v, w)
        - psi1_of_state(x - fd_eps, y, theta, v, w)
    ) / (2.0 * fd_eps)

    dpsi1_dy = (
        psi1_of_state(x, y + fd_eps, theta, v, w)
        - psi1_of_state(x, y - fd_eps, theta, v, w)
    ) / (2.0 * fd_eps)

    dpsi1_dtheta = (
        psi1_of_state(x, y, theta + fd_eps, v, w)
        - psi1_of_state(x, y, theta - fd_eps, v, w)
    ) / (2.0 * fd_eps)

    dpsi1_dv = (
        psi1_of_state(x, y, theta, v + fd_eps, w)
        - psi1_of_state(x, y, theta, v - fd_eps, w)
    ) / (2.0 * fd_eps)

    dpsi1_dw = (
        psi1_of_state(x, y, theta, v, w + fd_eps)
        - psi1_of_state(x, y, theta, v, w - fd_eps)
    ) / (2.0 * fd_eps)

    # dpsi1/dt = Lf_psi1 + Lg_psi1 * [a, alpha]
    Lf_psi1 = (
        dpsi1_dx * v * np.cos(theta)
        + dpsi1_dy * v * np.sin(theta)
        + dpsi1_dtheta * w
    )

    Lg_psi1 = np.array([dpsi1_dv, dpsi1_dw], dtype=float)

    # HOCLF:
    # psi2 = dpsi1/dt + lambda2 * psi1 <= 0
    #
    # => Lg_psi1 @ [a, alpha] <= -Lf_psi1 - lambda2 * psi1
    rhs = -Lf_psi1 - lambda2 * psi1

    if with_slack:
        # [a, alpha, delta], com psi2 <= delta
        # <=> Lg[a,alpha] - delta <= rhs
        G_clf = np.array([[Lg_psi1[0], Lg_psi1[1], -1.0]], dtype=float)
        h_clf = np.array([rhs], dtype=float)
    else:
        G_clf = Lg_psi1.reshape(1, 2)
        h_clf = np.array([rhs], dtype=float)

    info.update({
        "idx": idx,
        "psi0": V,
        "psi1": psi1,
        "Lf_psi1": Lf_psi1,
        "Lg_psi1": Lg_psi1,
        "dpsi1_dx": dpsi1_dx,
        "dpsi1_dy": dpsi1_dy,
        "dpsi1_dtheta": dpsi1_dtheta,
        "dpsi1_dv": dpsi1_dv,
        "dpsi1_dw": dpsi1_dw,
    })

    return G_clf, h_clf, info