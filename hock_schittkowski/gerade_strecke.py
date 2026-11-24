import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def main():
    opti = ca.Opti()
    n = 100

    # Strecke und Fahrzeug
    L = 100.0
    dL = L / n
    xStart = 0.0
    vStart = 1.0
    B = 4.0
    m = 790.0
    alpha = 0.001872
    F_MotorMax = 10410.0
    F_HaftMax = 5000.0

    # (Streckenposition y [m], Breite [m], Fahrbahnrand)
    hindernisse = [
        (40.0, 3.5, "links"),
        (50.0, 3.5, "rechts"),
        (72.0, 3.8, "rechts"),
        (90.0,1.5,"links"),
        (90.0,1.5,"rechts")
    ]
    eps = 0.01

    # Entscheidungsvariablen und Stützpunkte
    u = opti.variable(2 * n)
    x = u[:n]
    v = u[n:]
    x_stuetzpunkte = ca.vertcat(xStart, x)
    v_stuetzpunkte = ca.vertcat(vStart, v)

    # Seitenposition auf Höhe jedes Hindernisses
    hindernisbereiche = []
    for y_hindernis, breite, seite in hindernisse:
        if not 0 <= y_hindernis <= L:
            raise ValueError(f"Hindernis bei y={y_hindernis:g} m liegt außerhalb der Strecke.")
        if not 0 <= breite <= B - 2 * eps:
            raise ValueError(f"Ungültige Hindernisbreite bei y={y_hindernis:g} m.")

        if seite == "links":
            x_von, x_bis = 0, breite
            x_min, x_max = x_bis + eps, B - eps
        elif seite == "rechts":
            x_von, x_bis = B - breite, B
            x_min, x_max = eps, x_von - eps
        else:
            raise ValueError(f"Ungültige Hindernisseite {seite!r}. Erlaubt sind 'links' und 'rechts'.")
        hindernisbereiche.append((y_hindernis, x_von, x_bis))

        segment = min(int(y_hindernis / dL), n - 1)
        t = (y_hindernis - segment * dL) / dL
        x_am_hindernis = (
            (1 - t) * x_stuetzpunkte[segment] + t * x_stuetzpunkte[segment + 1]
        )
        opti.subject_to(opti.bounded(x_min, x_am_hindernis, x_max))

    # Streckenlängen, Fahrzeiten und Kräfte pro Segment
    dx = x_stuetzpunkte[1:] - x_stuetzpunkte[:-1]
    s = ca.sqrt(dL**2 + dx**2)
    vm = (v_stuetzpunkte[1:] + v_stuetzpunkte[:-1]) / 2
    dt = s / vm
    dv = v_stuetzpunkte[1:] - v_stuetzpunkte[:-1]
    aBahn = dv / dt

    # Die Stützpunktgeschwindigkeit zeigt in das folgende Segment.
    # Am Ziel gilt die Richtung des letzten Segments.
    richtung_x = dx / s
    richtung_y = dL / s
    v_x = v_stuetzpunkte * ca.vertcat(richtung_x, richtung_x[-1])
    v_y = v_stuetzpunkte * ca.vertcat(richtung_y, richtung_y[-1])
    a_x = (v_x[1:] - v_x[:-1]) / dt
    a_y = (v_y[1:] - v_y[:-1]) / dt
    a_quadrat = a_x**2 + a_y**2
    a = ca.sqrt(a_quadrat)
    F_Reifen = m * a
    F_Luft = -m * alpha * vm**2
    F_Brems_Motor = m * aBahn - F_Luft

    # Fahrzeit minimieren und Fahrzeug- und Fahrbahngrenzen einhalten
    f = ca.sum1(dt)
    opti.minimize(f)
    opti.subject_to(F_Brems_Motor / F_MotorMax <= 1)
    # Die quadrierte, normierte Haftgrenze ist auch bei a = 0 glatt.
    opti.subject_to(a_quadrat / (F_HaftMax / m)**2 <= 1)
    opti.subject_to(opti.bounded(0, x, B))
    opti.subject_to(v >= 0.001)

    # Eine moderate Beschleunigung liefert Ipopt ein Startprofil für v.
    a_init = 0.25 * F_HaftMax / m
    v_init = np.sqrt(vStart**2 + 2 * a_init * dL * np.arange(1, n + 1))
    opti.set_initial(x, xStart)
    opti.set_initial(v, v_init)
    opti.solver("ipopt")
    sol = opti.solve()

    print("x* =", sol.value(x))
    print("v* =", sol.value(v))
    print("dt* =", sol.value(dt))
    print("T* =", sol.value(f))
    print("F_Brems_Motor =", sol.value(F_Brems_Motor))
    print("F_Reifen =", sol.value(F_Reifen))

    x_opt = np.atleast_1d(sol.value(x))
    y_stuetzpunkte = [i * dL for i in range(n + 1)]
    streckenpunkte = np.column_stack(([xStart, *x_opt], y_stuetzpunkte))
    streckensegmente = np.stack((streckenpunkte[:-1], streckenpunkte[1:]), axis=1)
    v_opt = sol.value(v_stuetzpunkte)
    fig, ax = plt.subplots(figsize=(5, 8), layout="constrained")
    fahrlinie = LineCollection(
        streckensegmente, array=np.atleast_1d(sol.value(vm)), cmap="viridis",
        label="Optimale Strecke",
    )
    fahrlinie.set_clim(v_opt.min(), v_opt.max())
    ax.add_collection(fahrlinie)
    ax.scatter(
        streckenpunkte[:, 0], y_stuetzpunkte, c=v_opt,
        cmap=fahrlinie.cmap, norm=fahrlinie.norm,
    )
    fig.colorbar(fahrlinie, ax=ax, label="Geschwindigkeitsbetrag [m/s]")
    ax.axvline(0, color="gray", linestyle="--", label="Fahrbahnrand")
    ax.axvline(B, color="gray", linestyle="--")
    for index, (y_hindernis, x_von, x_bis) in enumerate(hindernisbereiche):
        ax.plot(
            [x_von, x_bis], [y_hindernis, y_hindernis],
            color="tab:red", linewidth=4, solid_capstyle="butt",
            label="Hindernisse" if index == 0 else None,
        )
    ax.set(xlabel="Seitliche Position x [m]", ylabel="Strecke y [m]")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    _, (ax_v, ax_a, ax_f) = plt.subplots(
        3, 1, sharex=True, figsize=(10, 8), layout="constrained",
    )
    ax_v.plot(
        y_stuetzpunkte, sol.value(v_stuetzpunkte),
        "o", markersize=5, markerfacecolor="none", label="Betrag v an den Stützpunkten",
    )
    ax_v.plot(
        y_stuetzpunkte, sol.value(v_x),
        ".", color="tab:green", label="$v_x$ seitlich",
    )
    ax_v.plot(
        y_stuetzpunkte, sol.value(v_y),
        ".", markersize=4, color="tab:red", label="$v_y$ längs der Strecke",
    )
    ax_v.stairs(
        np.atleast_1d(sol.value(vm)), y_stuetzpunkte, baseline=None,
        color="tab:orange", label="Mittelwert pro Segment",
    )
    ax_v.set_ylabel("Geschwindigkeit [m/s]")
    ax_v.legend(ncol=2)

    ax_a.stairs(
        np.atleast_1d(sol.value(a)), y_stuetzpunkte, baseline=None,
        color="tab:blue", label="Betrag a",
    )
    ax_a.stairs(
        np.atleast_1d(sol.value(aBahn)), y_stuetzpunkte, baseline=None,
        color="tab:orange", linestyle="--", label="Bahnbeschleunigung",
    )
    ax_a.stairs(
        np.atleast_1d(sol.value(a_x)), y_stuetzpunkte, baseline=None,
        color="tab:green", label="$a_x$ seitlich",
    )
    ax_a.stairs(
        np.atleast_1d(sol.value(a_y)), y_stuetzpunkte, baseline=None,
        color="tab:red", linestyle=":", label="$a_y$ längs der Strecke",
    )
    ax_a.set_ylabel("Beschleunigung [m/s²]")
    ax_a.legend(ncol=2)

    ax_f.stairs(
        np.atleast_1d(sol.value(F_Brems_Motor)), y_stuetzpunkte, baseline=None,
        color="tab:blue", label="Antrieb/Bremse",
    )
    ax_f.stairs(
        np.atleast_1d(sol.value(F_Reifen)), y_stuetzpunkte, baseline=None,
        color="tab:orange", label="Reifenkraft (Betrag)",
    )
    ax_f.stairs(
        np.atleast_1d(sol.value(F_Luft)), y_stuetzpunkte, baseline=None,
        color="tab:green", label="Luftwiderstand",
    )
    ax_f.axhline(F_MotorMax, color="tab:blue", linestyle="--", label="Motorgrenze")
    ax_f.axhline(F_HaftMax, color="tab:orange", linestyle="--", label="Haftgrenze")
    tick_schritt = (n + 9) // 10
    y_ticks = y_stuetzpunkte[:-1:tick_schritt] + [y_stuetzpunkte[-1]]
    ax_f.set(xlabel="Strecke y [m]", ylabel="Kraft [N]", xticks=y_ticks)
    ax_f.legend(ncol=2)

    for achse in (ax_v, ax_a, ax_f):
        achse.grid(alpha=0.3)

    plt.show()


if __name__ == "__main__":
    main()
