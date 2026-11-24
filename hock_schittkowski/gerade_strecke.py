
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Create the optimization problem using CasADi
    opti = ca.Opti()

    # Number of segments into which the track is divided
    n = 40

    # --------------------------------------------------
    # Track and vehicle parameters
    # --------------------------------------------------

    L = 100.0              # Total longitudinal length of the track [m]
    dL = L / n             # Longitudinal length of each segment [m]

    xStart = 2.5           # Initial lateral position of the vehicle [m]
    vStart = 1.0           # Initial vehicle velocity [m/s]

    B = 4.0                # Total width of the track [m]
    m = 790.0              # Vehicle mass [kg]

    alpha = 0.001872       # Aerodynamic resistance coefficient

    F_MotorMax = 10410.0   # Maximum available motor/braking force [N]
    F_HaftMax = 8000.0     # Maximum available tire grip force [N]

    # --------------------------------------------------
    # Obstacle parameters
    # --------------------------------------------------

    y_hindernis = 60.0     # Longitudinal position of the obstacle [m]
    b = 3.5                # Lateral position of the obstacle boundary [m]
    c = B                  # Right track boundary [m]
    eps = 0.01             # Safety margin around obstacle boundaries [m]

    # Check whether the obstacle is located within the track
    if not 0 <= y_hindernis <= L:
        raise ValueError("Das Hindernis muss zwischen y=0 und y=L liegen.")

    # --------------------------------------------------
    # Decision variables and support points
    # --------------------------------------------------

    # Define the optimization variables:
    # First n variables represent lateral positions x
    # Last n variables represent vehicle velocities v
    u = opti.variable(2 * n)

    x = u[:n]              # Lateral positions at the optimization points
    v = u[n:]              # Vehicle velocities at the optimization points

    # Add the fixed initial position and initial velocity
    # This results in n+1 support points for both quantities
    x_stuetzpunkte = ca.vertcat(xStart, x)
    v_stuetzpunkte = ca.vertcat(vStart, v)

    # --------------------------------------------------
    # Determine the lateral position at the obstacle
    # --------------------------------------------------

    # Determine which track segment contains the obstacle
    segment = min(int(y_hindernis / dL), n - 1)

    # Calculate the relative position of the obstacle
    # within the corresponding segment (between 0 and 1)
    t = (y_hindernis - segment * dL) / dL

    # Calculate the lateral position at the obstacle
    # using linear interpolation between two support points
    x_am_hindernis = (
        (1 - t) * x_stuetzpunkte[segment] + t * x_stuetzpunkte[segment + 1]
    )

    # Constraint: The vehicle must pass the obstacle
    # on its right-hand side, with a safety margin
    opti.subject_to(opti.bounded(b + eps, x_am_hindernis, c - eps))

    # --------------------------------------------------
    # Calculate segment lengths, travel times and forces
    # --------------------------------------------------

    # Calculate the lateral displacement between consecutive support points
    dx = x_stuetzpunkte[1:] - x_stuetzpunkte[:-1]

    # Calculate the actual length of each segment
    # using the Pythagorean theorem
    s = ca.sqrt(dL**2 + dx**2)

    # Calculate the average velocity for each segment
    vm = (v_stuetzpunkte[1:] + v_stuetzpunkte[:-1]) / 2

    # Calculate the travel time for each segment:
    # distance divided by average velocity
    dt = s / vm

    # Calculate the change in velocity between consecutive support points
    dv = v_stuetzpunkte[1:] - v_stuetzpunkte[:-1]

    # Calculate acceleration along the trajectory
    # based on the change in velocity over time
    aBahn = dv / dt

    # --------------------------------------------------
    # Calculate velocity components in x and y directions
    # --------------------------------------------------

    # Calculate the directional components of each segment
    # dx / s gives the lateral direction component
    # dL / s gives the longitudinal direction component
    richtung_x = dx / s
    richtung_y = dL / s

    # Calculate the lateral velocity component at each support point
    # The direction of the following segment is used
    # At the final point, the direction of the last segment is used
    v_x = v_stuetzpunkte * ca.vertcat(richtung_x, richtung_x[-1])

    # Calculate the longitudinal velocity component at each support point
    # The same directional convention is used as for v_x
    v_y = v_stuetzpunkte * ca.vertcat(richtung_y, richtung_y[-1])

    # --------------------------------------------------
    # Calculate acceleration components
    # --------------------------------------------------

    # Calculate lateral acceleration from the change in lateral velocity
    a_x = (v_x[1:] - v_x[:-1]) / dt

    # Calculate longitudinal acceleration from the change
    # in longitudinal velocity
    a_y = (v_y[1:] - v_y[:-1]) / dt

    # Calculate the squared magnitude of the total acceleration
    # This avoids taking the square root until it is needed
    a_quadrat = a_x**2 + a_y**2

    # Calculate the magnitude of the total acceleration vector
    a = ca.sqrt(a_quadrat)

    # --------------------------------------------------
    # Calculate the forces acting on the vehicle
    # --------------------------------------------------

    # Calculate the required tire force using Newton's second law:
    # F = m * a
    F_Reifen = m * a

    # Calculate aerodynamic resistance
    # The negative sign indicates that air resistance
    # acts against the direction of travel
    # Resistance increases quadratically with average velocity
    F_Luft = -m * alpha * vm**2

    # Calculate the required motor/braking force
    # This force accounts for acceleration and air resistance
    F_Brems_Motor = m * aBahn - F_Luft

    # --------------------------------------------------
    # Define the objective function and constraints
    # --------------------------------------------------

    # Objective function: Minimize the total travel time
    # by summing the travel times of all individual segments
    f = ca.sum1(dt)
    opti.minimize(f)

    # Constraint: The required motor/braking force
    # must not exceed the maximum available motor force
    opti.subject_to(F_Brems_Motor / F_MotorMax <= 1)

    # Constraint: The squared total acceleration must remain
    # below the squared maximum acceleration allowed by tire grip
    # Using squared acceleration avoids a square root
    # and keeps the constraint smooth when acceleration is zero
    opti.subject_to(a_quadrat / (F_HaftMax / m)**2 <= 1)

    # Constraint: The vehicle must remain within
    # the lateral boundaries of the track
    opti.subject_to(opti.bounded(0, x, B))

    # Constraint: Velocity must always be positive
    # This prevents division by zero when calculating travel time
    opti.subject_to(v >= 0.001)

    # --------------------------------------------------
    # Initialize and solve the optimization problem
    # --------------------------------------------------

    # Set initial guesses for all optimization variables:
    # All lateral positions are initialized at xStart
    # All velocities are initialized at vStart
    opti.set_initial(u, [xStart] * n + [vStart] * n)

    # Select IPOPT as the nonlinear optimization solver
    opti.solver("ipopt")

    # Solve the optimization problem
    sol = opti.solve()

    # --------------------------------------------------
    # Print the optimization results
    # --------------------------------------------------

    # Print the optimized lateral positions
    print("x* =", sol.value(x))

    # Print the optimized velocities
    print("v* =", sol.value(v))

    # Print the travel time of each segment
    print("dt* =", sol.value(dt))

    # Print the total optimized travel time
    print("T* =", sol.value(f))

    # Print the required motor/braking force for each segment
    print("F_Brems_Motor =", sol.value(F_Brems_Motor))

    # Print the required tire force for each segment
    print("F_Reifen =", sol.value(F_Reifen))

    # --------------------------------------------------
    # Prepare the optimized trajectory for plotting
    # --------------------------------------------------

    # Convert the optimized lateral positions into a NumPy array
    x_opt = np.atleast_1d(sol.value(x))

    # Generate the longitudinal coordinates of all support points
    # The first point is y = 0 and the last point is y = L
    y_stuetzpunkte = [i * dL for i in range(n + 1)]

    # --------------------------------------------------
    # Plot 1: Optimized vehicle trajectory and obstacle
    # --------------------------------------------------

    # Create a figure showing the vehicle's optimized path
    # along the track and its position relative to the obstacle
    _, ax = plt.subplots(figsize=(5, 8), layout="constrained")

    # Plot the optimized trajectory using the lateral and
    # longitudinal positions of all support points
    ax.plot([xStart, *x_opt], y_stuetzpunkte, "o-", label="Optimale Strecke")

    # Plot the left track boundary
    ax.axvline(0, color="gray", linestyle="--", label="Fahrbahnrand")

    # Plot the right track boundary
    ax.axvline(B, color="gray", linestyle="--")

    # Plot the obstacle as a red horizontal line
    # It extends from x = 0 to x = b at y = y_hindernis
    ax.plot(
        [0, b], [y_hindernis, y_hindernis],
        color="tab:red", linewidth=4, solid_capstyle="butt", label="Hindernis",
    )

    # Set the axis labels
    ax.set(xlabel="Seitliche Position x [m]", ylabel="Strecke y [m]")

    # Add a grid to improve readability
    ax.grid(alpha=0.3)

    # Display the legend
    ax.legend(loc="upper left")

    # --------------------------------------------------
    # Create three vertically arranged plots:
    # Velocity, acceleration and forces
    # --------------------------------------------------

    # sharex=True means all plots use the same longitudinal axis
    _, (ax_v, ax_a, ax_f) = plt.subplots(
        3, 1, sharex=True, figsize=(10, 8), layout="constrained",
    )

    # --------------------------------------------------
    # Plot 2: Velocity components
    # --------------------------------------------------

    # Plot the total vehicle speed at the support points
    ax_v.plot(
        y_stuetzpunkte, sol.value(v_stuetzpunkte),
        "o", markersize=5, markerfacecolor="none", label="Betrag v an den Stützpunkten",
    )

    # Plot the lateral velocity component
    ax_v.plot(
        y_stuetzpunkte, sol.value(v_x),
        ".", color="tab:green", label="$v_x$ seitlich",
    )

    # Plot the longitudinal velocity component
    ax_v.plot(
        y_stuetzpunkte, sol.value(v_y),
        ".", markersize=4, color="tab:red", label="$v_y$ längs der Strecke",
    )

    # Plot the average velocity for each segment
    # stairs() displays constant values over individual segments
    ax_v.stairs(
        np.atleast_1d(sol.value(vm)), y_stuetzpunkte, baseline=None,
        color="tab:orange", label="Mittelwert pro Segment",
    )

    # Set the y-axis label and display the legend
    ax_v.set_ylabel("Geschwindigkeit [m/s]")
    ax_v.legend(ncol=2)

    # --------------------------------------------------
    # Plot 3: Acceleration components
    # --------------------------------------------------

    # Plot the magnitude of the total acceleration
    ax_a.stairs(
        np.atleast_1d(sol.value(a)), y_stuetzpunkte, baseline=None,
        color="tab:blue", label="Betrag a",
    )

    # Plot the acceleration along the trajectory
    ax_a.stairs(
        np.atleast_1d(sol.value(aBahn)), y_stuetzpunkte, baseline=None,
        color="tab:orange", linestyle="--", label="Bahnbeschleunigung",
    )

    # Plot the lateral acceleration component
    ax_a.stairs(
        np.atleast_1d(sol.value(a_x)), y_stuetzpunkte, baseline=None,
        color="tab:green", label="$a_x$ seitlich",
    )

    # Plot the longitudinal acceleration component
    ax_a.stairs(
        np.atleast_1d(sol.value(a_y)), y_stuetzpunkte, baseline=None,
        color="tab:red", linestyle=":", label="$a_y$ längs der Strecke",
    )

    # Set the y-axis label and display the legend
    ax_a.set_ylabel("Beschleunigung [m/s²]")
    ax_a.legend(ncol=2)

    # --------------------------------------------------
    # Plot 4: Forces acting on the vehicle
    # --------------------------------------------------

    # Plot the required motor/braking force for each segment
    ax_f.stairs(
        np.atleast_1d(sol.value(F_Brems_Motor)), y_stuetzpunkte, baseline=None,
        color="tab:blue", label="Antrieb/Bremse",
    )

    # Plot the required tire force for each segment
    ax_f.stairs(
        np.atleast_1d(sol.value(F_Reifen)), y_stuetzpunkte, baseline=None,
        color="tab:orange", label="Reifenkraft (Betrag)",
    )

    # Plot the aerodynamic resistance for each segment
    ax_f.stairs(
        np.atleast_1d(sol.value(F_Luft)), y_stuetzpunkte, baseline=None,
        color="tab:green", label="Luftwiderstand",
    )

    # Add a horizontal line indicating the maximum motor force
    ax_f.axhline(F_MotorMax, color="tab:blue", linestyle="--", label="Motorgrenze")

    # Add a horizontal line indicating the maximum tire grip force
    ax_f.axhline(F_HaftMax, color="tab:orange", linestyle="--", label="Haftgrenze")

    # Calculate the spacing between the x-axis tick marks
    # This keeps the number of displayed labels manageable
    tick_schritt = (n + 9) // 10

    # Generate the longitudinal positions for the x-axis tick marks
    # Always include the final track position
    y_ticks = y_stuetzpunkte[:-1:tick_schritt] + [y_stuetzpunkte[-1]]

    # Set the axis labels and x-axis tick positions
    ax_f.set(xlabel="Strecke y [m]", ylabel="Kraft [N]", xticks=y_ticks)

    # Display the legend
    ax_f.legend(ncol=2)

    # --------------------------------------------------
    # Final plot formatting
    # --------------------------------------------------

    # Add a grid to all three plots to improve readability
    for achse in (ax_v, ax_a, ax_f):
        achse.grid(alpha=0.3)

    # Display all generated figures
    plt.show()


# Execute main() only when this Python file is run directly
# and not when it is imported as a module
if __name__ == "__main__":
    main()
