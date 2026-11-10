import numpy as np
import matplotlib.pyplot as plt


# one fourth-order Runge-Kutta step for dy/dt = f(t,y)
def rk4_step(f, t, y, h):

    k1 = h * f(t, y)
    k2 = h * f(t + 0.5*h, y + 0.5*k1)
    k3 = h * f(t + 0.5*h, y + 0.5*k2)
    k4 = h * f(t + h, y + k3)

    return y + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0


# adaptive RK4 using one full step and two half steps
# check selects which variables are subject to the error tolerance
# for orbits, check=(0, 1) controls the position error per unit time
def rk4_adaptive(f, t0, y0, t_end, h0, delta=1.0e-8,
                 check=None, stop=None, max_steps=1000000):

    t = float(t0)
    y = np.array(y0, dtype=float)
    h = float(h0)

    times = [t]
    states = [y.copy()]

    steps = 0
    while t < t_end:

        if max_steps is not None and steps >= max_steps:
            print(f"Step limit reached at t = {t:.6f}")
            break
        steps += 1

        h = min(h, t_end - t)
        if t + h == t:
            raise RuntimeError(f"Time step too small at t = {t:.8g}; increase delta")

        # one full RK4 step and two half steps
        y_full = rk4_step(f, t, y, h)
        y_half = rk4_step(f, t, y, 0.5*h)
        y_half = rk4_step(f, t + 0.5*h, y_half, 0.5*h)

        difference = y_half - y_full
        if check is not None:
            difference = difference[list(check)]
        error = np.max(np.abs(difference))/(15.0*h)

        if np.isfinite(error) and error <= delta:
            t += h
            y = y_half + (y_half - y_full)/15.0
            times.append(t)
            states.append(y.copy())

            # stop at the first accepted point within the target radius
            if stop is not None and stop(t, y) <= 0.0:
                break

        # local error per unit time scales as h^4
        factor = 0.5 if not np.isfinite(error) else (
            2.0 if error == 0.0 else 0.9*(delta/error)**0.25
        )
        h *= min(2.0, max(0.1, factor))

    return np.array(times), np.array(states)


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
})


def style_axes(ax):
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", direction="out",
                   length=7, width=1.1, pad=5)
    ax.tick_params(axis="both", which="minor", direction="out",
                   length=4, width=0.9)


# problem constants: G = M = 1, distances in 100 pc
G = 1.0
M = 1.0
mu = G*M/4.0
r_s = 1.0e-7
r0 = 1.0
v_circ = np.sqrt(mu/r0)

# physical time unit inferred from the given Schwarzschild radius
pc = 3.085677581e16   # m
c = 2.99792458e8      # m/s
G_SI = 6.67430e-11    # m^3/(kg s^2)
M_sun = 1.98847e30    # kg
Myr = 1.0e6 * 365.25 * 24.0 * 3600.0  # s
L = 100.0 * pc
M_phys = r_s * L * c**2/(2.0*G_SI)
time_Myr = np.sqrt(L**3/(G_SI*M_phys))/Myr


def smbh(t, y, A=0.0, B=1.0):

    x, y_pos, vx, vy = y
    r = np.hypot(x, y_pos)
    v = np.hypot(vx, vy)

    # gravity of the other BH and dynamical friction
    friction = A/(v**3 + B)
    ax = -mu*x/r**3 - friction*vx
    ay = -mu*y_pos/r**3 - friction*vy

    return np.array([vx, vy, ax, ay])


def radius(t, y):
    return np.hypot(y[0], y[1]) - r_s


# run an orbit; with friction, terminate on first crossing of r_s
def orbit(A, B, v_fraction=0.8, delta=1.0e-8,
          t_end=100.0, max_steps=1000000):

    y0 = [r0, 0.0, 0.0, v_fraction*v_circ]

    def f(t, y):
        return smbh(t, y, A, B)

    stop = radius if A > 0.0 else None
    times, states = rk4_adaptive(f, 0.0, y0, t_end, 0.02,
                                 delta=delta, check=(0, 1), stop=stop, max_steps=max_steps)

    reached = np.hypot(states[-1, 0], states[-1, 1]) <= r_s
    t_hit = times[-1] if reached else np.nan

    return times, states, t_hit


# part (a): no friction, with a pericenter of 10^-7
def part_a():

    # tangential speed at apocenter from the Kepler orbit
    r_peri = r_s
    v0 = np.sqrt(2.0*mu*r_peri/(r0*(r0 + r_peri)))
    a = 0.5*(r0 + r_peri)
    period = 2.0*np.pi*np.sqrt(a**3/mu)

    def f(t, y):
        return smbh(t, y, 0.0, 1.0)

    times, states = rk4_adaptive(f, 0.0, [r0, 0.0, 0.0, v0],
                                 10.0*period, 0.02, delta=1.0e-9, check=(0, 1))

    x, y_pos, vx, vy = states.T
    r = np.hypot(x, y_pos)
    energy = 0.5*(vx**2 + vy**2) - mu/r
    angular_momentum = x*vy - y_pos*vx

    # pericenter inferred from the conserved Kepler orbital elements
    eccentricity = np.sqrt(1.0 + 2.0*energy*angular_momentum**2/mu**2)
    r_peri_num = angular_momentum**2/(mu*(1.0 + eccentricity))

    print("PART (a): FRICTIONLESS ORBIT")
    print(f"Initial tangential speed: {v0:.10e}")
    print(f"Period: {period:.8f}; completed: {times[-1]/period:.6f} orbits")
    print(f"Minimum sampled radius: {r.min():.10e}")
    print(f"Maximum relative energy drift: {np.max(np.abs(energy/energy[0] - 1.0)):.3e}")
    print(f"Maximum relative angular momentum drift: {np.max(np.abs(angular_momentum/angular_momentum[0] - 1.0)):.3e}")
    print(f"Maximum relative pericenter drift: {np.max(np.abs(r_peri_num/r_peri - 1.0)):.3e}\n")

    # conservation over ten orbits
    fig, ax = plt.subplots(figsize=(5.1, 3.25))
    ax.plot(times/period, energy/energy[0] - 1.0,
            color="#0066FF", linewidth=1.35, label="Energy")
    ax.plot(times/period, angular_momentum/angular_momentum[0] - 1.0,
            color="#EF007D", linewidth=1.35, label="Angular momentum")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.9)
    ax.set_xlabel(r"$t/T$", labelpad=8)
    ax.set_ylabel("Fractional drift", labelpad=9)
    ax.set_xlim(0.0, 10.0)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    style_axes(ax)
    ax.legend(frameon=False, loc="lower left", fontsize=11)
    fig.tight_layout(pad=0.3)
    fig.savefig("results/p2a-conservation.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    # pericenter inferred from orbital energy and angular momentum
    fig, ax = plt.subplots(figsize=(5.1, 3.25))
    ax.plot(times/period, r_peri_num/r_peri - 1.0,
            color="#0066FF", linewidth=1.35)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.9)
    ax.set_xlabel(r"$t/T$", labelpad=8)
    ax.set_ylabel(r"$r_{\rm peri}/r_s - 1$", labelpad=9)
    ax.set_xlim(0.0, 10.0)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    style_axes(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig("results/p2a-pericenter.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# part (b): BH orbit with A = B = 1
def part_b():

    times, states, t_hit = orbit(1.0, 1.0, delta=1.0e-8)
    x, y_pos = states[:, 0], states[:, 1]
    r = np.hypot(x, y_pos)

    print("PART (b): A = B = 1")
    print(f"Accepted steps: {len(times) - 1}")
    print(f"Time to r_s: {t_hit:.8f} code units = {t_hit*time_Myr:.6f} Myr\n")

    # orbital trajectory in the xy plane
    fig, ax = plt.subplots(figsize=(5.1, 2.45))
    ax.plot(x, y_pos, color="#0066FF", linewidth=1.25)
    ax.plot(0.0, 0.0, "o", color="black", markersize=3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$ ($100\,\mathrm{pc}$)", labelpad=8)
    ax.set_ylabel(r"$y$ ($100\,\mathrm{pc}$)", labelpad=9)
    ax.set_xlim(x.min() - 0.012, x.max() + 0.012)
    ax.set_ylim(y_pos.min() - 0.012, y_pos.max() + 0.012)
    style_axes(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig("results/p2b-orbit.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    # radial evolution and the Schwarzschild-radius threshold
    fig, ax = plt.subplots(figsize=(5.1, 3.25))
    ax.plot(times*time_Myr, np.log10(r), color="#0066FF", linewidth=1.1)
    ax.axhline(np.log10(r_s), color="black", linestyle=":",
               linewidth=1.15, label=r"$r_s=10^{-7}$")
    ax.set_xlabel(r"$t$ (Myr)", labelpad=8)
    ax.set_ylabel(r"$\log_{10}(r)$", labelpad=9)
    ax.set_xlim(0.0, times[-1]*time_Myr)
    ax.set_ylim(np.log10(r_s) - 0.12, 0.15)
    ax.set_yticks(np.arange(-7, 1))
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=11)
    fig.tight_layout(pad=0.3)
    fig.savefig("results/p2b-radius.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# part (c): scan A and B in [0.5, 10]
def part_c():

    values = [0.5, 1.0, 2.0, 5.0, 10.0]
    results = {}
    ratios, times_myr, cutoffs_myr, parameters = [], [], [], []

    print("PART (c): PARAMETER SCAN")
    for A in values:
        for B in values:
            times, states, t_hit = orbit(A, B, delta=1.0e-7,
                                         t_end=np.inf, max_steps=None)
            results[(A, B)] = t_hit
            ratios.append(B/A)
            times_myr.append(t_hit*time_Myr)
            cutoffs_myr.append(times[-1]*time_Myr)
            parameters.append(A)
            label = (f"{t_hit*time_Myr:.5f} Myr" if np.isfinite(t_hit)
                     else f"not reached by {times[-1]*time_Myr:.3f} Myr")
            print(f"A = {A:4.1f}, B = {B:4.1f}, B/A = {B/A:6.3f}: {label}")

    ratios = np.array(ratios)
    times_myr = np.array(times_myr)
    cutoffs_myr = np.array(cutoffs_myr)
    parameters = np.array(parameters)
    reached = np.isfinite(times_myr)

    # one curve per A; points across each curve have different B
    colors = ["#0066FF", "#EF007D", "#00A86B", "#E68A00", "#7D3CFF"]
    fig, ax = plt.subplots(figsize=(5.1, 3.55))

    for A, color in zip(values, colors):
        selected = parameters == A
        x = ratios[selected]
        y = times_myr[selected]
        ax.plot(x, y, "o-", markersize=4, linewidth=1.35,
                color=color, label=rf"$A={A:g}$")

        # if an orbit has not reached r_s, show only its lower bound
        incomplete = selected & ~reached
        if np.any(incomplete):
            ax.plot(ratios[incomplete], cutoffs_myr[incomplete], "^",
                    markersize=6, color=color, markerfacecolor="none")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$B/A$", labelpad=8)
    ax.set_ylabel(r"Time to $r_s$ (Myr)", labelpad=9)
    ax.set_xlim(min(ratios)*0.9, max(ratios)*1.1)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=10, loc="upper left",
              ncol=2, columnspacing=0.8, handlelength=1.8,
              labelspacing=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig("results/p2c-parameter-scan.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print()
    return results


# part (d): vary initial velocity and compare A, B at fixed B/A
def part_d(results):

    fractions = [0.6, 0.7, 0.8, 0.9, 1.0]
    hit_times = []
    cutoffs = []

    print("PART (d): INITIAL VELOCITY")
    for fraction in fractions:
        times, _, t_hit = orbit(1.0, 1.0, v_fraction=fraction,
                            delta=1.0e-7, t_end=np.inf, max_steps=None)
        hit_times.append(t_hit*time_Myr)
        cutoffs.append(times[-1]*time_Myr)
        label = (f"{t_hit*time_Myr:.6f} Myr" if np.isfinite(t_hit)
                 else "not reached")
        print(f"v0/v_circ = {fraction:.1f}: {label}")

    # dependence on the initial tangential speed
    hit_times = np.array(hit_times)
    cutoffs = np.array(cutoffs)
    fractions = np.array(fractions)
    reached = np.isfinite(hit_times)

    fig, ax = plt.subplots(figsize=(5.1, 3.25))
    ax.plot(fractions, hit_times, "o-", markersize=4.5,
            linewidth=1.4, color="#0066FF")
    if np.any(~reached):
        ax.plot(fractions[~reached], cutoffs[~reached], "^",
                markersize=6, color="#EF007D", markerfacecolor="none",
                label="Lower bound")
        ax.legend(frameon=False, fontsize=11)
    ax.set_xlabel(r"$v_0/v_{\rm circ}$", labelpad=8)
    ax.set_ylabel(r"Time to $r_s$ (Myr)", labelpad=9)
    ax.set_xlim(min(fractions) - 0.012, max(fractions) + 0.012)
    ax.margins(y=0.08)
    style_axes(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig("results/p2d-initial-velocity.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    values = [0.5, 1.0, 2.0, 5.0, 10.0]
    fixed_ratio = [results[(A, A)]*time_Myr for A in values]

    print("\nPART (d): FIXED B/A = 1")
    for A, t_hit in zip(values, fixed_ratio):
        label = f"{t_hit:.6f} Myr" if np.isfinite(t_hit) else "not reached"
        print(f"A = B = {A:.1f}: {label}")

    # changing A and B together while holding B/A = 1
    fig, ax = plt.subplots(figsize=(5.1, 3.25))
    ax.plot(values, fixed_ratio, "o-", markersize=4.5,
            linewidth=1.4, color="#0066FF")
    ax.set_xlabel(r"$A$ ($B=A$)", labelpad=8)
    ax.set_ylabel(r"Time to $r_s$ (Myr)", labelpad=9)
    ax.set_xlim(min(values) - 0.3, max(values) + 0.3)
    ax.margins(y=0.08)
    style_axes(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig("results/p2d-fixed-ratio.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main():

    import os
    os.makedirs("results", exist_ok=True)

    print(f"BH mass implied by r_s: {M_phys/M_sun:.4e} solar masses")
    print(f"One code time unit: {time_Myr:.6f} Myr\n")

    part_a()
    part_b()
    results = part_c()
    part_d(results)


if __name__ == "__main__":
    main()
