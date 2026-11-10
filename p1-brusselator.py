import numpy as np
import matplotlib.pyplot as plt
import os

# Plotting

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


# general Brusselator with a = 1 and b = 3
def brusselator(t, r):
    x, y = r
    dx = 1.0 - 4.0*x + x**2*y
    dy = 3.0*x - x**2*y
    return np.array([dx, dy])


# modified midpoint estimate over H using n staggered midpoint steps
def modified_midpoint(f, r, t, H, n):

    h = H/n

    # stagger the two estimates by half a step
    r1 = r + 0.5*h*f(t, r) # midpoint estimation of one step
    r2 = r + h*f(t + 0.5*h, r1) # full step using derivative at midpoint estimation

    # staggered updates
    for i in range(1, n):
        r1 = r1 + h*f(t + i*h, r2)
        r2 = r2 + h*f(t + (i + 0.5)*h, r1)

    return r2


# Bulirsch-Stoer step with Richardson extrapolation
# split the interval recursively if n = 8 does not reach delta for prior interval
# times and states store the endpoints of accepted intervals
def step(f, r, t, H, delta, times, states):

    previous = []

    for n in range(1, 9):

        # first estimate of the current Richardson row
        current = [modified_midpoint(f, r, t, H, n)]

        if not np.all(np.isfinite(current[0])):
            break

        # Richardson extrapolation
        for m in range(1, n):
            correction = (current[m-1] - previous[m-1])/((n/(n-m))**2 - 1.0)
            current.append(current[m-1] + correction)

        if not np.all(np.isfinite(current[-1])):
            break

        # compare the last correction with the accuracy delta*H
        # need at least two approximations for error estimate
        # both concentrations need to satisfy the tolerance
        if n > 1 and np.all(np.abs(correction) <= delta*H):
            times.append(t + H)
            states.append(current[-1].copy())
            return current[-1]

        previous = current

    # if the extrapolation did not converge, solve each half separately
    if t + 0.5*H == t or t + H == t + 0.5*H:
        raise RuntimeError("Time interval too small to subdivide")

    # recursion; call step again but at half the first total interval
    r_half = step(f, r, t, 0.5*H, delta, times, states) # first half
    return step(f, r_half, t + 0.5*H, 0.5*H, delta, times, states) # second half


def main():

    # initial conditions and accuracy requirement
    delta = 1.0e-10
    t0 = 0.0
    t_end = 20.0
    H = t_end - t0
    r0 = np.array([0.0, 0.0])

    times = [t0]
    states = [r0.copy()]

    # begin with one interval
    with np.errstate(over="ignore", invalid="ignore"):
        step(brusselator, r0, t0, H, delta, times, states)

    times = np.array(times)
    states = np.array(states)

    print(f"Accepted intervals: {len(times)-1}")
    print(f"Smallest interval: {np.diff(times).min():.6g}")
    print(f"Largest interval:  {np.diff(times).max():.6g}")
    print(f"Final values: x(20) = {states[-1,0]:.12f}, y(20) = {states[-1,1]:.12f}")

    # both concentrations plotted. dots are the adaptive interval boundaries
    os.makedirs("results", exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.1, 3.5))
    ax.plot(times, states[:, 0], "o-", markersize=3.0, linewidth=1.4, color="#0066FF", label=r"$x(t)$")
    ax.plot(times, states[:, 1], "o-", markersize=3.0, linewidth=1.4, color="#EF007D", label=r"$y(t)$")

    ax.set_xlabel(r"$t$", labelpad=8)
    ax.set_ylabel("Concentration", labelpad=9)
    ax.set_xlim(0, 20)
    ax.set_xticks(np.arange(0, 21, 5))

    style_axes(ax)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig("results/newman-8-18.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
