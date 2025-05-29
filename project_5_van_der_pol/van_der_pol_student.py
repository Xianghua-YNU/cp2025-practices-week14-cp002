import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Tuple

def van_der_pol_ode(t, state, mu=1.0, omega=1.0):
    x, v = state
    return np.array([v, mu * (1 - x**2) * v - omega**2 * x])

def solve_ode(ode_func, initial_state, t_span, dt, **kwargs):
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
    sol = solve_ivp(ode_func, t_span, initial_state, t_eval=t_eval, args=tuple(kwargs.values()), method='RK45')
    return sol.t, sol.y.T

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(t, states[:, 0], label='Position x(t)')
    plt.plot(t, states[:, 1], label='Velocity v(t)')
    plt.xlabel('Time t')
    plt.ylabel('State Variables')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1])
    plt.xlabel('Position x')
    plt.ylabel('Velocity v')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def analyze_limit_cycle(states: np.ndarray, dt: float) -> Tuple[float, float]:
    skip = int(len(states) * 0.5)
    x = states[skip:, 0]
    t = np.arange(len(x)) * dt
    peaks = []
    for i in range(1, len(x)-1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            peaks.append((t[i], x[i]))
    if len(peaks) < 2:
        return np.nan, np.nan
    times, values = zip(*peaks)
    amplitude = np.mean(values)
    periods = np.diff(times)
    period = np.mean(periods) if len(periods) > 0 else np.nan
    return amplitude, period

def estimate_settling_time(states: np.ndarray, t: np.ndarray, tol: float = 0.01, min_peaks: int = 5) -> float:
    x = states[:, 0]
    peaks = []
    peak_times = []
    for i in range(1, len(x)-1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            peaks.append(x[i])
            peak_times.append(t[i])
    if len(peaks) < min_peaks + 1:
        return np.nan
    recent_peaks = peaks[-min_peaks:]
    if max(recent_peaks) - min(recent_peaks) < tol:
        return peak_times[-min_peaks]
    return np.nan

def main():
    mu_values = [1.0, 2.0, 4.0]
    omega = 1.0
    t_span = (0, 100)
    dt = 0.01
    initial_state = np.array([1.0, 0.0])

    print(" μ值  |   振幅   |   周期   |  稳态时间  ")
    print("----------------------------------")

    for mu in mu_values:
        t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        plot_time_evolution(t, states, f'Time Evolution (μ={mu})')
        plot_phase_space(states, f'Phase Space (μ={mu})')

        amplitude, period = analyze_limit_cycle(states, dt)
        settling_time = estimate_settling_time(states, t)

        print(f" {mu:.1f}  |  {amplitude:.3f}  |  {period:.3f}  |  {settling_time:.2f}")

if __name__ == "__main__":
    main()
