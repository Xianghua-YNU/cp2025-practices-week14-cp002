import matplotlib
matplotlib.use('Agg')  # 必须在 import pyplot 之前设置
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import os

def lotka_volterra_system(state: np.ndarray, t: float, alpha: float, beta: float,
                          gamma: float, delta: float) -> np.ndarray:
    """
    Right-hand side of the Lotka-Volterra equations.
    Args:
        state: [x, y] current state vector
        t: time (not used explicitly)
        alpha, beta, gamma, delta: model parameters
    Returns:
        Derivative vector [dx/dt, dy/dt]
    """
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = gamma * x * y - delta * y
    return np.array([dxdt, dydt])

def runge_kutta_4(f, y0: np.ndarray, t_span: Tuple[float, float],
                  dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    for i in range(n_steps - 1):
        h = dt
        yi = y[i]
        ti = t[i]
        k1 = h * f(yi, ti, *args)
        k2 = h * f(yi + k1/2, ti + h/2, *args)
        k3 = h * f(yi + k2/2, ti + h/2, *args)
        k4 = h * f(yi + k3, ti + h, *args)
        y[i+1] = yi + (k1 + 2*k2 + 2*k3 + k4) / 6
    return t, y

def euler_method(f, y0: np.ndarray, t_span: Tuple[float, float],
                 dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    for i in range(n_steps - 1):
        y[i+1] = y[i] + dt * f(y[i], t[i], *args)
    return t, y

def improved_euler_method(f, y0: np.ndarray, t_span: Tuple[float, float],
                         dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    for i in range(n_steps - 1):
        h = dt
        yi = y[i]
        ti = t[i]
        k1 = h * f(yi, ti, *args)
        k2 = h * f(yi + k1, ti + h, *args)
        y[i+1] = yi + (k1 + k2) / 2
    return t, y

def solve_lotka_volterra(alpha: float, beta: float, gamma: float, delta: float,
                        x0: float, y0: float, t_span: Tuple[float, float],
                        dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y0_vec = np.array([x0, y0])
    t, solution = runge_kutta_4(lotka_volterra_system, y0_vec, t_span, dt,
                               alpha, beta, gamma, delta)
    x = solution[:, 0]
    y = solution[:, 1]
    return t, x, y

def compare_methods(alpha: float, beta: float, gamma: float, delta: float,
                   x0: float, y0: float, t_span: Tuple[float, float],
                   dt: float) -> dict:
    y0_vec = np.array([x0, y0])
    args = (alpha, beta, gamma, delta)
    t_euler, sol_euler = euler_method(lotka_volterra_system, y0_vec, t_span, dt, *args)
    t_ie, sol_ie = improved_euler_method(lotka_volterra_system, y0_vec, t_span, dt, *args)
    t_rk4, sol_rk4 = runge_kutta_4(lotka_volterra_system, y0_vec, t_span, dt, *args)
    return {
        'euler': {'t': t_euler, 'x': sol_euler[:, 0], 'y': sol_euler[:, 1]},
        'improved_euler': {'t': t_ie, 'x': sol_ie[:, 0], 'y': sol_ie[:, 1]},
        'rk4': {'t': t_rk4, 'x': sol_rk4[:, 0], 'y': sol_rk4[:, 1]}
    }

def plot_population_dynamics(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                           title: str = "Lotka-Volterra Population Dynamics",
                           save_dir: str = r"C:\Users\31025\OneDrive\桌面\t") -> None:
    """
    Plot population dynamics and save to specified directory.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, x, 'b-', label='Prey (x)', linewidth=2)
    plt.plot(t, y, 'r-', label='Predator (y)', linewidth=2)
    plt.xlabel('Time t')
    plt.ylabel('Population')
    plt.title('Population vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(x, y, 'g-', linewidth=2)
    plt.plot(x[0], y[0], 'go', markersize=8, label='Start')
    plt.xlabel('Prey Population (x)')
    plt.ylabel('Predator Population (y)')
    plt.title('Phase Space Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'population_dynamics.png'))
    plt.close()

def plot_method_comparison(results: dict, save_dir: str = r"C:\Users\31025\OneDrive\桌面\t") -> None:
    """
    Plot comparison of different numerical methods and save to specified directory.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.figure(figsize=(15, 10))
    methods = ['euler', 'improved_euler', 'rk4']
    method_names = ['Euler Method', 'Improved Euler', '4th-order Runge-Kutta']
    colors = ['blue', 'orange', 'green']
    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        plt.subplot(2, 3, i+1)
        t = results[method]['t']
        x = results[method]['x']
        y = results[method]['y']
        plt.plot(t, x, color=color, linestyle='-', label='Prey', linewidth=2)
        plt.plot(t, y, color=color, linestyle='--', label='Predator', linewidth=2)
        plt.xlabel('Time t')
        plt.ylabel('Population')
        plt.title(f'{name} - Time Series')
        plt.legend()
        plt.grid(True, alpha=0.3)
    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        plt.subplot(2, 3, i+4)
        x = results[method]['x']
        y = results[method]['y']
        plt.plot(x, y, color=color, linewidth=2)
        plt.plot(x[0], y[0], 'o', color=color, markersize=6)
        plt.xlabel('Prey Population (x)')
        plt.ylabel('Predator Population (y)')
        plt.title(f'{name} - Phase Space')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'method_comparison.png'))
    plt.close()

def analyze_parameters(save_dir: str = r"C:\Users\31025\OneDrive\桌面\t") -> None:
    """
    Analyze the effect of different parameters and save the figure.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2.0
    t_span = (0, 30)
    dt = 0.01
    plt.figure(figsize=(15, 10))
    initial_conditions = [(1, 1), (2, 2), (3, 1), (1, 3)]
    for i, (x0, y0) in enumerate(initial_conditions):
        t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)
        plt.subplot(2, 2, 1)
        plt.plot(t, x, label=f'x0={x0}, y0={y0}', linewidth=2)
        plt.subplot(2, 2, 2)
        plt.plot(t, y, label=f'x0={x0}, y0={y0}', linewidth=2)
        plt.subplot(2, 2, 3)
        plt.plot(x, y, label=f'x0={x0}, y0={y0}', linewidth=2)
    plt.subplot(2, 2, 1)
    plt.xlabel('Time t')
    plt.ylabel('Prey Population (x)')
    plt.title('Prey Population under Different Initial Conditions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 2)
    plt.xlabel('Time t')
    plt.ylabel('Predator Population (y)')
    plt.title('Predator Population under Different Initial Conditions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 3)
    plt.xlabel('Prey Population (x)')
    plt.ylabel('Predator Population (y)')
    plt.title('Phase Space Trajectories for Different Initial Conditions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    x0, y0 = 2, 2
    t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)
    H = gamma * x + beta * y - delta * np.log(x) - alpha * np.log(y)
    plt.subplot(2, 2, 4)
    plt.plot(t, H, 'purple', linewidth=2)
    plt.xlabel('Time t')
    plt.ylabel('Conserved Quantity H')
    plt.title('Energy Conservation Test')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'parameter_analysis.png'))
    plt.close()

def main():
    """
    Main function: Demonstrate full analysis of the Lotka-Volterra model.
    """
    save_dir = r"C:\Users\31025\OneDrive\桌面\t"
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2.0
    x0, y0 = 2.0, 2.0
    t_span = (0, 30)
    dt = 0.01
    print("=== Lotka-Volterra Predator-Prey Model Analysis ===")
    print(f"Parameters: α={alpha}, β={beta}, γ={gamma}, δ={delta}")
    print(f"Initial conditions: x0={x0}, y0={y0}")
    print(f"Time range: {t_span}, Step size: {dt}")
    print("\n1. Solving with 4th-order Runge-Kutta method...")
    t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)
    plot_population_dynamics(t, x, y, title="Lotka-Volterra Population Dynamics", save_dir=save_dir)
    print("\n2. Comparing different numerical methods...")
    results = compare_methods(alpha, beta, gamma, delta, x0, y0, t_span, dt)
    plot_method_comparison(results, save_dir=save_dir)
    print("\n3. Analyzing parameter effects...")
    analyze_parameters(save_dir=save_dir)
    print("\n4. Numerical results statistics:")
    print(f"Prey population range: [{np.min(x):.3f}, {np.max(x):.3f}]")
    print(f"Predator population range: [{np.min(y):.3f}, {np.max(y):.3f}]")
    print(f"Average prey population: {np.mean(x):.3f}")
    print(f"Average predator population: {np.mean(y):.3f}")
    from scipy.signal import find_peaks
    peaks_x, _ = find_peaks(x, height=np.mean(x))
    if len(peaks_x) > 1:
        period = np.mean(np.diff(t[peaks_x]))
        print(f"Estimated period: {period:.3f}")

if __name__ == "__main__":
    main()
