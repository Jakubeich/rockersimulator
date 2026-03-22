"""
Numerical integration methods for the rocket state vector.

Each integrator has the same signature so they can be swapped at runtime
via SimulationConfig.integration_method.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from config import EnvironmentConfig, StageConfig
from physics import RocketState, state_derivative

# Type alias for the derivative function
DerivFn = Callable[
    [RocketState, float, float, StageConfig, EnvironmentConfig],
    np.ndarray,
]


def _clamp_fuel(state: RocketState) -> None:
    """Ensure fuel never goes negative after an integration step."""
    if state.fuel < 0:
        state.fuel = 0.0


# ---------------------------------------------------------------------------
# Explicit (forward) Euler – baseline, least accurate
# ---------------------------------------------------------------------------

def euler_step(
    state: RocketState,
    dt: float,
    throttle: float,
    gimbal: float,
    rcfg: StageConfig,
    ecfg: EnvironmentConfig,
) -> RocketState:
    arr = state.to_array()
    k = state_derivative(state, throttle, gimbal, rcfg, ecfg)
    new = RocketState.from_array(arr + dt * k)
    _clamp_fuel(new)
    return new


# ---------------------------------------------------------------------------
# Semi-implicit (symplectic) Euler – better energy conservation
# ---------------------------------------------------------------------------

def semi_implicit_euler_step(
    state: RocketState,
    dt: float,
    throttle: float,
    gimbal: float,
    rcfg: StageConfig,
    ecfg: EnvironmentConfig,
) -> RocketState:
    k = state_derivative(state, throttle, gimbal, rcfg, ecfg)

    # Update velocities first (using accelerations from current state)
    new_vx = state.vx + k[2] * dt
    new_vy = state.vy + k[3] * dt
    new_omega = state.omega + k[5] * dt

    # Then update positions using the *new* velocities
    new_x = state.x + new_vx * dt
    new_y = state.y + new_vy * dt
    new_theta = state.theta + new_omega * dt
    new_fuel = state.fuel + k[6] * dt

    result = RocketState(
        x=new_x, y=new_y,
        vx=new_vx, vy=new_vy,
        theta=new_theta, omega=new_omega,
        fuel=new_fuel,
    )
    _clamp_fuel(result)
    return result


# ---------------------------------------------------------------------------
# Classic 4th-order Runge–Kutta
# ---------------------------------------------------------------------------

def rk4_step(
    state: RocketState,
    dt: float,
    throttle: float,
    gimbal: float,
    rcfg: StageConfig,
    ecfg: EnvironmentConfig,
) -> RocketState:
    arr = state.to_array()

    k1 = state_derivative(state, throttle, gimbal, rcfg, ecfg)
    k2 = state_derivative(RocketState.from_array(arr + 0.5 * dt * k1), throttle, gimbal, rcfg, ecfg)
    k3 = state_derivative(RocketState.from_array(arr + 0.5 * dt * k2), throttle, gimbal, rcfg, ecfg)
    k4 = state_derivative(RocketState.from_array(arr + dt * k3), throttle, gimbal, rcfg, ecfg)

    new_arr = arr + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    result = RocketState.from_array(new_arr)
    _clamp_fuel(result)
    return result


# ---------------------------------------------------------------------------
# Registry – maps config string to callable
# ---------------------------------------------------------------------------

INTEGRATORS: dict[str, Callable] = {
    "euler": euler_step,
    "semi_implicit_euler": semi_implicit_euler_step,
    "rk4": rk4_step,
}


def get_integrator(name: str) -> Callable:
    try:
        return INTEGRATORS[name]
    except KeyError:
        raise ValueError(
            f"Unknown integrator '{name}'. Choose from: {list(INTEGRATORS)}"
        )
