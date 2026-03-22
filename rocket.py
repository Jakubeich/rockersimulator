"""
Vehicle model — represents a single stage that can fly independently.

Before separation the booster carries the upper stage as extra payload mass.
After separation both vehicles are stepped independently.
"""

from __future__ import annotations

import math
from enum import Enum, auto

from config import EnvironmentConfig, SimulationConfig, StageConfig
from integrators import get_integrator
from physics import (
    RocketState,
    air_density,
    dynamic_pressure,
    gravity_acceleration,
    moment_of_inertia,
)
from utils import clamp, magnitude, vec2


class EngineState(Enum):
    IDLE = auto()
    BURNING = auto()
    CUTOFF = auto()
    SHUTDOWN = auto()


class FlightPhase(Enum):
    PRE_LAUNCH = auto()
    POWERED_ASCENT = auto()
    COAST = auto()
    DESCENT = auto()
    POWERED_DESCENT = auto()
    LANDED = auto()
    CRASHED = auto()


class Vehicle:
    """A single rocket stage with its own physics state and controls."""

    def __init__(
        self,
        stage_cfg: StageConfig,
        env_cfg: EnvironmentConfig,
        sim_cfg: SimulationConfig,
        vehicle_id: str = "vehicle",
    ) -> None:
        self.cfg = stage_cfg
        self._ecfg = env_cfg
        self._integrate = get_integrator(sim_cfg.integration_method)
        self.vehicle_id = vehicle_id

        self.state = RocketState(
            x=0.0, y=0.0, vx=0.0, vy=0.0,
            theta=math.pi / 2, omega=0.0,
            fuel=stage_cfg.fuel_mass,
        )

        self.throttle: float = 0.0
        self.gimbal_angle: float = 0.0
        self.active_engines: int = stage_cfg.num_engines

        self.engine_state = EngineState.IDLE
        self.flight_phase = FlightPhase.PRE_LAUNCH
        self.is_active: bool = True
        self.has_left_ground: bool = False

        # Extra payload mass (e.g. upper stage riding on booster)
        self.payload_mass: float = 0.0

        self.time: float = 0.0
        self.max_altitude: float = 0.0
        self.max_speed: float = 0.0
        self.max_dynamic_pressure: float = 0.0

    # ----- Derived quantities --------------------------------------------

    @property
    def total_mass(self) -> float:
        return self.cfg.dry_mass + max(self.state.fuel, 0.0) + self.payload_mass

    @property
    def has_fuel(self) -> bool:
        return self.state.fuel > 0

    @property
    def altitude(self) -> float:
        return self.state.y

    @property
    def position(self):
        return vec2(self.state.x, self.state.y)

    @property
    def velocity(self):
        return vec2(self.state.vx, self.state.vy)

    @property
    def speed(self) -> float:
        return magnitude(self.velocity)

    @property
    def vertical_speed(self) -> float:
        return self.state.vy

    @property
    def horizontal_speed(self) -> float:
        return self.state.vx

    @property
    def current_thrust(self) -> float:
        if self.engine_state == EngineState.BURNING:
            engine_fraction = self.active_engines / max(self.cfg.num_engines, 1)
            return self.throttle * self.cfg.max_thrust * engine_fraction
        return 0.0

    @property
    def effective_max_thrust(self) -> float:
        return self.cfg.max_thrust * self.active_engines / max(self.cfg.num_engines, 1)

    @property
    def fuel_fraction(self) -> float:
        return self.state.fuel / self.cfg.fuel_mass if self.cfg.fuel_mass > 0 else 0.0

    @property
    def reserve_fuel(self) -> float:
        return self.cfg.fuel_mass * self.cfg.fuel_reserve_fraction

    @property
    def q(self) -> float:
        return dynamic_pressure(self.velocity, self.altitude, self._ecfg)

    @property
    def g_local(self) -> float:
        return gravity_acceleration(self.altitude, self._ecfg)

    @property
    def moi(self) -> float:
        return moment_of_inertia(self.total_mass, self.cfg)

    @property
    def rho(self) -> float:
        return air_density(self.altitude, self._ecfg)

    @property
    def is_terminated(self) -> bool:
        return self.flight_phase in (FlightPhase.CRASHED, FlightPhase.LANDED)

    # ----- Controls ------------------------------------------------------

    def set_throttle(self, value: float) -> None:
        self.throttle = clamp(value, 0.0, 1.0)
        if self.throttle > 0 and self.has_fuel:
            if self.engine_state in (EngineState.IDLE, EngineState.SHUTDOWN):
                self.engine_state = EngineState.BURNING

    def set_gimbal(self, value: float) -> None:
        self.gimbal_angle = clamp(value, -self.cfg.max_gimbal_angle, self.cfg.max_gimbal_angle)

    def adjust_gimbal(self, delta: float, dt: float) -> None:
        max_delta = self.cfg.gimbal_rate * dt
        clamped_delta = clamp(delta, -max_delta, max_delta)
        self.set_gimbal(self.gimbal_angle + clamped_delta)

    def shutdown_engine(self) -> None:
        self.throttle = 0.0
        if self.engine_state == EngineState.BURNING:
            self.engine_state = EngineState.SHUTDOWN

    def ignite(self, throttle: float = 1.0, engines: int | None = None) -> None:
        if engines is not None:
            self.active_engines = min(engines, self.cfg.num_engines)
        self.set_throttle(throttle)

    # ----- Simulation step -----------------------------------------------

    def step(self, dt: float) -> None:
        if not self.is_active or self.is_terminated:
            return

        effective_throttle = self.throttle if self.engine_state == EngineState.BURNING else 0.0
        # Scale burn rate by active engine fraction
        engine_frac = self.active_engines / max(self.cfg.num_engines, 1)

        # Temporarily adjust config for integration (engine fraction)
        orig_max_thrust = self.cfg.max_thrust
        orig_burn_rate = self.cfg.burn_rate
        self.cfg.max_thrust = orig_max_thrust * engine_frac
        self.cfg.burn_rate = orig_burn_rate * engine_frac

        # Also account for payload mass by adjusting dry_mass temporarily
        orig_dry = self.cfg.dry_mass
        self.cfg.dry_mass = orig_dry + self.payload_mass

        prev_state = self.state
        self.state = self._integrate(
            self.state, dt, effective_throttle, self.gimbal_angle,
            self.cfg, self._ecfg,
        )

        # Restore config
        self.cfg.max_thrust = orig_max_thrust
        self.cfg.burn_rate = orig_burn_rate
        self.cfg.dry_mass = orig_dry

        # Fuel depletion
        if prev_state.fuel > 0 and self.state.fuel <= 0:
            self.state.fuel = 0.0
            self.engine_state = EngineState.CUTOFF
            self.throttle = 0.0

        # Track airborne
        if self.state.y > 5.0:
            self.has_left_ground = True

        # Ground collision
        if self.state.y < 0:
            self.state.y = 0.0
            if self.has_left_ground:
                impact_speed = magnitude(vec2(prev_state.vx, prev_state.vy))
                self.state.vx = 0.0
                self.state.vy = 0.0
                self.state.omega = 0.0
                if impact_speed > 30.0:
                    self.flight_phase = FlightPhase.CRASHED
                else:
                    self.flight_phase = FlightPhase.LANDED
            else:
                self.state.vy = max(self.state.vy, 0.0)

        self._update_flight_phase()

        # Normalize theta to [-π, π] — sin/cos periodic, prevents visual spin
        self.state.theta = math.atan2(
            math.sin(self.state.theta), math.cos(self.state.theta))

        self.time += dt
        self.max_altitude = max(self.max_altitude, self.altitude)
        self.max_speed = max(self.max_speed, self.speed)
        self.max_dynamic_pressure = max(self.max_dynamic_pressure, self.q)

    def _update_flight_phase(self) -> None:
        if self.flight_phase in (FlightPhase.CRASHED, FlightPhase.LANDED):
            return

        on_ground = self.altitude <= 0 and self.speed < 1.0
        if on_ground and not self.has_left_ground:
            if self.engine_state == EngineState.BURNING:
                self.flight_phase = FlightPhase.POWERED_ASCENT
            else:
                self.flight_phase = FlightPhase.PRE_LAUNCH
            return

        if self.engine_state == EngineState.BURNING:
            if self.vertical_speed >= 0:
                self.flight_phase = FlightPhase.POWERED_ASCENT
            else:
                self.flight_phase = FlightPhase.POWERED_DESCENT
        elif self.vertical_speed > 0:
            self.flight_phase = FlightPhase.COAST
        else:
            self.flight_phase = FlightPhase.DESCENT

    # ----- State transfer (for separation) --------------------------------

    def copy_state_from(self, other: "Vehicle") -> None:
        """Copy kinematic state from another vehicle (at separation)."""
        self.state = RocketState(
            x=other.state.x, y=other.state.y,
            vx=other.state.vx, vy=other.state.vy,
            theta=other.state.theta, omega=other.state.omega,
            fuel=self.cfg.fuel_mass,
        )
        self.time = other.time
        self.has_left_ground = True
        self.is_active = True
