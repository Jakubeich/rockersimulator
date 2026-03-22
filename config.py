"""
Centralised configuration for the rocket simulator.

All physical quantities use SI units (m, s, kg, N, rad).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Stage configuration (one physical stage)
# ---------------------------------------------------------------------------

@dataclass
class StageConfig:
    """Physical properties of a single rocket stage."""

    dry_mass: float = 2_500.0
    fuel_mass: float = 20_000.0
    max_thrust: float = 300_000.0
    isp: float = 280.0
    burn_rate: float | None = None        # derived from thrust & Isp if None
    num_engines: int = 1
    min_throttle: float = 0.3             # deep-throttle limit
    fuel_reserve_fraction: float = 0.0    # fraction reserved for landing
    length: float = 30.0
    diameter: float = 2.0
    cd_axial: float = 0.3
    cd_lateral: float = 1.2
    reference_area: float | None = None
    moment_of_inertia_factor: float = 1 / 12
    max_gimbal_angle: float = 0.12        # rad (~7°)
    gimbal_rate: float = 0.5              # rad/s

    def __post_init__(self) -> None:
        if self.reference_area is None:
            self.reference_area = math.pi * (self.diameter / 2) ** 2
        if self.burn_rate is None:
            g0 = 9.80665
            self.burn_rate = self.max_thrust / (self.isp * g0)


# ---------------------------------------------------------------------------
# Vehicle = stack of stages
# ---------------------------------------------------------------------------

@dataclass
class VehicleConfig:
    """Two-stage vehicle (booster + upper stage)."""

    booster: StageConfig = field(default_factory=lambda: StageConfig(
        dry_mass=4_000.0,
        fuel_mass=30_000.0,
        max_thrust=500_000.0,
        isp=282.0,
        num_engines=9,
        min_throttle=0.35,
        fuel_reserve_fraction=0.20,
        length=35.0,
        diameter=2.5,
        max_gimbal_angle=0.12,
        gimbal_rate=0.6,
    ))
    upper_stage: StageConfig = field(default_factory=lambda: StageConfig(
        dry_mass=1_200.0,
        fuel_mass=8_000.0,
        max_thrust=80_000.0,
        isp=348.0,
        num_engines=1,
        min_throttle=0.4,
        fuel_reserve_fraction=0.50,   # reserve for return/landing
        length=10.0,
        diameter=2.5,
        max_gimbal_angle=0.10,
        gimbal_rate=0.4,
    ))


# ---------------------------------------------------------------------------
# Landing pad
# ---------------------------------------------------------------------------

@dataclass
class PadConfig:
    """Launch and landing pad configuration."""
    launch_x: float = 0.0        # launch pad world-x (booster starts here)
    landing_x: float = -2000.0   # landing zone for booster (~2km from launch)
    pad_width: float = 30.0      # m — physical width of each pad
    pad_height: float = 2.0      # m — visual height


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentConfig:
    g0: float = 9.80665
    use_altitude_gravity: bool = True
    earth_radius: float = 6_371_000.0
    rho0: float = 1.225
    scale_height: float = 8_500.0
    karman_line: float = 100_000.0     # m — boundary of space


# ---------------------------------------------------------------------------
# Autopilot / PID
# ---------------------------------------------------------------------------

@dataclass
class PIDGains:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0


@dataclass
class AutopilotConfig:
    ascent_pid: PIDGains = field(default_factory=lambda: PIDGains(kp=0.5, ki=0.02, kd=0.8))
    landing_pid: PIDGains = field(default_factory=lambda: PIDGains(kp=0.6, ki=0.01, kd=1.0))
    pitch_program_rate: float = 0.003        # rad/s
    gravity_turn_start_speed: float = 100.0  # m/s


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    dt: float = 0.01
    max_duration: float = 3000.0
    integration_method: str = "rk4"
    paused_on_start: bool = False
    time_warp: float = 1.0


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

@dataclass
class RenderConfig:
    screen_width: int = 1280
    screen_height: int = 800
    fps: int = 60
    pixels_per_meter: float = 2.0
    min_zoom: float = 0.0001
    max_zoom: float = 20.0
    zoom_speed: float = 1.15

    show_trajectory: bool = True
    show_vectors: bool = True
    show_debug: bool = False

    bg_color: tuple[int, int, int] = (5, 5, 20)
    ground_color: tuple[int, int, int] = (40, 120, 40)
    sky_gradient_top: tuple[int, int, int] = (0, 0, 0)
    sky_gradient_bottom: tuple[int, int, int] = (30, 60, 120)
    booster_color: tuple[int, int, int] = (220, 220, 230)
    upper_color: tuple[int, int, int] = (200, 200, 255)
    flame_color: tuple[int, int, int] = (255, 140, 20)
    booster_trail_color: tuple[int, int, int] = (100, 180, 255)
    upper_trail_color: tuple[int, int, int] = (255, 180, 100)
    thrust_vector_color: tuple[int, int, int] = (255, 80, 30)
    velocity_vector_color: tuple[int, int, int] = (80, 255, 80)
    pad_color: tuple[int, int, int] = (80, 80, 80)
    text_color: tuple[int, int, int] = (220, 220, 220)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    pads: PadConfig = field(default_factory=PadConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    autopilot: AutopilotConfig = field(default_factory=AutopilotConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
