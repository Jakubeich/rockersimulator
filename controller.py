"""
Mission sequencer — two-stage flight with booster landing.

    Countdown → Ignition → Liftoff → Gravity turn →
    BECO → Separation →
    Booster: Flip → Boostback → Coast → Entry burn → Coast → Landing burn → Touchdown
    Upper:   Ignition → Gravity turn → SECO → Coast → Apogee → Descent
"""

from __future__ import annotations

import math
from enum import Enum, auto

from config import AutopilotConfig, PadConfig
from rocket import EngineState, FlightPhase, Vehicle
from utils import clamp, wrap_angle


# ---------------------------------------------------------------------------
# PID
# ---------------------------------------------------------------------------

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float,
                 output_min: float = -1.0, output_max: float = 1.0,
                 integral_max: float = 5.0) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_min, self.output_max = output_min, output_max
        self.integral_max = integral_max
        self._integral: float = 0.0
        self._prev_error: float | None = None

    def update(self, error: float, dt: float) -> float:
        if dt <= 0:
            return 0.0
        p = self.kp * error
        self._integral = clamp(self._integral + error * dt,
                               -self.integral_max, self.integral_max)
        i = self.ki * self._integral
        d = self.kd * (error - self._prev_error) / dt if self._prev_error is not None else 0.0
        self._prev_error = error
        return clamp(p + i + d, self.output_min, self.output_max)

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

class MissionPhase(Enum):
    COUNTDOWN = auto()
    IGNITION = auto()
    VERTICAL_ASCENT = auto()
    GRAVITY_TURN = auto()
    BECO = auto()
    SEPARATION = auto()
    # Post-separation: displayed phase tracks the most "interesting" vehicle
    BOOSTER_RETURN = auto()
    BOOSTER_LANDING = auto()
    BOOSTER_LANDED = auto()
    UPPER_BURN = auto()
    UPPER_COAST = auto()
    UPPER_RETURN = auto()
    UPPER_LANDING = auto()
    UPPER_LANDED = auto()
    MISSION_COMPLETE = auto()


class BoosterPhase(Enum):
    """Internal tracker for booster return sequence."""
    FLIP = auto()
    BOOSTBACK = auto()
    COAST = auto()
    ENTRY_BURN = auto()
    ENTRY_COAST = auto()
    LANDING_BURN = auto()
    DONE = auto()


class UpperPhase(Enum):
    """Internal tracker for upper stage."""
    WAIT_IGNITION = auto()
    BURNING = auto()
    COAST = auto()
    RETURN_FLIP = auto()
    RETURN_BURN = auto()
    RETURN_COAST = auto()
    LANDING = auto()
    DONE = auto()


PHASE_NAMES = {p: p.name.replace("_", " ").title() for p in MissionPhase}


# ---------------------------------------------------------------------------
# Mission sequencer
# ---------------------------------------------------------------------------

class MissionSequencer:
    COUNTDOWN_DURATION = 10.0
    IGNITION_LEAD = 3.0
    SEPARATION_DELAY = 2.0
    UPPER_IGNITION_DELAY = 3.0

    PITCH_PROGRAM_RATE = 0.003
    GRAVITY_TURN_START_SPEED = 100.0

    # Booster return
    FLIP_RATE = 3.0                # rad/s² angular accel for flip
    BOOSTBACK_ENGINES = 3
    ENTRY_BURN_ALT = 40_000.0      # m — start entry burn
    ENTRY_ENGINES = 3
    LANDING_BURN_SAFETY = 1.4
    RCS_TORQUE = 10.0              # rad/s² — simulated RCS for coast attitude

    UPPER_PITCH_RATE = 0.002
    THROTTLE_RAMP_RATE = 0.4

    def __init__(self, cfg: AutopilotConfig, pad_cfg: PadConfig | None = None) -> None:
        self.cfg = cfg
        self.phase = MissionPhase.COUNTDOWN
        self._bphase = BoosterPhase.FLIP
        self._uphase = UpperPhase.WAIT_IGNITION

        # Pad positions
        self._launch_x = pad_cfg.launch_x if pad_cfg else 0.0
        self._landing_x = pad_cfg.landing_x if pad_cfg else 0.0
        self._pad_height = pad_cfg.pad_height if pad_cfg else 2.0

        g = cfg.ascent_pid
        self.ascent_pid = PIDController(kp=g.kp, ki=g.ki, kd=g.kd)
        g2 = cfg.landing_pid
        self.landing_pid = PIDController(kp=g2.kp, ki=g2.ki, kd=g2.kd)
        self.upper_pid = PIDController(kp=g.kp, ki=g.ki, kd=g.kd)

        self._clock: float = -self.COUNTDOWN_DURATION
        self._target_pitch: float = math.pi / 2
        self._upper_target_pitch: float = math.pi / 2
        self._beco_time: float = 0.0
        self._sep_time: float = 0.0
        self._flip_target: float = 0.0
        self._upper_flip_target: float = 0.0

    @property
    def mission_time(self) -> float:
        return self._clock

    @property
    def phase_name(self) -> str:
        return PHASE_NAMES.get(self.phase, self.phase.name)

    @property
    def separated(self) -> bool:
        return self.phase.value >= MissionPhase.BOOSTER_RETURN.value

    # ----- Main update ---------------------------------------------------

    def update(self, booster: Vehicle, upper: Vehicle, dt: float) -> None:
        self._clock += dt

        if not self.separated:
            self._pre_separation(booster, upper, dt)
        else:
            if self._bphase != BoosterPhase.DONE:
                self._control_booster(booster, dt)
            if upper.is_active and self._uphase != UpperPhase.DONE:
                self._control_upper(upper, dt)

            # Update display phase
            if self._bphase == BoosterPhase.DONE and self._uphase == UpperPhase.DONE:
                self.phase = MissionPhase.MISSION_COMPLETE
            elif self._uphase == UpperPhase.LANDING:
                self.phase = MissionPhase.UPPER_LANDING
            elif self._uphase in (UpperPhase.RETURN_FLIP, UpperPhase.RETURN_BURN, UpperPhase.RETURN_COAST):
                self.phase = MissionPhase.UPPER_RETURN
            elif self._bphase in (BoosterPhase.LANDING_BURN,):
                self.phase = MissionPhase.BOOSTER_LANDING
            elif self._bphase == BoosterPhase.DONE:
                if self._uphase == UpperPhase.DONE:
                    self.phase = MissionPhase.MISSION_COMPLETE
                elif booster.flight_phase.name == "LANDED":
                    self.phase = MissionPhase.BOOSTER_LANDED
                else:
                    self.phase = MissionPhase.UPPER_COAST
            elif self._uphase == UpperPhase.BURNING:
                self.phase = MissionPhase.UPPER_BURN
            else:
                self.phase = MissionPhase.BOOSTER_RETURN

    # ----- Pre-separation ------------------------------------------------

    def _pre_separation(self, b: Vehicle, u: Vehicle, dt: float) -> None:
        ph = self.phase

        if ph == MissionPhase.COUNTDOWN:
            b.set_throttle(0.0)
            b.set_gimbal(0.0)
            # Hold on pad
            b.state.x = 0.0
            b.state.y = self._pad_height
            b.state.vx = b.state.vy = b.state.omega = 0.0
            if self._clock >= -self.IGNITION_LEAD:
                self.phase = MissionPhase.IGNITION

        elif ph == MissionPhase.IGNITION:
            t = self._clock + self.IGNITION_LEAD
            b.set_throttle(clamp(t * self.THROTTLE_RAMP_RATE, 0.0, 1.0))
            b.set_gimbal(0.0)
            b.state.x = 0.0
            b.state.y = self._pad_height
            b.state.vx = b.state.vy = b.state.omega = 0.0
            if self._clock >= 0.0:
                self.phase = MissionPhase.VERTICAL_ASCENT

        elif ph == MissionPhase.VERTICAL_ASCENT:
            b.set_throttle(1.0)
            self._steer(b, math.pi / 2, self.ascent_pid, dt)
            if b.speed >= self.GRAVITY_TURN_START_SPEED:
                self.phase = MissionPhase.GRAVITY_TURN
                self._target_pitch = math.pi / 2

        elif ph == MissionPhase.GRAVITY_TURN:
            b.set_throttle(1.0)
            self._target_pitch -= self.PITCH_PROGRAM_RATE * dt
            self._target_pitch = max(self._target_pitch, 0.1)
            self._steer(b, self._target_pitch, self.ascent_pid, dt)

            if b.state.fuel <= b.reserve_fuel:
                b.shutdown_engine()
                self.phase = MissionPhase.BECO
                self._beco_time = self._clock

        elif ph == MissionPhase.BECO:
            b.set_gimbal(0.0)
            if self._clock - self._beco_time >= self.SEPARATION_DELAY:
                self._do_separation(b, u)

    def _do_separation(self, b: Vehicle, u: Vehicle) -> None:
        b.payload_mass = 0.0
        u.copy_state_from(b)
        # Small separation impulse
        dx = math.cos(b.state.theta) * 2.0
        dy = math.sin(b.state.theta) * 2.0
        u.state.vx += dx
        u.state.vy += dy

        self._sep_time = self._clock
        self._flip_target = b.state.theta + math.pi
        self._bphase = BoosterPhase.FLIP
        self._uphase = UpperPhase.WAIT_IGNITION
        self._upper_target_pitch = u.state.theta
        self.phase = MissionPhase.BOOSTER_RETURN

    # ----- Booster return ------------------------------------------------

    def _control_booster(self, b: Vehicle, dt: float) -> None:
        if b.is_terminated:
            self._bphase = BoosterPhase.DONE
            return

        match self._bphase:
            case BoosterPhase.FLIP:
                self._booster_flip(b, dt)
            case BoosterPhase.BOOSTBACK:
                self._booster_boostback(b, dt)
            case BoosterPhase.COAST:
                self._booster_coast(b, dt)
            case BoosterPhase.ENTRY_BURN:
                self._booster_entry_burn(b, dt)
            case BoosterPhase.ENTRY_COAST:
                self._booster_entry_coast(b, dt)
            case BoosterPhase.LANDING_BURN:
                self._booster_landing_burn(b, dt)

    def _booster_flip(self, b: Vehicle, dt: float) -> None:
        b.set_throttle(0.0)
        error = wrap_angle(self._flip_target - b.state.theta)
        if abs(error) < 0.1 and abs(b.state.omega) < 0.5:
            # Flip complete — kill rotation and move on
            b.state.omega = 0.0
            b.state.theta = self._flip_target
            self._bphase = BoosterPhase.BOOSTBACK
        else:
            # P-D control for flip: accelerate toward target, decelerate near it
            desired_omega = clamp(error * 2.0, -self.FLIP_RATE, self.FLIP_RATE)
            omega_error = desired_omega - b.state.omega
            b.state.omega += clamp(omega_error, -self.FLIP_RATE * dt * 3, self.FLIP_RATE * dt * 3)

    def _booster_boostback(self, b: Vehicle, dt: float) -> None:
        b.ignite(throttle=1.0, engines=self.BOOSTBACK_ENGINES)

        # Phase 1: Kill upward velocity (retrograde burn)
        # Phase 2: Once mostly stopped, push toward pad
        if b.speed > 150:
            # Point retrograde — kill all velocity
            target = math.atan2(-b.state.vy, -b.state.vx)
        else:
            # Speed is low — point toward pad
            dx = self._landing_x - b.state.x
            target = math.atan2(0.0, dx) if abs(dx) > 100 else math.pi / 2
        self._steer(b, target, self.landing_pid, dt)

        # End when predicted coast landing overshoots the pad.
        # Overshoot compensates for entry burn reducing horizontal coast distance.
        predicted_x = self._predict_landing_x(b)
        sign = 1.0 if b.state.x > self._landing_x else -1.0
        dist = abs(b.state.x - self._landing_x)
        overshoot = max(dist * 0.55, 3000)
        overshoot_target = self._landing_x - sign * overshoot
        if abs(predicted_x - overshoot_target) < 3000 and b.speed < 200:
            b.shutdown_engine()
            self._bphase = BoosterPhase.COAST

    def _booster_coast(self, b: Vehicle, dt: float) -> None:
        b.set_throttle(0.0)
        # Orient retrograde for re-entry using RCS (no thrust = no gimbal torque)
        if b.speed > 20:
            retro = math.atan2(-b.state.vy, -b.state.vx)
            self._rcs_steer(b, retro, dt)
        else:
            self._rcs_steer(b, math.pi / 2, dt)

        # Transition to entry burn when descending through entry altitude
        if b.altitude <= self.ENTRY_BURN_ALT and b.state.vy < -50:
            self._bphase = BoosterPhase.ENTRY_BURN

    LANDING_ENGINES = 3  # 3-engine landing burn (like SpaceX heavy missions)

    def _booster_entry_burn(self, b: Vehicle, dt: float) -> None:
        b.ignite(throttle=1.0, engines=self.ENTRY_ENGINES)
        # Point mostly retrograde with strong horizontal correction toward pad
        if b.speed > 30:
            retro = math.atan2(-b.state.vy, -b.state.vx)
            pred_x = self._predict_landing_x(b)
            error_x = self._landing_x - pred_x
            h_correction = clamp(error_x * 0.0005, -0.3, 0.3)
            self._steer(b, retro + h_correction, self.landing_pid, dt)
        else:
            self._steer(b, math.pi / 2, self.landing_pid, dt)

        # End when speed is manageable or fuel gone
        speed_ok = b.speed < 150
        out_of_fuel = not b.has_fuel
        low_alt = b.altitude < 8_000
        if speed_ok or out_of_fuel:
            b.shutdown_engine()
            self._bphase = BoosterPhase.ENTRY_COAST
        elif low_alt:
            self._bphase = BoosterPhase.LANDING_BURN

    def _booster_entry_coast(self, b: Vehicle, dt: float) -> None:
        b.set_throttle(0.0)
        self._rcs_steer(b, math.pi / 2, dt)

        if b.altitude > 0 and b.state.vy < -5:
            h_needed = self._suicide_burn_alt(b)
            if b.altitude <= h_needed * self.LANDING_BURN_SAFETY:
                self._bphase = BoosterPhase.LANDING_BURN

    def _booster_landing_burn(self, b: Vehicle, dt: float) -> None:
        # Proportional-derivative guidance toward pad
        dx_to_pad = self._landing_x - b.state.x
        # Desired horizontal velocity: proportional to distance, limited
        desired_vx = clamp(dx_to_pad * 0.1, -150, 150)
        vx_error = desired_vx - b.state.vx
        # Tilt angle: up to 25° from vertical at high altitude, 10° near ground
        max_tilt = 0.4 if b.altitude > 1000 else 0.2
        h_correction = clamp(vx_error * 0.015, -max_tilt, max_tilt)
        target_angle = math.pi / 2 + h_correction
        if b.engine_state.name == 'BURNING':
            self._steer(b, target_angle, self.landing_pid, dt)
        else:
            self._rcs_steer(b, target_angle, dt)

        if b.altitude <= self._pad_height + 0.5 and abs(b.state.vy) < 30.0:
            b.shutdown_engine()
            b.state.vy = 0.0
            b.state.vx = 0.0
            b.state.omega = 0.0
            b.state.theta = math.pi / 2
            b.state.y = self._pad_height
            b.state.x = self._landing_x
            b.flight_phase = FlightPhase.LANDED
            return

        # Terminal guidance: lateral RCS to steer precisely to pad
        if b.altitude < 500:
            dx_to_pad = self._landing_x - b.state.x
            desired_vx = clamp(dx_to_pad * 0.5, -30, 30)
            accel = 20.0 if b.altitude < 50 else (15.0 if b.altitude < 200 else 8.0)
            vx_correction = clamp((desired_vx - b.state.vx) * 0.8, -accel * dt, accel * dt)
            b.state.vx += vx_correction

        v = abs(b.state.vy)
        h = max(b.altitude, 0.5)

        # Compute suicide burn altitude — if we're above it, coast (engines off)
        h_needed = self._suicide_burn_alt(b)
        above_suicide_point = h > h_needed * self.LANDING_BURN_SAFETY and v > 5

        if above_suicide_point:
            # Too high — coast, engines off
            b.shutdown_engine()
        elif b.state.vy < -1.0:
            # At or below suicide burn point — brake
            desired_decel = v * v / (2 * h) + b.g_local

            # 3 engines for fast braking, 1 engine for final approach
            if v > 30 or h > 500:
                engines = self.LANDING_ENGINES
            else:
                engines = 1

            engine_thrust = b.cfg.max_thrust * engines / max(b.cfg.num_engines, 1)
            max_decel = engine_thrust / b.total_mass
            throttle = clamp(desired_decel / max(max_decel, 0.1),
                             b.cfg.min_throttle, 1.0)
            b.ignite(throttle=throttle, engines=engines)
        elif b.state.vy >= 0 and b.altitude > 5.0:
            # Going up — cut engines
            b.shutdown_engine()
        else:
            # Near ground, low speed — gentle single-engine touchdown
            b.ignite(throttle=b.cfg.min_throttle, engines=1)

    def _suicide_burn_alt(self, b: Vehicle) -> float:
        """Altitude at which to start landing burn."""
        v = abs(b.state.vy)
        landing_thrust = b.cfg.max_thrust * self.LANDING_ENGINES / max(b.cfg.num_engines, 1)
        decel = landing_thrust / b.total_mass - b.g_local
        if decel <= 0:
            return 999999.0
        return v * v / (2 * decel)

    def _predict_landing_x(self, b: Vehicle) -> float:
        """Predict horizontal position when booster reaches ground, assuming ballistic coast."""
        g = b.g_local
        vy = b.state.vy
        alt = max(b.altitude, 1.0)
        # Time to reach ground: solve alt + vy*t - 0.5*g*t² = 0
        # t = (vy + sqrt(vy² + 2*g*alt)) / g
        discriminant = vy * vy + 2 * g * alt
        if discriminant < 0:
            return b.state.x
        t_ground = (vy + math.sqrt(discriminant)) / g
        return b.state.x + b.state.vx * t_ground

    def _rcs_steer(self, b: Vehicle, target: float, dt: float) -> None:
        """Attitude control using simulated RCS (direct torque) — for coast phases."""
        error = wrap_angle(target - b.state.theta)
        desired_omega = clamp(error * 1.5, -1.0, 1.0)
        omega_error = desired_omega - b.state.omega
        b.state.omega += clamp(omega_error, -self.RCS_TORQUE * dt, self.RCS_TORQUE * dt)

    # ----- Upper stage ---------------------------------------------------

    def _control_upper(self, u: Vehicle, dt: float) -> None:
        if u.is_terminated:
            self._uphase = UpperPhase.DONE
            return

        t_since_sep = self._clock - self._sep_time

        if self._uphase == UpperPhase.WAIT_IGNITION:
            if t_since_sep >= self.UPPER_IGNITION_DELAY:
                self._uphase = UpperPhase.BURNING

        elif self._uphase == UpperPhase.BURNING:
            has_main_fuel = u.state.fuel > u.reserve_fuel
            if has_main_fuel and u.engine_state != EngineState.CUTOFF:
                u.ignite(throttle=1.0, engines=1)
                self._upper_target_pitch -= self.UPPER_PITCH_RATE * dt
                self._upper_target_pitch = max(self._upper_target_pitch, 0.05)
                self._steer(u, self._upper_target_pitch, self.upper_pid, dt)
            else:
                u.shutdown_engine()
                self._uphase = UpperPhase.COAST

        elif self._uphase == UpperPhase.COAST:
            u.set_throttle(0.0)
            u.set_gimbal(0.0)
            # Once past apogee, start return sequence
            if u.state.vy < -10:
                self._upper_flip_target = u.state.theta + math.pi
                self._uphase = UpperPhase.RETURN_FLIP

        elif self._uphase == UpperPhase.RETURN_FLIP:
            self._upper_return_flip(u, dt)

        elif self._uphase == UpperPhase.RETURN_BURN:
            self._upper_return_burn(u, dt)

        elif self._uphase == UpperPhase.RETURN_COAST:
            self._upper_return_coast(u, dt)

        elif self._uphase == UpperPhase.LANDING:
            self._upper_landing(u, dt)

    def _upper_return_flip(self, u: Vehicle, dt: float) -> None:
        """Flip upper stage 180° for return burn."""
        error = wrap_angle(self._upper_flip_target - u.state.theta)
        if abs(error) < 0.1 and abs(u.state.omega) < 0.5:
            u.state.omega = 0.0
            u.state.theta = self._upper_flip_target
            self._uphase = UpperPhase.RETURN_BURN
        else:
            desired_omega = clamp(error * 2.0, -self.FLIP_RATE, self.FLIP_RATE)
            omega_error = desired_omega - u.state.omega
            u.state.omega += clamp(omega_error, -self.FLIP_RATE * dt * 3, self.FLIP_RATE * dt * 3)

    def _upper_return_burn(self, u: Vehicle, dt: float) -> None:
        """Retrograde burn redirecting trajectory toward launch pad."""
        min_landing_fuel = 575.0
        if u.state.fuel <= min_landing_fuel:
            u.shutdown_engine()
            u.state.omega = 0.0  # kill spin before coast
            self._uphase = UpperPhase.RETURN_COAST
            return

        u.ignite(throttle=1.0, engines=1)

        # Compute desired velocity to reach launch pad.
        # Use drag-adjusted estimate: at high alt, drag will kill most speed
        # below ~30km, so cap desired_vx to avoid overshooting.
        g = u.g_local
        alt = max(u.altitude, 1.0)
        disc = u.state.vy ** 2 + 2 * g * alt
        t_ground = (u.state.vy + math.sqrt(max(disc, 0))) / g if disc > 0 else 100.0
        t_ground = max(t_ground, 10.0)

        desired_vx = (self._launch_x - u.state.x) / t_ground
        dv_x = desired_vx - u.state.vx
        # Gently reduce vertical speed
        dv_y = -u.state.vy * 0.3

        target = math.atan2(dv_y, dv_x)
        self._steer(u, target, self.upper_pid, dt)

        # End burn: estimate x position when entering thick atmosphere (~40km)
        # Below 40km, drag kills most horizontal speed.
        atmo_alt = 40_000.0
        if alt > atmo_alt:
            dh = alt - atmo_alt
            vy_down = abs(u.state.vy)  # downward speed
            # t from: dh = vy_down*t + 0.5*g*t²  (falling)
            # → t = (-vy_down + sqrt(vy_down² + 2*g*dh)) / g
            t_to_atmo = (-vy_down + math.sqrt(vy_down ** 2 + 2 * g * dh)) / g
            t_to_atmo = max(t_to_atmo, 1.0)
            x_at_atmo = u.state.x + u.state.vx * t_to_atmo
        else:
            x_at_atmo = u.state.x
        near_pad = abs(x_at_atmo - self._launch_x) < 15000
        if near_pad:
            u.shutdown_engine()
            u.state.omega = 0.0  # kill spin before coast
            self._uphase = UpperPhase.RETURN_COAST

    def _upper_return_coast(self, u: Vehicle, dt: float) -> None:
        """Coast back, orient for landing."""
        u.set_throttle(0.0)
        u.state.theta = wrap_angle(u.state.theta)
        if u.altitude > 3000 and u.speed > 50:
            retro = math.atan2(-u.state.vy, -u.state.vx)
            self._rcs_steer(u, retro, dt)
        else:
            self._rcs_steer(u, math.pi / 2, dt)

        # Start landing sequence when low enough (higher threshold if fast)
        h_speed = abs(u.state.vx)
        landing_alt = 5000 if h_speed < 100 else 15000
        if u.altitude < landing_alt and u.state.vy < -10:
            # Stabilize orientation for landing
            u.state.omega = 0.0
            self._uphase = UpperPhase.LANDING

    def _upper_landing(self, u: Vehicle, dt: float) -> None:
        """Upper stage landing: retrograde braking then vertical touchdown."""
        if u.altitude <= self._pad_height + 0.5 and u.speed < 30.0:
            u.shutdown_engine()
            u.state.vy = 0.0
            u.state.vx = 0.0
            u.state.omega = 0.0
            u.state.theta = math.pi / 2
            u.state.y = self._pad_height
            u.state.x = self._launch_x
            u.flight_phase = FlightPhase.LANDED
            return

        h = max(u.altitude, 0.5)
        total_v = u.speed
        max_decel = u.cfg.max_thrust / u.total_mass

        # Active attitude control: directly set theta to target and zero omega.
        # Simulates powerful grid fin + RCS attitude control system.
        # Terminal guidance handles lateral position accuracy independently.

        # Terminal guidance: lateral RCS to steer precisely to pad
        if h < 5000:
            dx_to_pad = self._launch_x - u.state.x
            desired_vx = clamp(dx_to_pad * 0.25, -100, 100)
            accel = 20.0 if h < 500 else (12.0 if h < 2000 else 8.0)
            vx_correction = clamp((desired_vx - u.state.vx) * 0.5, -accel * dt, accel * dt)
            u.state.vx += vx_correction

        if abs(u.state.vx) > 50 or u.altitude > 2000:
            # Phase 1: retrograde burn to kill total velocity, with pad correction
            if total_v > 20:
                retro = math.atan2(-u.state.vy, -u.state.vx)
                pred_x = self._predict_landing_x_for(u)
                dx = self._launch_x - pred_x
                h_corr = clamp(dx * 0.00005, -0.15, 0.15)
                target = retro - h_corr
            else:
                target = math.pi / 2

            # Direct attitude control — no spin
            u.state.theta = target
            u.state.omega = 0.0

            # Burn immediately if speed is high (need to brake before ground)
            h_needed = total_v * total_v / (2 * max(max_decel - u.g_local, 1.0))
            if total_v > 100 and u.has_fuel:
                u.ignite(throttle=1.0, engines=1)
            elif h <= h_needed * 1.4 and total_v > 30 and u.has_fuel:
                u.ignite(throttle=1.0, engines=1)
            elif total_v < 30:
                u.shutdown_engine()
            else:
                u.shutdown_engine()
        else:
            # Phase 2: vertical landing with pad correction
            dx_to_pad = self._launch_x - u.state.x
            desired_vx = clamp(dx_to_pad * 0.05, -50, 50)
            vx_error = desired_vx - u.state.vx
            max_tilt = 0.2
            h_corr = clamp(vx_error * 0.015, -max_tilt, max_tilt)
            target = math.pi / 2 - h_corr

            # Direct attitude control — no spin
            u.state.theta = target
            u.state.omega = 0.0

            v_vert = abs(u.state.vy)
            # Suicide burn altitude check
            h_needed = v_vert * v_vert / (2 * max(max_decel - u.g_local, 1.0))
            dx_remaining = abs(self._launch_x - u.state.x)
            safety = 3.0 if dx_remaining > 100 else 1.4
            above_suicide = h > h_needed * safety and v_vert > 5

            if above_suicide:
                u.shutdown_engine()
            elif u.state.vy < -1.0 and u.has_fuel:
                desired_decel = v_vert * v_vert / (2 * h) + u.g_local
                throttle = clamp(desired_decel / max(max_decel, 0.1),
                                 u.cfg.min_throttle, 1.0)
                u.ignite(throttle=throttle, engines=1)
            elif u.state.vy >= 0 and u.altitude > 5.0:
                u.shutdown_engine()

    def _predict_landing_x_for(self, v: Vehicle) -> float:
        """Predict landing x for any vehicle."""
        g = v.g_local
        vy = v.state.vy
        alt = max(v.altitude, 1.0)
        discriminant = vy * vy + 2 * g * alt
        if discriminant < 0:
            return v.state.x
        t_ground = (vy + math.sqrt(discriminant)) / g
        return v.state.x + v.state.vx * t_ground

    # ----- Helpers -------------------------------------------------------

    def _steer(self, v: Vehicle, target: float, pid: PIDController, dt: float) -> None:
        error = wrap_angle(target - v.state.theta)
        damped = error - 0.5 * v.state.omega
        cmd = pid.update(damped, dt)
        desired = -cmd * v.cfg.max_gimbal_angle
        v.adjust_gimbal(desired - v.gimbal_angle, dt)

    def reset(self) -> None:
        self.phase = MissionPhase.COUNTDOWN
        self._bphase = BoosterPhase.FLIP
        self._uphase = UpperPhase.WAIT_IGNITION
        self.ascent_pid.reset()
        self.landing_pid.reset()
        self.upper_pid.reset()
        self._clock = -self.COUNTDOWN_DURATION
        self._target_pitch = math.pi / 2
        self._upper_target_pitch = math.pi / 2
        self._beco_time = self._sep_time = 0.0
        self._upper_flip_target = 0.0
