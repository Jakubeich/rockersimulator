"""
Telemetry recorder — separate tracking for booster and upper stage.

Output files in output/:
    telemetry_booster.csv / telemetry_upper.csv
    flight_log.txt       — unified event log
    flight_summary.txt   — key statistics for both vehicles
    telemetry_plots.png  — graphs
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rocket import Vehicle

OUTPUT_DIR = Path("output")


@dataclass(slots=True)
class TelemetryFrame:
    time: float = 0.0
    vehicle_id: str = ""
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0
    theta: float = 0.0
    omega: float = 0.0
    mass: float = 0.0
    fuel: float = 0.0
    thrust: float = 0.0
    throttle: float = 0.0
    dynamic_pressure: float = 0.0
    altitude: float = 0.0
    downrange: float = 0.0
    gimbal: float = 0.0

    @classmethod
    def field_names(cls) -> list[str]:
        import dataclasses
        return [f.name for f in dataclasses.fields(cls)]


@dataclass(slots=True)
class FlightEvent:
    time: float
    mission_time: float
    vehicle: str
    phase: str
    altitude: float
    speed: float
    fuel: float
    message: str


class VehicleTelemetry:
    """Records telemetry for a single vehicle."""

    def __init__(self, vehicle_id: str, sample_interval: float = 0.1) -> None:
        self.vehicle_id = vehicle_id
        self.sample_interval = sample_interval
        self.frames: list[TelemetryFrame] = []
        self._last_time: float = -math.inf

    def record(self, v: Vehicle) -> None:
        if v.time - self._last_time < self.sample_interval:
            return
        self._last_time = v.time

        self.frames.append(TelemetryFrame(
            time=v.time,
            vehicle_id=self.vehicle_id,
            x=v.state.x, y=v.state.y,
            vx=v.state.vx, vy=v.state.vy,
            speed=v.speed,
            theta=v.state.theta, omega=v.state.omega,
            mass=v.total_mass, fuel=v.state.fuel,
            thrust=v.current_thrust, throttle=v.throttle,
            dynamic_pressure=v.q,
            altitude=v.altitude, downrange=v.state.x,
            gimbal=v.gimbal_angle,
        ))

    @property
    def trajectory(self) -> list[tuple[float, float]]:
        return [(f.x, f.y) for f in self.frames]

    def clear(self) -> None:
        self.frames.clear()
        self._last_time = -math.inf


class MissionTelemetry:
    """Aggregates telemetry for both vehicles + events."""

    def __init__(self, sample_interval: float = 0.1) -> None:
        self.booster = VehicleTelemetry("booster", sample_interval)
        self.upper = VehicleTelemetry("upper", sample_interval)
        self.events: list[FlightEvent] = []
        self._last_phase: str = ""

    def record(self, booster: Vehicle, upper: Vehicle,
               mission_time: float = 0.0, phase_name: str = "") -> None:
        if booster.is_active:
            self.booster.record(booster)
        if upper.is_active:
            self.upper.record(upper)

        if phase_name and phase_name != self._last_phase:
            v = booster if booster.is_active else upper
            self.events.append(FlightEvent(
                time=v.time, mission_time=mission_time,
                vehicle="mission", phase=phase_name,
                altitude=v.altitude, speed=v.speed,
                fuel=v.state.fuel, message=f"Phase: {phase_name}",
            ))
            self._last_phase = phase_name

    def log_event(self, vehicle: Vehicle, mission_time: float,
                  vehicle_id: str, message: str) -> None:
        self.events.append(FlightEvent(
            time=vehicle.time, mission_time=mission_time,
            vehicle=vehicle_id, phase=self._last_phase,
            altitude=vehicle.altitude, speed=vehicle.speed,
            fuel=vehicle.state.fuel, message=message,
        ))

    def clear(self) -> None:
        self.booster.clear()
        self.upper.clear()
        self.events.clear()
        self._last_phase = ""

    # ----- Export --------------------------------------------------------

    def export_all(self) -> Path:
        OUTPUT_DIR.mkdir(exist_ok=True)
        self._export_csv(self.booster.frames, OUTPUT_DIR / "telemetry_booster.csv")
        self._export_csv(self.upper.frames, OUTPUT_DIR / "telemetry_upper.csv")
        self._export_log(OUTPUT_DIR / "flight_log.txt")
        self._export_summary(OUTPUT_DIR / "flight_summary.txt")
        return OUTPUT_DIR

    def _export_csv(self, frames: list[TelemetryFrame], path: Path) -> None:
        if not frames:
            return
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(TelemetryFrame.field_names())
            for fr in frames:
                w.writerow([getattr(fr, n) for n in TelemetryFrame.field_names()])

    def _export_log(self, path: Path) -> None:
        with path.open("w") as f:
            f.write("=" * 90 + "\n  FLIGHT LOG\n" + "=" * 90 + "\n\n")
            f.write(f"{'Mission T':>12s}  {'Vehicle':<10s}  {'Phase':<22s}  "
                    f"{'Alt (m)':>10s}  {'Spd (m/s)':>10s}  Message\n")
            f.write("-" * 90 + "\n")
            for ev in self.events:
                mt = f"T{ev.mission_time:+.2f}s"
                f.write(f"{mt:>12s}  {ev.vehicle:<10s}  {ev.phase:<22s}  "
                        f"{ev.altitude:10.1f}  {ev.speed:10.1f}  {ev.message}\n")

    def _export_summary(self, path: Path) -> None:
        with path.open("w") as f:
            f.write("=" * 55 + "\n  MISSION SUMMARY\n" + "=" * 55 + "\n")
            for label, frames in [("BOOSTER", self.booster.frames),
                                  ("UPPER STAGE", self.upper.frames)]:
                if not frames:
                    continue
                max_alt = max(fr.altitude for fr in frames)
                max_spd = max(fr.speed for fr in frames)
                max_q = max(fr.dynamic_pressure for fr in frames)
                dur = frames[-1].time - frames[0].time
                f.write(f"\n  {label}\n  {'-'*len(label)}\n")
                f.write(f"  Duration:      {dur:10.1f} s\n")
                f.write(f"  Max altitude:  {max_alt:10.0f} m ({max_alt/1000:.1f} km)\n")
                f.write(f"  Max speed:     {max_spd:10.0f} m/s ({max_spd*3.6:.0f} km/h)\n")
                f.write(f"  Max Q:         {max_q:10.0f} Pa\n")

            f.write(f"\n  EVENTS\n  ------\n")
            for ev in self.events:
                f.write(f"  T{ev.mission_time:+8.1f}s  {ev.message}\n")

    # ----- Plots ---------------------------------------------------------

    def plot(self, save_to_file: bool = True) -> None:
        try:
            import matplotlib
            if save_to_file:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=False)
        fig.suptitle("Mission Telemetry", fontsize=15, fontweight="bold")

        def _plot(ax, ylabel, b_data, u_data, b_t, u_t):
            if b_data:
                ax.plot(b_t, b_data, linewidth=1.0, color="tab:blue", label="Booster")
            if u_data:
                ax.plot(u_t, u_data, linewidth=1.0, color="tab:orange", label="Upper")
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

        bt = [f.time for f in self.booster.frames]
        ut = [f.time for f in self.upper.frames]

        _plot(axes[0][0], "Altitude (km)",
              [f.altitude / 1000 for f in self.booster.frames],
              [f.altitude / 1000 for f in self.upper.frames], bt, ut)
        _plot(axes[0][1], "Speed (m/s)",
              [f.speed for f in self.booster.frames],
              [f.speed for f in self.upper.frames], bt, ut)
        _plot(axes[1][0], "Vy (m/s)",
              [f.vy for f in self.booster.frames],
              [f.vy for f in self.upper.frames], bt, ut)
        _plot(axes[1][1], "Downrange (km)",
              [f.downrange / 1000 for f in self.booster.frames],
              [f.downrange / 1000 for f in self.upper.frames], bt, ut)
        _plot(axes[2][0], "Mass (kg)",
              [f.mass for f in self.booster.frames],
              [f.mass for f in self.upper.frames], bt, ut)
        _plot(axes[2][1], "Fuel (kg)",
              [f.fuel for f in self.booster.frames],
              [f.fuel for f in self.upper.frames], bt, ut)
        _plot(axes[3][0], "Thrust (N)",
              [f.thrust for f in self.booster.frames],
              [f.thrust for f in self.upper.frames], bt, ut)
        _plot(axes[3][1], "Pitch (deg)",
              [math.degrees(f.theta) for f in self.booster.frames],
              [math.degrees(f.theta) for f in self.upper.frames], bt, ut)

        for ax in axes[-1]:
            ax.set_xlabel("Time (s)")
        plt.tight_layout()

        if save_to_file:
            OUTPUT_DIR.mkdir(exist_ok=True)
            p = OUTPUT_DIR / "telemetry_plots.png"
            plt.savefig(p, dpi=150)
            print(f"  Plots saved to {p}")
            plt.close(fig)
        else:
            plt.show()
