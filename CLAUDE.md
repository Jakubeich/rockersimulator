# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Simulator

```bash
# Install dependencies
pip install -r requirements.txt   # pygame-ce, numpy, matplotlib

# Run
python main.py
```

No build step. No tests exist yet. No linter configured.

## Architecture

A 2D two-stage rocket flight simulator (Python 3.12+, Pygame) with Newtonian physics, automatic flight control, and real-time visualization. All quantities use SI units (m, s, kg, N, rad).

### Physics Pipeline

`physics.py` (pure functions, no state) → `integrators.py` (Euler/RK4) → `rocket.py` (Vehicle class)

- **RocketState** is a 7-element vector: x, y, vx, vy, theta, omega, fuel
- `state_derivative()` computes dy/dt from forces (gravity, thrust with gimbal, drag axial+lateral, torques)
- The integrator advances state each dt=0.01s; RK4 is the default
- Coordinate system: +x right, +y up, θ CCW from +x axis (θ=π/2 = pointing up)

### Flight Control

`controller.py` contains `MissionSequencer` which automates the full mission:

1. **Pre-separation**: Countdown → Ignition → Vertical ascent → Gravity turn → BECO → Separation
2. **Booster return**: Flip → Boostback → Coast → Entry burn → Landing burn → Touchdown at landing pad (x=-2000m)
3. **Upper stage**: Burn to space → Coast → Return flip → Retrograde burn → Coast → Landing at launch pad (x=0m)

Key control concepts:
- **PIDController** for attitude steering (gimbal angle)
- **_rcs_steer()** for coast-phase attitude control (direct omega manipulation simulating RCS thrusters)
- **Suicide burn**: `h_start = v²/(2*(T/m - g))` — altitude to start deceleration
- **Terminal guidance**: direct vx correction below 500m (booster) / 5000m (upper) simulating lateral RCS
- **Direct attitude control** for upper stage landing (theta set directly to target, simulating grid fins)
- Sign convention trap: `θ > π/2` tilts thrust LEFT (−x), `θ < π/2` tilts RIGHT (+x). The booster landing uses `π/2 + h_correction` (historically calibrated, "wrong" sign that works due to trajectory equilibrium). Upper stage uses corrected signs.

### Simulation Loop

`simulation.py` ties everything together:
- Inner loop runs physics at dt=0.01s with time warp support (up to 2000 steps/frame)
- `sequencer.update()` runs before each `vehicle.step()`
- Telemetry sampled at 0.1s intervals; exported to CSV on exit

### Rendering

`renderer.py` draws in layers: sky gradient → stars → atmosphere glow → ground → pads → particles → vehicles → re-entry glow → HUD

- **Camera**: smooth follow with lookahead, auto-zoom based on altitude/distance
- **ParticleSystem**: exhaust smoke from engines
- **StarField**: fades in above 10km altitude
- Vehicle models: tapered body, ogive nose, grid fins, landing legs, multi-layer flame with Mach diamonds

### Configuration

`config.py` uses nested dataclasses. `AppConfig` is the top-level container holding `VehicleConfig` (booster + upper stage `StageConfig`), `PadConfig`, `EnvironmentConfig`, `AutopilotConfig`, `SimulationConfig`, `RenderConfig`. Default values produce a working SpaceX-style mission.

### Telemetry

`telemetry.py` records per-vehicle flight data, exports `telemetry_booster.csv`, `telemetry_upper.csv`, `flight_log.txt`, `flight_summary.txt`, and generates `telemetry_plots.png` (matplotlib) to `output/`.

## Key Relationships

- `Vehicle` temporarily mutates its `StageConfig` during `step()` to account for engine fraction and payload mass, then restores originals
- Booster carries upper stage as `payload_mass` until separation, when `copy_state_from()` transfers kinematic state
- `wrap_angle()` and theta normalization in `Vehicle.step()` prevent visual spin accumulation (sin/cos are periodic, so this is physics-neutral)
- The controller uses `BoosterPhase` and `UpperPhase` internal enums separately from the display `MissionPhase`

## Keyboard Controls

P=Pause, R=Reset, TAB=Switch focus, +/-=Zoom, T=Trail, V=Vectors, F3=Debug, Z=AutoZoom, 1-4=Time warp, ESC=Quit
