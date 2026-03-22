"""
Simulation loop — two-stage rocket with automatic flight sequence.
"""

from __future__ import annotations

import pygame

from config import AppConfig
from controller import MissionSequencer
from renderer import Renderer
from rocket import Vehicle
from telemetry import MissionTelemetry


class Simulation:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg

        # Create vehicles
        self.booster = Vehicle(
            cfg.vehicle.booster, cfg.environment, cfg.simulation, "booster")
        self.upper = Vehicle(
            cfg.vehicle.upper_stage, cfg.environment, cfg.simulation, "upper")

        # Upper stage starts inactive (riding on booster)
        self.upper.is_active = False
        # Booster carries upper stage as payload
        self.booster.payload_mass = (
            cfg.vehicle.upper_stage.dry_mass + cfg.vehicle.upper_stage.fuel_mass
        )
        # Place booster on launch pad surface
        self.booster.state.y = cfg.pads.pad_height

        self.sequencer = MissionSequencer(cfg.autopilot, cfg.pads)
        self.telemetry = MissionTelemetry(sample_interval=0.1)
        self.renderer = Renderer(
            cfg.render, cfg.pads,
            cfg.vehicle.booster, cfg.vehicle.upper_stage)

        self.paused = cfg.simulation.paused_on_start
        self.running = True
        self.clock = pygame.time.Clock()
        self._phys_acc: float = 0.0

    def run(self) -> None:
        while self.running:
            frame_dt = min(self.clock.tick(self.cfg.render.fps) / 1000.0, 0.1)
            self._handle_events()
            if not self.paused and not self._mission_done():
                self._step_physics(frame_dt)
            self._render()
        self._on_exit()

    def _mission_done(self) -> bool:
        from controller import MissionPhase
        if self.sequencer.phase == MissionPhase.MISSION_COMPLETE:
            return True
        b_done = self.booster.is_terminated or not self.booster.is_active
        u_done = not self.upper.is_active or self.upper.is_terminated
        return b_done and u_done

    def _step_physics(self, frame_dt: float) -> None:
        dt = self.cfg.simulation.dt
        self._phys_acc += frame_dt * self.cfg.simulation.time_warp
        steps = 0
        while self._phys_acc >= dt and steps < 2000:
            self.sequencer.update(self.booster, self.upper, dt)
            self.booster.step(dt)
            if self.upper.is_active:
                self.upper.step(dt)
            self.telemetry.record(
                self.booster, self.upper,
                self.sequencer.mission_time, self.sequencer.phase_name)
            self._phys_acc -= dt
            steps += 1

            if self.booster.time > self.cfg.simulation.max_duration:
                break

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._on_key(event.key)
            elif event.type == pygame.MOUSEWHEEL:
                (self.renderer.camera.zoom_in if event.y > 0
                 else self.renderer.camera.zoom_out)()
            elif event.type == pygame.VIDEORESIZE:
                self.renderer.handle_resize(event.w, event.h)

    def _on_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_p:
            self.paused = not self.paused
        elif key == pygame.K_r:
            self._reset()
        elif key == pygame.K_TAB:
            self.renderer.toggle_focus()
        elif key == pygame.K_t:
            self.cfg.render.show_trajectory = not self.cfg.render.show_trajectory
        elif key == pygame.K_v:
            self.cfg.render.show_vectors = not self.cfg.render.show_vectors
        elif key == pygame.K_F3:
            self.cfg.render.show_debug = not self.cfg.render.show_debug
        elif key == pygame.K_z:
            self.renderer.toggle_auto_zoom()
        elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.renderer.camera.zoom_in()
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.renderer.camera.zoom_out()
        elif key == pygame.K_1:
            self.cfg.simulation.time_warp = 1.0
        elif key == pygame.K_2:
            self.cfg.simulation.time_warp = 2.0
        elif key == pygame.K_3:
            self.cfg.simulation.time_warp = 5.0
        elif key == pygame.K_4:
            self.cfg.simulation.time_warp = 10.0

    def _render(self) -> None:
        self.renderer.draw(
            self.booster, self.upper,
            self.telemetry.booster.trajectory,
            self.telemetry.upper.trajectory,
            self.sequencer, self.paused)

    def _reset(self) -> None:
        cfg = self.cfg
        self.booster = Vehicle(
            cfg.vehicle.booster, cfg.environment, cfg.simulation, "booster")
        self.upper = Vehicle(
            cfg.vehicle.upper_stage, cfg.environment, cfg.simulation, "upper")
        self.upper.is_active = False
        self.booster.payload_mass = (
            cfg.vehicle.upper_stage.dry_mass + cfg.vehicle.upper_stage.fuel_mass
        )
        self.booster.state.y = cfg.pads.pad_height
        self.sequencer.reset()
        self.telemetry.clear()
        self.paused = cfg.simulation.paused_on_start
        self._phys_acc = 0.0

    def _on_exit(self) -> None:
        pygame.quit()
        if self.telemetry.booster.frames or self.telemetry.upper.frames:
            out = self.telemetry.export_all()
            nb = len(self.telemetry.booster.frames)
            nu = len(self.telemetry.upper.frames)
            print(f"\nFlight data exported to {out}/")
            print(f"  Booster:  {nb} frames")
            print(f"  Upper:    {nu} frames")
            print(f"  Events:   {len(self.telemetry.events)}")
            self.telemetry.plot(save_to_file=True)
