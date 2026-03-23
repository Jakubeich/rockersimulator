"""
Pygame renderer — draws two vehicles, trajectories, landing pad, HUD.

Coordinate: world +x right, +y up (metres). Screen +x right, +y down (px).
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np
import pygame

from config import PadConfig, RenderConfig, StageConfig
from rocket import EngineState, FlightPhase, Vehicle
from utils import clamp, direction_from_angle, magnitude


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class Camera:
    def __init__(self, cfg: RenderConfig) -> None:
        self.cfg = cfg
        self.zoom: float = cfg.pixels_per_meter
        self.target_x: float = 0.0
        self.target_y: float = 0.0
        # Smoothed theta for visual offset — prevents PID oscillation jitter
        self._smooth_theta: float = math.pi / 2
        self._initialized: bool = False

    def world_to_screen(self, wx: float, wy: float) -> tuple[int, int]:
        sx = self.cfg.screen_width / 2 + (wx - self.target_x) * self.zoom
        sy = self.cfg.screen_height / 2 - (wy - self.target_y) * self.zoom
        return round(sx), round(sy)

    def world_to_screen_f(self, wx: float, wy: float) -> tuple[float, float]:
        """Float version — avoids rounding jitter for vehicle drawing."""
        sx = self.cfg.screen_width / 2 + (wx - self.target_x) * self.zoom
        sy = self.cfg.screen_height / 2 - (wy - self.target_y) * self.zoom
        return sx, sy

    def zoom_in(self) -> None:
        self.zoom = min(self.zoom * self.cfg.zoom_speed, self.cfg.max_zoom)

    def zoom_out(self) -> None:
        self.zoom = max(self.zoom / self.cfg.zoom_speed, self.cfg.min_zoom)

    def follow(self, v: Vehicle, visual_offset: float = 0.0) -> None:
        # Use raw theta — the position smoothing handles all jitter
        a = v.state.theta
        tx = v.state.x + math.cos(a) * visual_offset
        ty = v.state.y + math.sin(a) * visual_offset

        # Zoom-adaptive smoothing:
        #   high zoom (close up) → instant follow (no visible lag)
        #   low zoom (far away)  → gentle smoothing (cinematic)
        smoothing = clamp(self.zoom * 0.8, 0.12, 1.0)

        dx = tx - self.target_x
        dy = ty - self.target_y

        # Large jumps (focus switch) → snap faster
        dist_px = math.hypot(dx, dy) * self.zoom
        if dist_px > 500:
            smoothing = 1.0

        self.target_x += dx * smoothing
        self.target_y += dy * smoothing

    def auto_zoom(self, v: Vehicle) -> None:
        extent = max(abs(v.altitude), abs(v.state.x), 50.0)
        desired = self.cfg.screen_height * 0.4 / extent
        desired = clamp(desired, self.cfg.min_zoom, self.cfg.max_zoom)
        # Very gentle zoom changes — zooming out slower to reduce jitter
        rate = 0.015 if desired < self.zoom else 0.01
        new_zoom = self.zoom + (desired - self.zoom) * rate
        # Snap when change is sub-perceptual
        if abs(new_zoom - self.zoom) / max(self.zoom, 0.001) < 0.0001:
            self.zoom = desired
        else:
            self.zoom = new_zoom


# ---------------------------------------------------------------------------
# Particle system for exhaust
# ---------------------------------------------------------------------------

class Particle:
    __slots__ = ('x', 'y', 'vx', 'vy', 'life', 'max_life', 'size', 'color')

    def __init__(self, x, y, vx, vy, life, size, color):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.size = size
        self.color = color


class ParticleSystem:
    def __init__(self, max_particles: int = 300) -> None:
        self._particles: list[Particle] = []
        self._max = max_particles

    def emit(self, x, y, vx, vy, count=1, life=1.0, size=3.0,
             color=(255, 200, 100)) -> None:
        for _ in range(count):
            spread = 0.3
            pvx = vx + random.uniform(-spread, spread) * abs(vx + 1)
            pvy = vy + random.uniform(-spread, spread) * abs(vy + 1)
            p = Particle(x, y, pvx, pvy,
                         life * random.uniform(0.6, 1.0),
                         size * random.uniform(0.5, 1.5), color)
            self._particles.append(p)
        # Trim
        if len(self._particles) > self._max:
            self._particles = self._particles[-self._max:]

    def update(self, dt: float) -> None:
        alive = []
        for p in self._particles:
            p.life -= dt
            if p.life > 0:
                p.x += p.vx * dt
                p.y += p.vy * dt
                p.vy -= 2.0 * dt  # slight gravity on particles
                p.size *= (1.0 - 0.5 * dt)
                alive.append(p)
        self._particles = alive

    def draw(self, screen: pygame.Surface, camera: Camera) -> None:
        for p in self._particles:
            sx, sy = camera.world_to_screen(p.x, p.y)
            if not (0 <= sx < screen.get_width() and 0 <= sy < screen.get_height()):
                continue
            t = p.life / p.max_life
            alpha = int(180 * t)
            r, g, b = p.color
            r = min(255, int(r * t + 80 * (1 - t)))
            g = min(255, int(g * t * 0.6))
            b = min(255, int(b * t * 0.3))
            sz = max(round(p.size * camera.zoom * 0.5), 1)
            if sz <= 1:
                screen.set_at((sx, sy), (r, g, b))
            else:
                surf = pygame.Surface((sz * 2, sz * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, (r, g, b, alpha), (sz, sz), sz)
                screen.blit(surf, (sx - sz, sy - sz), special_flags=pygame.BLEND_ADD)


# ---------------------------------------------------------------------------
# Star field
# ---------------------------------------------------------------------------

class StarField:
    def __init__(self, count: int = 200) -> None:
        self._stars = [(random.random(), random.random(),
                        random.uniform(0.3, 1.0), random.randint(1, 2))
                       for _ in range(count)]
        self._twinkle_t: float = 0.0

    def draw(self, screen: pygame.Surface, camera_y: float) -> None:
        self._twinkle_t += 0.016
        w, h = screen.get_width(), screen.get_height()
        # Stars fade in above 10km, full above 50km
        fade = clamp((camera_y - 10_000) / 40_000, 0.0, 1.0)
        if fade < 0.01:
            return
        for rx, ry, brightness, size in self._stars:
            sx = int(rx * w)
            sy = int(ry * h * 0.7)
            twinkle = 0.7 + 0.3 * math.sin(self._twinkle_t * brightness * 3 + rx * 100)
            alpha = int(255 * fade * brightness * twinkle)
            c = (alpha, alpha, min(255, alpha + 30))
            if size <= 1:
                screen.set_at((sx, sy), c)
            else:
                pygame.draw.circle(screen, c, (sx, sy), size)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class Renderer:
    def __init__(self, cfg: RenderConfig, pad_cfg: PadConfig,
                 booster_cfg: StageConfig, upper_cfg: StageConfig) -> None:
        self.cfg = cfg
        self.pad = pad_cfg
        self.booster_cfg = booster_cfg
        self.upper_cfg = upper_cfg
        self.screen = pygame.display.set_mode(
            (cfg.screen_width, cfg.screen_height),
            pygame.RESIZABLE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Rocket Simulator")
        self.camera = Camera(cfg)

        # Fonts
        self.font = pygame.font.SysFont("consolas", 14)
        self.font_large = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_title = pygame.font.SysFont("consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 12)

        self._auto_zoom = True
        self._focus: str = "booster"

        # Visual systems
        self._stars = StarField(250)
        self._particles = ParticleSystem(500)
        self._frame_count = 0

    # ----- Main draw ---------------------------------------------------------

    def draw(
        self,
        booster: Vehicle,
        upper: Vehicle,
        booster_trail: Sequence[tuple[float, float]],
        upper_trail: Sequence[tuple[float, float]],
        sequencer: object,
        paused: bool,
        render_alpha: float = 0.0,
    ) -> None:
        # Interpolate states for smooth rendering (fix-your-timestep)
        self._apply_interpolation(booster, render_alpha)
        self._apply_interpolation(upper, render_alpha)

        focus = booster if self._focus == "booster" else upper
        focus_cfg = self.booster_cfg if self._focus == "booster" else self.upper_cfg
        if not focus.is_active:
            focus = booster if booster.is_active else upper
            focus_cfg = self.booster_cfg if focus is booster else self.upper_cfg

        self.camera.follow(focus, visual_offset=focus_cfg.length / 2)
        if self._auto_zoom:
            self.camera.auto_zoom(focus)

        self._frame_count += 1

        # Update particles
        self._particles.update(1 / 60)

        # Emit exhaust particles
        self._emit_exhaust(booster, self.booster_cfg)
        self._emit_exhaust(upper, self.upper_cfg)

        # Draw layers
        self._draw_sky(focus)
        self._stars.draw(self.screen, self.camera.target_y)
        self._draw_atmosphere_glow(focus)
        self._draw_ground()
        self._draw_landing_pad(booster, upper)

        if self.cfg.show_trajectory:
            self._draw_trail(booster_trail, self.cfg.booster_trail_color, "B")
            self._draw_trail(upper_trail, self.cfg.upper_trail_color, "U")

        # Particles behind vehicles
        self._particles.draw(self.screen, self.camera)

        # Draw vehicles
        if booster.is_active and booster.flight_phase != FlightPhase.CRASHED:
            self._draw_vehicle(booster, self.booster_cfg, self.cfg.booster_color)
            if self.cfg.show_vectors and not booster.is_terminated:
                self._draw_vectors(booster, self.booster_cfg)
        if upper.is_active and upper.flight_phase != FlightPhase.CRASHED:
            self._draw_vehicle(upper, self.upper_cfg, self.cfg.upper_color)
            if self.cfg.show_vectors and not upper.is_terminated:
                self._draw_vectors(upper, self.upper_cfg)

        # Re-entry glow on top
        self._draw_reentry_glow(booster, self.booster_cfg)
        self._draw_reentry_glow(upper, self.upper_cfg)

        # UI
        self._draw_hud(booster, upper, sequencer, paused)
        if self.cfg.show_debug:
            self._draw_debug(focus)
        if paused:
            self._draw_paused()
        self._draw_controls()
        self._draw_end_status(booster, upper)

        pygame.display.flip()

        # Restore original states after rendering
        self._restore_state(booster)
        self._restore_state(upper)

    def _apply_interpolation(self, v: Vehicle, alpha: float) -> None:
        """Swap vehicle state with interpolated state for rendering."""
        if alpha <= 0 or v._prev_state is None or v.is_terminated:
            v._render_backup = None
            return
        v._render_backup = v.state  # type: ignore[attr-defined]
        v.state = v.render_state(alpha)

    @staticmethod
    def _restore_state(v: Vehicle) -> None:
        backup = getattr(v, '_render_backup', None)
        if backup is not None:
            v.state = backup
            v._render_backup = None  # type: ignore[attr-defined]

    # ----- Background --------------------------------------------------------

    def _draw_sky(self, focus: Vehicle) -> None:
        w, h = self.cfg.screen_width, self.cfg.screen_height
        alt = max(self.camera.target_y, 0)

        # Altitude-based sky transition
        # Ground level: blue sky, 50km+: black space
        space_factor = clamp(alt / 50_000, 0.0, 1.0)

        # Ground-level colors
        top_ground = (15, 25, 80)
        bot_ground = (60, 120, 200)
        # Space colors
        top_space = (0, 0, 3)
        bot_space = (2, 4, 15)

        top = tuple(int(top_ground[i] * (1 - space_factor) + top_space[i] * space_factor) for i in range(3))
        bot = tuple(int(bot_ground[i] * (1 - space_factor) + bot_space[i] * space_factor) for i in range(3))

        # Draw gradient with bands
        band_h = max(h // 40, 2)
        for y in range(0, h, band_h):
            t = y / h
            c = tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3))
            pygame.draw.rect(self.screen, c, (0, y, w, band_h))

    def _draw_atmosphere_glow(self, focus: Vehicle) -> None:
        """Draw horizon glow at the edge of atmosphere."""
        alt = self.camera.target_y
        if alt < 20_000:
            return
        _, gy_f = self.camera.world_to_screen_f(0, 0)
        gy = round(gy_f)
        if gy > self.cfg.screen_height + 100:
            return
        # Thin blue line at horizon
        intensity = clamp((alt - 20_000) / 80_000, 0.0, 0.8)
        glow_h = max(int(30 * intensity), 2)
        glow_surf = pygame.Surface((self.cfg.screen_width, glow_h * 2), pygame.SRCALPHA)
        for i in range(glow_h):
            t = 1.0 - i / glow_h
            alpha = int(60 * t * intensity)
            pygame.draw.line(glow_surf, (100, 160, 255, alpha),
                             (0, glow_h + i), (self.cfg.screen_width, glow_h + i))
            pygame.draw.line(glow_surf, (100, 160, 255, alpha // 2),
                             (0, glow_h - i), (self.cfg.screen_width, glow_h - i))
        self.screen.blit(glow_surf, (0, gy - glow_h))

    def _draw_ground(self) -> None:
        _, gy_f = self.camera.world_to_screen_f(0, 0)
        gy = round(gy_f)
        w, h = self.cfg.screen_width, self.cfg.screen_height
        if gy >= h:
            return

        # Ground gradient: dark at depth, lighter at surface
        ground_h = h - gy
        if ground_h > 0:
            for i in range(min(ground_h, h)):
                t = i / max(ground_h, 1)
                r = int(35 + 15 * (1 - t))
                g = int(90 + 40 * (1 - t))
                b = int(35 + 15 * (1 - t))
                pygame.draw.line(self.screen, (r, g, b),
                                 (0, gy + i), (w, gy + i))

        # Surface line with highlight
        pygame.draw.line(self.screen, (70, 160, 70), (0, gy), (w, gy), 2)
        pygame.draw.line(self.screen, (90, 200, 90), (0, gy - 1), (w, gy - 1), 1)

    def _draw_landing_pad(self, booster: Vehicle, upper: Vehicle) -> None:
        # Determine which vehicle (if any) is at the launch pad
        launch_vehicle_len = self.booster_cfg.length  # default tower size
        at_launch = None
        # Check if upper stage has landed at launch pad
        if (upper.flight_phase == FlightPhase.LANDED
                and abs(upper.state.x - self.pad.launch_x) < self.pad.pad_width):
            at_launch = self.upper_cfg
        # Before separation, booster is at launch pad
        elif (not booster.has_left_ground
              or (booster.flight_phase == FlightPhase.PRE_LAUNCH)):
            at_launch = self.booster_cfg

        self._draw_pad_at(self.pad.launch_x, is_launch=True,
                          vehicle_cfg=at_launch,
                          tower_ref_length=self.booster_cfg.length)
        self._draw_pad_at(self.pad.landing_x, is_launch=False)

    def _draw_pad_at(self, world_x: float, is_launch: bool,
                     vehicle_cfg: StageConfig | None = None,
                     tower_ref_length: float = 35.0) -> None:
        px, py = self.camera.world_to_screen(world_x, self.pad.pad_height / 2)
        _, gy = self.camera.world_to_screen(0, 0)
        hw = max(int(self.pad.pad_width / 2 * self.camera.zoom), 10)
        hh = max(int(self.pad.pad_height * self.camera.zoom), 2)

        # Pad base with 3D effect
        base_dark = (50, 50, 55) if is_launch else (40, 40, 45)
        base_light = (80, 80, 85) if is_launch else (65, 65, 70)
        rect = pygame.Rect(px - hw, py - hh, hw * 2, hh * 2)
        pygame.draw.rect(self.screen, base_dark, rect)
        # Top highlight
        pygame.draw.rect(self.screen, base_light,
                         pygame.Rect(px - hw, py - hh, hw * 2, max(hh, 1)))
        # Border
        pygame.draw.rect(self.screen, (110, 110, 120), rect, 1)

        # Hazard stripes on edges
        stripe_w = max(int(2 * self.camera.zoom), 1)
        if stripe_w >= 1 and hh >= 2:
            for i in range(0, hw * 2, stripe_w * 4):
                sx = px - hw + i
                pygame.draw.line(self.screen, (180, 160, 0),
                                 (sx, py - hh), (sx + stripe_w * 2, py + hh), 1)

        if is_launch:
            self._draw_launch_tower(px, py, hw, hh, vehicle_cfg, tower_ref_length)
        else:
            # Landing zone: SpaceX-style target
            radius = max(hw * 2 // 3, 6)
            lw = max(int(1.5 * self.camera.zoom), 1)
            # Outer circle
            pygame.draw.circle(self.screen, (180, 180, 190), (px, py), radius, lw)
            # Middle circle
            mid_r = max(radius * 2 // 3, 4)
            pygame.draw.circle(self.screen, (200, 60, 60), (px, py), mid_r, lw)
            # Inner filled circle
            inner_r = max(radius // 3, 3)
            pygame.draw.circle(self.screen, (220, 70, 70), (px, py), inner_r)
            # X marking
            xr = radius * 2 // 3
            pygame.draw.line(self.screen, (200, 60, 60),
                             (px - xr, py - xr), (px + xr, py + xr), lw)
            pygame.draw.line(self.screen, (200, 60, 60),
                             (px + xr, py - xr), (px - xr, py + xr), lw)
            # Label
            if hw > 15:
                label = self.font_small.render("LANDING ZONE", True, (140, 150, 160))
                self.screen.blit(label, (px - label.get_width() // 2, py + hh + 4))

    def _draw_launch_tower(self, px: int, py: int, hw: int, hh: int,
                           vehicle_cfg: StageConfig | None,
                           tower_ref_length: float) -> None:
        """Draw Mechazilla-style launch tower with chopstick arms."""
        zoom = self.camera.zoom

        # Tower height based on booster length (+ nose + margin)
        tower_world_h = tower_ref_length * 1.15
        tower_h = max(int(tower_world_h * zoom), 8)
        tower_w = max(int(4 * zoom), 2)

        # Tower position: right edge of pad
        tx = px + hw - tower_w

        # Tower body — two vertical rails with cross-bracing
        rail_w = max(int(1.5 * zoom), 1)
        rail_gap = max(tower_w - rail_w * 2, 1)

        # Left rail
        for i in range(rail_w):
            c = (85, 85, 95)
            pygame.draw.line(self.screen, c,
                             (tx + i, py - hh - tower_h), (tx + i, py - hh))
        # Right rail
        for i in range(rail_w):
            c = (95, 95, 105)
            pygame.draw.line(self.screen, c,
                             (tx + rail_w + rail_gap + i, py - hh - tower_h),
                             (tx + rail_w + rail_gap + i, py - hh))

        # Cross bracing
        brace_spacing = max(int(8 * zoom), 4)
        brace_w = max(int(1 * zoom), 1)
        for y_off in range(0, tower_h, brace_spacing):
            by = py - hh - y_off
            pygame.draw.line(self.screen, (70, 70, 80),
                             (tx, by), (tx + tower_w, by - brace_spacing // 2), brace_w)

        # Tower top cap
        cap_h = max(int(3 * zoom), 2)
        pygame.draw.rect(self.screen, (130, 130, 140),
                         (tx - 2, py - hh - tower_h - cap_h, tower_w + 4, cap_h))

        # Red warning light (blinking)
        if self._frame_count % 60 < 30:
            light_r = max(int(2 * zoom), 2)
            light_pos = (tx + tower_w // 2, py - hh - tower_h - cap_h - light_r - 1)
            pygame.draw.circle(self.screen, (255, 30, 30), light_pos, light_r)

        # --- Chopstick arms (at grid fin height of vehicle on pad) ---
        if vehicle_cfg is not None:
            # Grid fins are at 0.75 * length above nozzle (nozzle at pad surface)
            gridfin_world_y = self.pad.pad_height + vehicle_cfg.length * 0.75
            _, arm_y = self.camera.world_to_screen(0, gridfin_world_y)

            arm_reach = max(int((self.pad.pad_width * 0.4) * zoom), 6)
            arm_thick = max(int(2.5 * zoom), 2)
            chop_gap = max(int(vehicle_cfg.diameter * 0.6 * zoom), 3)

            # Upper chopstick
            upper_arm_y = arm_y - chop_gap // 2
            pygame.draw.line(self.screen, (120, 100, 80),
                             (tx, upper_arm_y),
                             (tx - arm_reach, upper_arm_y), arm_thick)
            # Grip notch on upper arm (angled down)
            notch = max(int(3 * zoom), 2)
            pygame.draw.line(self.screen, (140, 120, 90),
                             (tx - arm_reach, upper_arm_y),
                             (tx - arm_reach + notch, upper_arm_y + notch), arm_thick)

            # Lower chopstick
            lower_arm_y = arm_y + chop_gap // 2
            pygame.draw.line(self.screen, (110, 90, 70),
                             (tx, lower_arm_y),
                             (tx - arm_reach, lower_arm_y), arm_thick)
            # Grip notch on lower arm (angled up)
            pygame.draw.line(self.screen, (130, 110, 80),
                             (tx - arm_reach, lower_arm_y),
                             (tx - arm_reach + notch, lower_arm_y - notch), arm_thick)

        # Label
        if hw > 15:
            label = self.font_small.render("LAUNCH PAD", True, (140, 150, 160))
            self.screen.blit(label, (px - label.get_width() // 2, py + hh + 4))

    # ----- Trail -------------------------------------------------------------

    def _draw_trail(self, pts: Sequence[tuple[float, float]],
                    color: tuple[int, int, int], label: str = "") -> None:
        if len(pts) < 2:
            return
        screen_pts = [self.camera.world_to_screen(x, y) for x, y in pts]
        # Fade trail: older = dimmer
        n = len(screen_pts)
        if n > 3:
            # Draw in segments with fading alpha
            step = max(1, n // 50)
            for i in range(0, n - 1, step):
                t = i / n
                alpha = int(40 + 180 * t)
                c = (min(255, color[0] * alpha // 255),
                     min(255, color[1] * alpha // 255),
                     min(255, color[2] * alpha // 255))
                end = min(i + step + 1, n)
                if end - i >= 2:
                    pygame.draw.lines(self.screen, c, False, screen_pts[i:end], 1)
        else:
            pygame.draw.lines(self.screen, color, False, screen_pts, 1)

    # ----- Vehicle -----------------------------------------------------------

    def _draw_vehicle(self, v: Vehicle, scfg: StageConfig,
                      color: tuple[int, int, int]) -> None:
        # Offset visual center by +length/2 along axis so nozzle sits at state.y
        a_off = v.state.theta
        vis_x = v.state.x + math.cos(a_off) * scfg.length / 2
        vis_y = v.state.y + math.sin(a_off) * scfg.length / 2
        # Use float center to avoid per-vertex rounding jitter
        fcx, fcy = self.camera.world_to_screen_f(vis_x, vis_y)
        cx, cy = round(fcx), round(fcy)
        hl = max(scfg.length / 2 * self.camera.zoom, 10)
        hw = max(scfg.diameter / 2 * self.camera.zoom, 2.5)
        a = v.state.theta
        ca, sa = math.cos(a), math.sin(a)

        def l2s(lx: float, ly: float) -> tuple[float, float]:
            """Local-to-screen using float center — round only at draw time."""
            return (fcx + ca * lx - sa * ly, fcy - sa * lx - ca * ly)

        def l2si(lx: float, ly: float) -> tuple[int, int]:
            """Integer version for APIs that need it."""
            x, y = l2s(lx, ly)
            return round(x), round(y)

        # Landing legs — smooth deployment based on altitude, no hard threshold
        leg_factor = 0.0
        if v.flight_phase == FlightPhase.LANDED:
            leg_factor = 1.0
        elif v.altitude < 2000 and v.has_left_ground:
            leg_factor = clamp(1.0 - v.altitude / 2000, 0.0, 1.0)
        if leg_factor > 0.01:
            leg_len = hw * 2.5 * leg_factor
            leg_w = max(round(1.5 * self.camera.zoom), 1)
            for sign in (1, -1):
                base = l2si(-hl * 0.9, sign * hw * 0.5)
                tip = l2si(-hl * 0.9 - leg_len * 0.5, sign * (hw + leg_len))
                pygame.draw.line(self.screen, (140, 140, 150), base, tip, leg_w)
                foot = l2si(-hl * 0.9 - leg_len * 0.3, sign * (hw + leg_len + hw * 0.3))
                pygame.draw.line(self.screen, (140, 140, 150), tip, foot, leg_w)

        # Body — tapered shape
        body_pts = [
            l2s(hl * 0.7, hw),           # top right of body
            l2s(-hl, hw * 0.9),          # bottom right
            l2s(-hl, -hw * 0.9),         # bottom left
            l2s(hl * 0.7, -hw),          # top left of body
        ]
        pygame.draw.polygon(self.screen, color, body_pts)
        # Body highlight stripe
        stripe = [
            l2s(hl * 0.5, hw * 0.3),
            l2s(-hl * 0.5, hw * 0.3),
            l2s(-hl * 0.5, hw * 0.1),
            l2s(hl * 0.5, hw * 0.1),
        ]
        r, g, b = color
        highlight = (min(255, r + 40), min(255, g + 40), min(255, b + 40))
        pygame.draw.polygon(self.screen, highlight, stripe)

        # Interstage / tank line
        tank_y = hl * 0.1
        tank_line_a = l2s(tank_y, hw * 0.95)
        tank_line_b = l2s(tank_y, -hw * 0.95)
        pygame.draw.line(self.screen, (min(255, r + 20), min(255, g + 20), min(255, b + 20)),
                         tank_line_a, tank_line_b, 1)

        # Nose cone — smooth ogive shape
        nose_steps = 6
        nose_pts_right = []
        nose_pts_left = []
        for i in range(nose_steps + 1):
            t = i / nose_steps
            nx = hl * 0.7 + hw * 1.8 * t
            ny = hw * math.cos(t * math.pi / 2)
            nose_pts_right.append(l2s(nx, ny))
            nose_pts_left.append(l2s(nx, -ny))
        nose_all = nose_pts_right + list(reversed(nose_pts_left))
        # Nose color gradient
        nose_base = (200, 50, 50)
        pygame.draw.polygon(self.screen, nose_base, nose_all)
        # Nose tip highlight
        if len(nose_pts_right) > 2:
            tip_pts = nose_pts_right[-3:] + nose_pts_left[-3:][::-1]
            pygame.draw.polygon(self.screen, (230, 80, 80), tip_pts)

        # Grid fins — smooth deployment (no hard threshold flicker)
        if v.has_left_ground:
            # Smoothly interpolate: fully deployed when descending, half when ascending
            fin_deploy = clamp(0.75 - v.vertical_speed * 0.005, 0.5, 1.0)
            fs = hw * 1.2 * fin_deploy
            fw = max(round(1.5 * self.camera.zoom), 1)
            for sign in (1, -1):
                # Grid fin shape
                fin = [
                    l2s(hl * 0.5, sign * hw),
                    l2s(hl * 0.5 + fs * 0.3, sign * (hw + fs)),
                    l2s(hl * 0.5 - fs * 0.3, sign * (hw + fs * 0.7)),
                ]
                pygame.draw.polygon(self.screen, (160, 160, 170), fin)
                pygame.draw.polygon(self.screen, (130, 130, 140), fin, 1)

        # Engine nozzle
        nozzle_w = hw * 0.7
        nozzle_h = hw * 0.4
        nozzle = [
            l2s(-hl, nozzle_w * 0.6),
            l2s(-hl - nozzle_h, nozzle_w),
            l2s(-hl - nozzle_h, -nozzle_w),
            l2s(-hl, -nozzle_w * 0.6),
        ]
        pygame.draw.polygon(self.screen, (80, 80, 90), nozzle)
        pygame.draw.polygon(self.screen, (60, 60, 70), nozzle, 1)

        # Flame
        if v.engine_state == EngineState.BURNING and v.throttle > 0:
            self._draw_flame(v, scfg, fcx, fcy, hl, hw, a)

    def _draw_flame(self, v: Vehicle, scfg: StageConfig,
                    cx: float, cy: float, hl: float, hw: float,
                    angle: float) -> None:
        throttle = v.throttle
        fa = angle + math.pi + v.gimbal_angle
        fc, fs = math.cos(fa), math.sin(fa)
        bx = math.cos(angle) * (-hl)
        by = math.sin(angle) * (-hl)

        def fp(lx, ly):
            return (cx + bx + fc * lx - fs * ly,
                    cy - by - fs * lx - fc * ly)

        # Gentle flicker — slower frequency, smaller amplitude to reduce visual noise
        flicker = 0.92 + 0.08 * math.sin(self._frame_count * 0.3)
        flicker2 = 0.94 + 0.06 * math.sin(self._frame_count * 0.5 + 1.0)

        # Outer flame (orange-red)
        fl = hl * 0.7 * throttle * flicker
        fw = hw * 0.9 * throttle
        outer = [fp(0, fw), fp(fl, fw * 0.1), fp(fl * 1.1, 0),
                 fp(fl, -fw * 0.1), fp(0, -fw)]
        pygame.draw.polygon(self.screen, (255, 100, 20), outer)

        # Middle flame (orange-yellow)
        fl2 = fl * 0.75 * flicker2
        fw2 = fw * 0.6
        mid = [fp(0, fw2), fp(fl2, fw2 * 0.05), fp(fl2 * 1.05, 0),
               fp(fl2, -fw2 * 0.05), fp(0, -fw2)]
        pygame.draw.polygon(self.screen, (255, 180, 50), mid)

        # Core flame (white-yellow)
        fl3 = fl * 0.45
        fw3 = fw * 0.3
        core = [fp(0, fw3), fp(fl3, 0), fp(0, -fw3)]
        pygame.draw.polygon(self.screen, (255, 255, 200), core)

        # Mach diamonds (at high throttle)
        if throttle > 0.5:
            for i in range(3):
                dx = fl * (0.2 + i * 0.15)
                dy = fw * 0.15 * (1 - i * 0.2)
                diamond = [fp(dx - dy, 0), fp(dx, dy),
                           fp(dx + dy, 0), fp(dx, -dy)]
                alpha = int(150 * throttle * (1 - i * 0.3))
                pygame.draw.polygon(self.screen, (255, 220, min(255, 100 + alpha)), diamond)

    def _emit_exhaust(self, v: Vehicle, scfg: StageConfig) -> None:
        if v.engine_state != EngineState.BURNING or v.throttle < 0.1:
            return
        if not v.is_active:
            return
        a = v.state.theta
        # Emit from engine base (nozzle is at state position after visual offset)
        ex = v.state.x
        ey = v.state.y
        # Exhaust velocity (opposite to thrust)
        ea = a + math.pi + v.gimbal_angle
        speed = 50.0 * v.throttle
        evx = math.cos(ea) * speed + v.state.vx * 0.1
        evy = math.sin(ea) * speed + v.state.vy * 0.1

        count = max(1, int(v.throttle * 3))
        # Smoke (gray)
        self._particles.emit(ex, ey, evx * 0.3, evy * 0.3,
                             count=count, life=1.5, size=4.0,
                             color=(180, 160, 140))

    # ----- Re-entry glow -----------------------------------------------------

    def _draw_reentry_glow(self, v: Vehicle, scfg: StageConfig) -> None:
        if v.speed < 400 or v.rho < 0.0001:
            return
        intensity = clamp(v.q / 80_000, 0.0, 1.0)
        if intensity < 0.03:
            return

        a = v.state.theta
        vis_x = v.state.x + math.cos(a) * scfg.length / 2
        vis_y = v.state.y + math.sin(a) * scfg.length / 2
        cx, cy = self.camera.world_to_screen(vis_x, vis_y)
        # Glow oriented along velocity — use round() for stable sizing
        radius = max(round(20 * intensity * self.camera.zoom * 5), 4)
        radius = min(radius, 80)

        # Multi-layer glow
        for layer in range(3):
            r = radius * (3 - layer) // 3
            if r < 2:
                continue
            glow_surf = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
            if layer == 0:
                c = (255, 60, 10, int(50 * intensity))
            elif layer == 1:
                c = (255, 140, 30, int(80 * intensity))
            else:
                c = (255, 220, 100, int(100 * intensity))
            pygame.draw.circle(glow_surf, c, (r * 2, r * 2), r)
            self.screen.blit(glow_surf,
                             (cx - r * 2, cy - r * 2),
                             special_flags=pygame.BLEND_ADD)

    # ----- Vectors -----------------------------------------------------------

    def _draw_vectors(self, v: Vehicle, scfg: StageConfig | None = None) -> None:
        if scfg is not None:
            a = v.state.theta
            vis_x = v.state.x + math.cos(a) * scfg.length / 2
            vis_y = v.state.y + math.sin(a) * scfg.length / 2
        else:
            vis_x, vis_y = v.state.x, v.state.y
        cx, cy = self.camera.world_to_screen(vis_x, vis_y)
        if v.speed > 1.0:
            s = min(80, v.speed * 0.3)
            vd = v.velocity / v.speed
            ex, ey = round(cx + vd[0] * s), round(cy - vd[1] * s)
            pygame.draw.line(self.screen, self.cfg.velocity_vector_color,
                             (cx, cy), (ex, ey), 2)
            _draw_arrowhead(self.screen, (cx, cy), (ex, ey),
                            self.cfg.velocity_vector_color, 6)
        if v.current_thrust > 0:
            ta = v.state.theta + v.gimbal_angle
            td = direction_from_angle(ta)
            s = min(60, v.current_thrust / v.cfg.max_thrust * 60)
            ex, ey = round(cx + td[0] * s), round(cy - td[1] * s)
            pygame.draw.line(self.screen, self.cfg.thrust_vector_color,
                             (cx, cy), (ex, ey), 2)
            _draw_arrowhead(self.screen, (cx, cy), (ex, ey),
                            self.cfg.thrust_vector_color, 6)

    # ----- HUD ---------------------------------------------------------------

    def _draw_hud(self, b: Vehicle, u: Vehicle, seq: object, paused: bool) -> None:
        mt = getattr(seq, 'mission_time', 0.0)
        pname = getattr(seq, 'phase_name', '--')
        sep = getattr(seq, 'separated', False)

        focus_v = b if self._focus == "booster" else u
        focus_label = "BOOSTER" if self._focus == "booster" else "UPPER STAGE"

        # Semi-transparent HUD background
        panel_w = 280
        panel_h = 340
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 140))
        # Border
        pygame.draw.rect(panel, (60, 80, 120, 200), (0, 0, panel_w, panel_h), 1)
        self.screen.blit(panel, (8, 8))

        x, y = 16, 14

        # Mission time
        t_str = f"T{mt:+.1f}s" if mt < 0 else f"T+{mt:.1f}s"
        self._hud_text(t_str, x, y, self.font_large, (100, 200, 255))
        y += 24

        # Phase
        self._hud_text(pname, x, y, self.font, (255, 200, 80))
        y += 22

        # Focus label
        self._hud_text(f">> {focus_label}", x, y, self.font,
                       (100, 180, 255) if self._focus == "booster" else (255, 180, 100))
        y += 22

        if focus_v.is_active:
            # Separator
            pygame.draw.line(self.screen, (60, 80, 120), (x, y), (x + panel_w - 20, y), 1)
            y += 6

            # Telemetry
            alt_km = focus_v.altitude / 1000
            alt_str = f"{alt_km:.2f} km" if alt_km >= 1 else f"{focus_v.altitude:.0f} m"
            self._hud_row("ALT", alt_str, x, y); y += 18
            self._hud_row("SPEED", f"{focus_v.speed:.1f} m/s", x, y); y += 18
            self._hud_row("V/S", f"{focus_v.vertical_speed:+.1f} m/s", x, y,
                          val_color=(100, 255, 100) if focus_v.vertical_speed > 0 else (255, 100, 100)); y += 18
            self._hud_row("H/S", f"{focus_v.horizontal_speed:+.1f} m/s", x, y); y += 18

            y += 4
            pygame.draw.line(self.screen, (60, 80, 120), (x, y), (x + panel_w - 20, y), 1)
            y += 6

            # Engine info
            self._hud_row("THRTL", f"{focus_v.throttle * 100:.0f}%", x, y,
                          val_color=(100, 255, 100) if focus_v.throttle > 0 else (120, 120, 120)); y += 18
            self._hud_row("ENGINE", focus_v.engine_state.name, x, y,
                          val_color=(100, 255, 100) if focus_v.engine_state == EngineState.BURNING else (180, 180, 180)); y += 18

            # Fuel bar
            y += 4
            self._hud_row("FUEL", f"{focus_v.state.fuel:.0f} kg", x, y); y += 16
            bar_w = panel_w - 30
            bar_h = 10
            fuel_frac = focus_v.fuel_fraction
            # Background
            pygame.draw.rect(self.screen, (40, 40, 50), (x, y, bar_w, bar_h))
            # Fill
            if fuel_frac > 0.3:
                bar_color = (60, 200, 60)
            elif fuel_frac > 0.1:
                bar_color = (220, 180, 40)
            else:
                bar_color = (220, 60, 40)
            fill_w = int(bar_w * fuel_frac)
            if fill_w > 0:
                pygame.draw.rect(self.screen, bar_color, (x, y, fill_w, bar_h))
            pygame.draw.rect(self.screen, (100, 100, 120), (x, y, bar_w, bar_h), 1)
            y += 16

            # Dynamic pressure
            self._hud_row("Q", f"{focus_v.q:.0f} Pa", x, y); y += 18
            self._hud_row("PHASE", focus_v.flight_phase.name, x, y); y += 18
        else:
            y += 8
            self._hud_text("INACTIVE", x, y, self.font, (120, 120, 120))
            y += 20

        # Other vehicle mini status
        other = u if self._focus == "booster" else b
        other_label = "Upper" if self._focus == "booster" else "Booster"
        if other.is_active and sep:
            y = panel_h - 12
            status = f"TERMINATED" if other.is_terminated else other.flight_phase.name
            c = (255, 80, 80) if other.is_terminated and other.flight_phase == FlightPhase.CRASHED else (160, 160, 180)
            txt = f"{other_label}: {other.altitude/1000:.1f}km {other.speed:.0f}m/s {status}"
            self._hud_text(txt, x, y, self.font_small, c)

    def _hud_text(self, text: str, x: int, y: int, font, color) -> None:
        s = font.render(text, True, color)
        self.screen.blit(s, (x, y))

    def _hud_row(self, label: str, value: str, x: int, y: int,
                 label_color=(140, 150, 170), val_color=(220, 220, 230)) -> None:
        lbl = self.font.render(f"{label:>6s}", True, label_color)
        val = self.font.render(f" {value}", True, val_color)
        self.screen.blit(lbl, (x, y))
        self.screen.blit(val, (x + lbl.get_width(), y))

    def _draw_debug(self, v: Vehicle) -> None:
        lines = [
            f"g(h)  {v.g_local:.4f} m/s^2",
            f"rho   {v.rho:.6f} kg/m^3",
            f"MoI   {v.moi:.0f} kg*m^2",
            f"Gimbal{math.degrees(v.gimbal_angle):+6.2f} deg",
            f"Theta {math.degrees(v.state.theta):+7.1f} deg",
            f"Omega {math.degrees(v.state.omega):+7.2f} deg/s",
            f"Zoom  {self.camera.zoom:.5f} px/m",
            f"MaxAlt{v.max_altitude/1000:.1f} km",
            f"MaxSpd{v.max_speed:.0f} m/s",
            f"MaxQ  {v.max_dynamic_pressure:.0f} Pa",
        ]
        # Debug panel
        pw, ph = 240, len(lines) * 16 + 10
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 160))
        pygame.draw.rect(panel, (80, 80, 40, 180), (0, 0, pw, ph), 1)
        dx = self.cfg.screen_width - pw - 8
        self.screen.blit(panel, (dx, 8))

        x = dx + 8
        y = 14
        for line in lines:
            s = self.font_small.render(line, True, (200, 200, 120))
            self.screen.blit(s, (x, y))
            y += 16

    def _draw_paused(self) -> None:
        w, h = self.cfg.screen_width, self.cfg.screen_height
        # Dim overlay
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 80))
        self.screen.blit(overlay, (0, 0))
        # Pause text
        s = self.font_title.render("PAUSED", True, (255, 255, 100))
        r = s.get_rect(center=(w // 2, h // 2 - 20))
        self.screen.blit(s, r)
        s2 = self.font.render("Press P to resume", True, (180, 180, 180))
        r2 = s2.get_rect(center=(w // 2, h // 2 + 10))
        self.screen.blit(s2, r2)

    def _draw_controls(self) -> None:
        controls = "P=Pause  R=Reset  TAB=Focus  +/-=Zoom  T=Trail  V=Vectors  F3=Debug  Z=AutoZoom  1-4=Warp  ESC=Quit"
        # Semi-transparent bar at bottom
        bar_h = 22
        bar = pygame.Surface((self.cfg.screen_width, bar_h), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 120))
        self.screen.blit(bar, (0, self.cfg.screen_height - bar_h))
        s = self.font_small.render(controls, True, (120, 130, 150))
        self.screen.blit(s, (10, self.cfg.screen_height - bar_h + 5))

    def _draw_end_status(self, b: Vehicle, u: Vehicle) -> None:
        parts = []
        if b.is_terminated:
            label = "LANDED" if b.flight_phase == FlightPhase.LANDED else "CRASHED"
            color = (60, 255, 60) if label == "LANDED" else (255, 60, 60)
            parts.append((f"Booster: {label}", color))
        if u.is_terminated:
            label2 = "LANDED" if u.flight_phase == FlightPhase.LANDED else "CRASHED"
            color2 = (60, 255, 60) if label2 == "LANDED" else (255, 60, 60)
            parts.append((f"Upper: {label2} (max {u.max_altitude/1000:.0f}km)", color2))

        if not parts:
            return

        # End status panel
        text = "  |  ".join(p[0] for p in parts) + "  |  R = Restart"
        s = self.font_large.render(text, True, (255, 255, 255))
        pw = s.get_width() + 40
        ph = 40
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        # Color based on best outcome
        if all(p[1] == (60, 255, 60) for p in parts):
            panel.fill((0, 60, 0, 180))
            border = (60, 200, 60, 220)
        else:
            panel.fill((60, 0, 0, 180))
            border = (200, 60, 60, 220)
        pygame.draw.rect(panel, border, (0, 0, pw, ph), 2)

        px = self.cfg.screen_width // 2 - pw // 2
        py = self.cfg.screen_height - 65
        self.screen.blit(panel, (px, py))
        r = s.get_rect(center=(self.cfg.screen_width // 2, py + ph // 2))
        self.screen.blit(s, r)

    # ----- Utility -----------------------------------------------------------

    def toggle_focus(self) -> str:
        self._focus = "upper" if self._focus == "booster" else "booster"
        self.camera._initialized = False  # re-init smooth theta for new target
        return self._focus

    def toggle_auto_zoom(self) -> bool:
        self._auto_zoom = not self._auto_zoom
        return self._auto_zoom

    def disable_auto_zoom(self) -> None:
        self._auto_zoom = False

    def handle_resize(self, w: int, h: int) -> None:
        self.cfg.screen_width = w
        self.cfg.screen_height = h
        self.screen = pygame.display.set_mode(
            (w, h), pygame.RESIZABLE | pygame.DOUBLEBUF)


def _draw_arrowhead(screen, start, end, color, size=6):
    """Draw a small arrowhead at the end of a line."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1:
        return
    dx /= length
    dy /= length
    # Perpendicular
    px, py = -dy, dx
    tip = end
    left = (int(end[0] - dx * size + px * size * 0.4),
            int(end[1] - dy * size + py * size * 0.4))
    right = (int(end[0] - dx * size - px * size * 0.4),
             int(end[1] - dy * size - py * size * 0.4))
    pygame.draw.polygon(screen, color, [tip, left, right])
