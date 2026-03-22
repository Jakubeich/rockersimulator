"""
Pygame renderer — draws two vehicles, trajectories, landing pad, HUD.

Coordinate: world +x right, +y up (metres). Screen +x right, +y down (px).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pygame

from config import LandingPadConfig, RenderConfig, StageConfig
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

    def world_to_screen(self, wx: float, wy: float) -> tuple[int, int]:
        sx = self.cfg.screen_width / 2 + (wx - self.target_x) * self.zoom
        sy = self.cfg.screen_height / 2 - (wy - self.target_y) * self.zoom
        return int(sx), int(sy)

    def zoom_in(self) -> None:
        self.zoom = min(self.zoom * self.cfg.zoom_speed, self.cfg.max_zoom)

    def zoom_out(self) -> None:
        self.zoom = max(self.zoom / self.cfg.zoom_speed, self.cfg.min_zoom)

    def follow(self, v: Vehicle, smoothing: float = 0.12) -> None:
        tx, ty = v.state.x, v.state.y
        speed = (v.state.vx ** 2 + v.state.vy ** 2) ** 0.5
        if speed > 10:
            lead = min(speed * 0.3, 500.0) / max(self.zoom, 0.001)
            tx += v.state.vx / speed * lead * 0.3
            ty += v.state.vy / speed * lead * 0.3
        self.target_x += (tx - self.target_x) * smoothing
        self.target_y += (ty - self.target_y) * smoothing

    def auto_zoom(self, v: Vehicle) -> None:
        extent = max(abs(v.altitude), abs(v.state.x), 50.0)
        desired = self.cfg.screen_height * 0.4 / extent
        desired = clamp(desired, self.cfg.min_zoom, self.cfg.max_zoom)
        rate = 0.04 if desired < self.zoom else 0.02
        self.zoom += (desired - self.zoom) * rate


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class Renderer:
    def __init__(self, cfg: RenderConfig, pad_cfg: LandingPadConfig,
                 booster_cfg: StageConfig, upper_cfg: StageConfig) -> None:
        self.cfg = cfg
        self.pad = pad_cfg
        self.booster_cfg = booster_cfg
        self.upper_cfg = upper_cfg
        self.screen = pygame.display.set_mode(
            (cfg.screen_width, cfg.screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("Rocket Simulator")
        self.camera = Camera(cfg)
        self.font = pygame.font.SysFont("consolas", 14)
        self.font_large = pygame.font.SysFont("consolas", 18)
        self._auto_zoom = True
        self._focus: str = "booster"  # "booster" or "upper"

    # ----- Main draw -----------------------------------------------------

    def draw(
        self,
        booster: Vehicle,
        upper: Vehicle,
        booster_trail: Sequence[tuple[float, float]],
        upper_trail: Sequence[tuple[float, float]],
        sequencer: object,
        paused: bool,
    ) -> None:
        focus = booster if self._focus == "booster" else upper
        if not focus.is_active:
            focus = booster if booster.is_active else upper

        self.camera.follow(focus)
        if self._auto_zoom:
            self.camera.auto_zoom(focus)

        self._draw_sky()
        self._draw_ground()
        self._draw_landing_pad()

        if self.cfg.show_trajectory:
            self._draw_trail(booster_trail, self.cfg.booster_trail_color)
            self._draw_trail(upper_trail, self.cfg.upper_trail_color)

        # Draw vehicles
        if booster.is_active and not booster.is_terminated:
            self._draw_vehicle(booster, self.booster_cfg, self.cfg.booster_color)
            if self.cfg.show_vectors:
                self._draw_vectors(booster)
        if upper.is_active and not upper.is_terminated:
            self._draw_vehicle(upper, self.upper_cfg, self.cfg.upper_color)

        # Re-entry glow
        self._draw_reentry_glow(booster, self.booster_cfg)

        self._draw_hud(booster, upper, sequencer, paused)
        if self.cfg.show_debug:
            self._draw_debug(focus)
        if paused:
            s = self.font_large.render("PAUSED", True, (255, 255, 80))
            self.screen.blit(s, (self.cfg.screen_width // 2 - 40, 10))
        self._draw_controls()
        self._draw_end_status(booster, upper)

        pygame.display.flip()

    # ----- Background ----------------------------------------------------

    def _draw_sky(self) -> None:
        h = self.cfg.screen_height
        top, bot = self.cfg.sky_gradient_top, self.cfg.sky_gradient_bottom
        for y in range(h):
            t = y / h
            c = tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3))
            pygame.draw.line(self.screen, c, (0, y), (self.cfg.screen_width, y))

    def _draw_ground(self) -> None:
        _, gy = self.camera.world_to_screen(0, 0)
        if gy < self.cfg.screen_height:
            pygame.draw.rect(self.screen, self.cfg.ground_color,
                             (0, gy, self.cfg.screen_width, self.cfg.screen_height - gy))
            pygame.draw.line(self.screen, (80, 180, 80),
                             (0, gy), (self.cfg.screen_width, gy), 2)

    def _draw_landing_pad(self) -> None:
        px, py = self.camera.world_to_screen(self.pad.x_position, self.pad.height / 2)
        hw = max(int(self.pad.width / 2 * self.camera.zoom), 8)
        hh = max(int(self.pad.height * self.camera.zoom), 2)
        rect = pygame.Rect(px - hw, py - hh, hw * 2, hh * 2)
        pygame.draw.rect(self.screen, self.cfg.pad_color, rect)
        # X marking
        pygame.draw.line(self.screen, (200, 60, 60),
                         (px - hw // 2, py - hh), (px + hw // 2, py + hh), 2)
        pygame.draw.line(self.screen, (200, 60, 60),
                         (px + hw // 2, py - hh), (px - hw // 2, py + hh), 2)

    # ----- Trail ---------------------------------------------------------

    def _draw_trail(self, pts: Sequence[tuple[float, float]],
                    color: tuple[int, int, int]) -> None:
        if len(pts) < 2:
            return
        screen_pts = [self.camera.world_to_screen(x, y) for x, y in pts]
        pygame.draw.lines(self.screen, color, False, screen_pts, 1)

    # ----- Vehicle -------------------------------------------------------

    def _draw_vehicle(self, v: Vehicle, scfg: StageConfig,
                      color: tuple[int, int, int]) -> None:
        cx, cy = self.camera.world_to_screen(v.state.x, v.state.y)
        hl = max(scfg.length / 2 * self.camera.zoom, 10)
        hw = max(scfg.diameter / 2 * self.camera.zoom, 2.5)
        a = v.state.theta
        ca, sa = math.cos(a), math.sin(a)

        def l2s(lx: float, ly: float) -> tuple[int, int]:
            return int(cx + ca * lx - sa * ly), int(cy - sa * lx - ca * ly)

        # Body
        body = [l2s(hl, hw), l2s(-hl, hw), l2s(-hl, -hw), l2s(hl, -hw)]
        pygame.draw.polygon(self.screen, color, body)
        # Nose
        nose = [l2s(hl + hw * 1.5, 0), l2s(hl, hw), l2s(hl, -hw)]
        pygame.draw.polygon(self.screen, (255, 60, 60), nose)
        # Fins
        fs = hw * 1.5
        for sign in (1, -1):
            fin = [l2s(-hl, sign * hw),
                   l2s(-hl - fs, sign * (hw + fs)),
                   l2s(-hl + fs * 0.5, sign * hw)]
            pygame.draw.polygon(self.screen, (180, 180, 190), fin)

        # Flame
        if v.engine_state == EngineState.BURNING and v.throttle > 0:
            fl = hl * 0.5 * v.throttle + hl * 0.15 * np.random.random() * v.throttle
            fw = hw * 0.8 * v.throttle
            fa = a + math.pi + v.gimbal_angle
            fc, fs2 = math.cos(fa), math.sin(fa)
            bx, by = ca * (-hl), sa * (-hl)

            def fp(lx, ly):
                return (int(cx + bx + fc * lx - fs2 * ly),
                        int(cy - by - fs2 * lx - fc * ly))

            flame = [fp(0, fw), fp(fl, 0), fp(0, -fw)]
            core = [fp(0, fw * 0.4), fp(fl * 0.7, 0), fp(0, -fw * 0.4)]
            pygame.draw.polygon(self.screen, self.cfg.flame_color, flame)
            pygame.draw.polygon(self.screen, (255, 255, 100), core)

    # ----- Re-entry glow -------------------------------------------------

    def _draw_reentry_glow(self, v: Vehicle, scfg: StageConfig) -> None:
        if v.speed < 500 or v.rho < 0.0001 or v.vertical_speed > 0:
            return
        intensity = clamp(v.q / 100_000, 0.0, 1.0)
        if intensity < 0.05:
            return

        cx, cy = self.camera.world_to_screen(v.state.x, v.state.y)
        radius = max(int(15 * intensity), 3)
        alpha = int(120 * intensity)
        glow_surf = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (255, 100, 20, alpha),
                           (radius * 2, radius * 2), radius)
        pygame.draw.circle(glow_surf, (255, 200, 50, alpha // 2),
                           (radius * 2, radius * 2), radius * 2)
        self.screen.blit(glow_surf,
                         (cx - radius * 2, cy - radius * 2),
                         special_flags=pygame.BLEND_ADD)

    # ----- Vectors -------------------------------------------------------

    def _draw_vectors(self, v: Vehicle) -> None:
        cx, cy = self.camera.world_to_screen(v.state.x, v.state.y)
        if v.speed > 1.0:
            s = min(80, v.speed * 0.3)
            vd = v.velocity / v.speed
            ex, ey = int(cx + vd[0] * s), int(cy - vd[1] * s)
            pygame.draw.line(self.screen, self.cfg.velocity_vector_color,
                             (cx, cy), (ex, ey), 2)
        if v.current_thrust > 0:
            ta = v.state.theta + v.gimbal_angle
            td = direction_from_angle(ta)
            s = min(60, v.current_thrust / v.cfg.max_thrust * 60)
            ex, ey = int(cx + td[0] * s), int(cy - td[1] * s)
            pygame.draw.line(self.screen, self.cfg.thrust_vector_color,
                             (cx, cy), (ex, ey), 2)

    # ----- HUD -----------------------------------------------------------

    def _draw_hud(self, b: Vehicle, u: Vehicle, seq: object, paused: bool) -> None:
        mt = getattr(seq, 'mission_time', 0.0)
        pname = getattr(seq, 'phase_name', '—')
        sep = getattr(seq, 'separated', False)
        t_str = f"T{mt:+.2f}s" if mt < 0 else f"T+{mt:.2f}s"

        focus_v = b if self._focus == "booster" else u
        focus_label = "BOOSTER" if self._focus == "booster" else "UPPER STAGE"

        lines = [
            f"{t_str}  |  {pname}",
            f"Focus: {focus_label}  (TAB to switch)",
            "",
        ]

        if focus_v.is_active:
            lines += [
                f"Alt   {focus_v.altitude:10.1f} m  ({focus_v.altitude/1000:.1f} km)",
                f"Dwnrg {focus_v.state.x:10.1f} m",
                f"Speed {focus_v.speed:10.1f} m/s",
                f"Vy    {focus_v.vertical_speed:+10.1f} m/s",
                f"Vx    {focus_v.horizontal_speed:+10.1f} m/s",
                "",
                f"Mass  {focus_v.total_mass:10.1f} kg",
                f"Fuel  {focus_v.state.fuel:10.1f} kg ({focus_v.fuel_fraction*100:.0f}%)",
                f"Thrust{focus_v.current_thrust:10.0f} N",
                f"Thrtl {focus_v.throttle*100:9.0f} %",
                "",
                f"Theta {math.degrees(focus_v.state.theta):+9.1f} deg",
                f"Q     {focus_v.q:10.1f} Pa",
                f"Eng   {focus_v.engine_state.name}",
                f"Phase {focus_v.flight_phase.name}",
            ]
        else:
            lines.append("(inactive)")

        # Other vehicle status
        other = u if self._focus == "booster" else b
        other_label = "Upper" if self._focus == "booster" else "Booster"
        if other.is_active and sep:
            lines += [
                "",
                f"{other_label}: alt {other.altitude/1000:.1f}km  "
                f"spd {other.speed:.0f}m/s  "
                f"{'TERMINATED' if other.is_terminated else other.flight_phase.name}",
            ]

        x, y = 10, 10
        for line in lines:
            s = self.font.render(line, True, self.cfg.text_color)
            self.screen.blit(s, (x, y))
            y += 18

    def _draw_debug(self, v: Vehicle) -> None:
        lines = [
            f"g(h)  {v.g_local:.4f} m/s²",
            f"rho   {v.rho:.6f} kg/m³",
            f"MoI   {v.moi:.0f} kg·m²",
            f"Gimbal{math.degrees(v.gimbal_angle):+6.2f} deg",
            f"Zoom  {self.camera.zoom:.5f} px/m",
            f"MaxAlt{v.max_altitude/1000:.1f} km",
            f"MaxSpd{v.max_speed:.0f} m/s",
        ]
        x = self.cfg.screen_width - 230
        y = 10
        for line in lines:
            s = self.font.render(line, True, (180, 180, 100))
            self.screen.blit(s, (x, y))
            y += 18

    def _draw_controls(self) -> None:
        s = self.font.render(
            "P=Pause  R=Reset  TAB=Focus  +/-=Zoom  T=Trail  V=Vectors  "
            "F3=Debug  Z=AutoZoom  1-4=Warp  ESC=Quit",
            True, (140, 140, 140))
        self.screen.blit(s, (10, self.cfg.screen_height - 22))

    def _draw_end_status(self, b: Vehicle, u: Vehicle) -> None:
        parts = []
        if b.is_terminated:
            label = "LANDED" if b.flight_phase == FlightPhase.LANDED else "CRASHED"
            color = (60, 255, 60) if label == "LANDED" else (255, 60, 60)
            parts.append((f"Booster: {label}", color))
        if u.is_terminated:
            parts.append((f"Upper: max {u.max_altitude/1000:.0f}km", (200, 200, 255)))

        if parts:
            text = "  |  ".join(p[0] for p in parts) + "  |  R=Restart"
            color = parts[0][1]
            s = self.font_large.render(text, True, color)
            r = s.get_rect(center=(self.cfg.screen_width // 2,
                                   self.cfg.screen_height - 45))
            self.screen.blit(s, r)

    # ----- Utility -------------------------------------------------------

    def toggle_focus(self) -> str:
        self._focus = "upper" if self._focus == "booster" else "booster"
        return self._focus

    def toggle_auto_zoom(self) -> bool:
        self._auto_zoom = not self._auto_zoom
        return self._auto_zoom

    def handle_resize(self, w: int, h: int) -> None:
        self.cfg.screen_width = w
        self.cfg.screen_height = h
        self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
