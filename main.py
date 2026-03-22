#!/usr/bin/env python3
"""
Rocket Simulator – advanced 2D rocket flight simulation.

Launch: python main.py

Dependencies: pygame, numpy, matplotlib (optional, for post-flight plots).
"""

from __future__ import annotations

import sys

import pygame

from config import AppConfig
from simulation import Simulation


def main() -> None:
    pygame.init()

    cfg = AppConfig()
    sim = Simulation(cfg)

    try:
        sim.run()
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()

    sys.exit(0)


if __name__ == "__main__":
    main()
