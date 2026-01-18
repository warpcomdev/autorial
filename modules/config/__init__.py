"""Shared configuration helpers for modules."""

from __future__ import annotations

import configparser


def get_float(section: configparser.SectionProxy | dict, key: str, default: float) -> float:
    """Read a float from the config file for module settings."""
    try:
        return float(section.get(key, default))
    except ValueError:
        return default


def get_int(section: configparser.SectionProxy | dict, key: str, default: int) -> int:
    """Read an int from the config file for module settings."""
    try:
        return int(section.get(key, default))
    except ValueError:
        return default


__all__ = ["get_float", "get_int"]
