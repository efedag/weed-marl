"""Drone -> robot mesaj kanalları."""

from sim.comms.base import Channel, Message
from sim.comms.perfect import PerfectChannel

CHANNELS = {
    "perfect": PerfectChannel,
}


def make_channel(name: str = "perfect", params: dict | None = None) -> Channel:
    """Config'deki ``channel`` adına ve parametrelerine göre kanal kurar."""
    try:
        cls = CHANNELS[name]
    except KeyError:
        raise ValueError(
            f"Bilinmeyen kanal: {name!r}. Seçenekler: {sorted(CHANNELS)}"
        ) from None
    return cls(**(params or {}))


__all__ = ["Channel", "Message", "PerfectChannel", "CHANNELS", "make_channel"]
