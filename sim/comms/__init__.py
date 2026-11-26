"""Drone -> robot mesaj kanalları."""

from sim.comms.base import Channel, Message
from sim.comms.bernoulli import BernoulliChannel
from sim.comms.gilbert_elliott import GilbertElliottChannel
from sim.comms.perfect import PerfectChannel

CHANNELS = {
    "perfect": PerfectChannel,
    "bernoulli": BernoulliChannel,
    "gilbert_elliott": GilbertElliottChannel,
}


def make_channel(name: str = "perfect", params: dict | None = None) -> Channel:
    """Config'deki ``channel`` adına ve parametrelerine göre kanal kurar.

    Örnekler:
        make_channel("bernoulli", {"p": 0.25})
        make_channel("gilbert_elliott", {"mean_loss": 0.25, "mean_burst_length": 5})
        make_channel("gilbert_elliott", {"p_gb": 0.05, "p_bg": 0.2, "e_g": 0.0, "e_b": 1.0})
    """
    params = dict(params or {})
    if name not in CHANNELS:
        raise ValueError(f"Bilinmeyen kanal: {name!r}. Seçenekler: {sorted(CHANNELS)}")
    if name == "gilbert_elliott" and "mean_loss" in params:
        return GilbertElliottChannel.from_mean_loss(**params)
    return CHANNELS[name](**params)


__all__ = [
    "Channel",
    "Message",
    "PerfectChannel",
    "BernoulliChannel",
    "GilbertElliottChannel",
    "CHANNELS",
    "make_channel",
]
