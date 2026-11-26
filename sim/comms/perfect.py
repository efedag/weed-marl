"""Hiç mesaj kaybetmeyen kanal (referans koşul)."""

import numpy as np

from sim.comms.base import Channel


class PerfectChannel(Channel):
    def _loss_mask(self, messages):
        return np.zeros(len(messages), dtype=bool)
