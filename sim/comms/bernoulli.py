"""Bağımsız mesaj kaybı (Bernoulli kanal)."""

from sim.comms.base import Channel


class BernoulliChannel(Channel):
    """Her mesaj diğerlerinden bağımsız olarak ``p`` olasılıkla kaybolur.

    Deney değerleri: p ∈ {0, 0.1, 0.25, 0.5}.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError("p [0, 1] aralığında olmalı")
        self.p = float(p)

    @property
    def mean_loss(self) -> float:
        return self.p

    def _loss_mask(self, messages):
        return self._rng.random(len(messages)) < self.p
