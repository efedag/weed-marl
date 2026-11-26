"""Zamansal olarak ilişkili mesaj kaybı (Gilbert-Elliott kanal)."""

import numpy as np

from sim.comms.base import Channel

GOOD, BAD = 0, 1


class GilbertElliottChannel(Channel):
    """İki durumlu (Good/Bad) Markov kanal.

    Her (gönderen, alıcı) bağlantısının kendi Markov durumu vardır. Bir mesaj
    bağlantının o anki durumuna göre ``e_g`` (Good) ya da ``e_b`` (Bad)
    olasılıkla kaybolur; ardından durum ``p_gb`` (Good→Bad) ya da ``p_bg``
    (Bad→Good) olasılıkla değişir. Bağlantının ilk durumu durağan dağılımdan
    çekilir: π_B = p_gb / (p_gb + p_bg).

    Ortalama kayıp = π_G·e_g + π_B·e_b; Bad durumunda ortalama kalış süresi
    (patlama uzunluğu) = 1 / p_bg mesaj.

    Varsayılan e_g, e_b ve patlama uzunluğu Nayak et al. (2020)'ye göre
    ayarlanacak; şimdilik config'den açık parametre olarak verilir.
    Deney değerleri: ortalama kayıp 0.25 ve 0.5.
    """

    def __init__(self, p_gb: float, p_bg: float, e_g: float = 0.0, e_b: float = 1.0):
        super().__init__()
        for name, value in (("p_gb", p_gb), ("p_bg", p_bg), ("e_g", e_g), ("e_b", e_b)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} [0, 1] aralığında olmalı")
        if p_gb + p_bg == 0:
            raise ValueError("p_gb ve p_bg aynı anda 0 olamaz")
        self.p_gb = float(p_gb)
        self.p_bg = float(p_bg)
        self.e_g = float(e_g)
        self.e_b = float(e_b)
        self._loss_prob = np.array([self.e_g, self.e_b])
        self._leave_prob = np.array([self.p_gb, self.p_bg])

    @classmethod
    def from_mean_loss(
        cls,
        mean_loss: float,
        mean_burst_length: float = 5.0,
        e_g: float = 0.0,
        e_b: float = 1.0,
    ) -> "GilbertElliottChannel":
        """Ortalama kayıp ve ortalama patlama uzunluğundan kanal kurar.

        p_bg = 1 / mean_burst_length; π_B = (mean_loss − e_g) / (e_b − e_g);
        p_gb = π_B · p_bg / (1 − π_B).
        """
        if not e_g < e_b:
            raise ValueError("e_g, e_b'den küçük olmalı")
        if not e_g <= mean_loss <= e_b:
            raise ValueError("mean_loss [e_g, e_b] aralığında olmalı")
        if mean_burst_length < 1:
            raise ValueError("mean_burst_length en az 1 olmalı")
        p_bg = 1.0 / mean_burst_length
        pi_b = (mean_loss - e_g) / (e_b - e_g)
        if pi_b >= 1.0:
            return cls(p_gb=1.0, p_bg=0.0, e_g=e_g, e_b=e_b)
        p_gb = pi_b * p_bg / (1.0 - pi_b)
        if p_gb > 1.0:
            raise ValueError(
                f"mean_loss={mean_loss} ve mean_burst_length={mean_burst_length} "
                f"ile p_gb={p_gb:.3f} > 1 çıkıyor; patlama uzunluğunu artırın"
            )
        return cls(p_gb=p_gb, p_bg=p_bg, e_g=e_g, e_b=e_b)

    @property
    def stationary_bad(self) -> float:
        return self.p_gb / (self.p_gb + self.p_bg)

    @property
    def mean_loss(self) -> float:
        pi_b = self.stationary_bad
        return (1.0 - pi_b) * self.e_g + pi_b * self.e_b

    @property
    def mean_burst_length(self) -> float:
        return float("inf") if self.p_bg == 0 else 1.0 / self.p_bg

    def link_state(self, sender: str, receiver: str) -> int | None:
        return self._links.get((sender, receiver))

    def _loss_mask(self, messages):
        rng = self._rng
        # Mesaj başına iki sayı: kayıp kararı ve durum geçişi
        draws = rng.random((len(messages), 2))
        lost = np.empty(len(messages), dtype=bool)
        for i, message in enumerate(messages):
            state = self._links.get(message.link)
            if state is None:
                state = BAD if rng.random() < self.stationary_bad else GOOD
            lost[i] = draws[i, 0] < self._loss_prob[state]
            if draws[i, 1] < self._leave_prob[state]:
                state = 1 - state
            self._links[message.link] = state
        return lost
