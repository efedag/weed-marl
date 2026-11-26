"""İletişim kanalı için ortak arayüz."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Message:
    """Bir (gönderen, alıcı) bağlantısı üzerinden giden tek mesaj."""

    sender: str
    receiver: str
    step: int
    payload: Any = None

    @property
    def link(self) -> tuple[str, str]:
        return (self.sender, self.receiver)


class Channel(ABC):
    """Mesajları iletir ya da kaybeder.

    Her (gönderen, alıcı) bağlantısının kendi durumu olabilir; alt sınıflar
    bunu ``_links`` sözlüğünde tutar. Bir ``transmit`` çağrısındaki mesajlar
    listedeki sırayla işlenir, aynı bağlantıdaki mesajlar art arda gönderilmiş
    sayılır.

    Sayaçlar: ``sent`` gönderilen, ``lost`` kaybolan mesaj sayısı (son
    ``reset``'ten beri).
    """

    def __init__(self):
        self._rng = np.random.default_rng()
        self._links: dict[tuple[str, str], Any] = {}
        self.sent = 0
        self.lost = 0

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._links = {}
        self.sent = 0
        self.lost = 0

    def transmit(self, messages: list[Message]) -> list[Message]:
        """Mesajları gönderir ve ulaşanları aynı sırayla döndürür."""
        if not messages:
            return []
        lost = np.asarray(self._loss_mask(messages), dtype=bool)
        self.sent += len(messages)
        self.lost += int(lost.sum())
        return [m for m, is_lost in zip(messages, lost) if not is_lost]

    @property
    def loss_rate(self) -> float:
        return self.lost / self.sent if self.sent else 0.0

    @abstractmethod
    def _loss_mask(self, messages: list[Message]) -> np.ndarray:
        """Her mesaj için kaybolup kaybolmadığını (True = kayıp) döndürür."""
