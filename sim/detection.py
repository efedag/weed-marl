"""Drone dedektörü için takılabilir arayüz.

WeedEnv, drone'un algılama yarıçapındaki hücrelerin gerçek etiketlerini
``detect`` metoduna verir ve dönen etiket/güven dizilerini drone'un
keşfedilen haritasına yazar. Sema Nur'un simüle dedektörü bu arayüzü
uygulayarak ``WeedEnv(detector=...)`` ile takılabilir.
"""

from typing import Protocol

import numpy as np


class Detector(Protocol):
    def detect(
        self, true_labels: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bir hücre bloğu için tahmini etiketleri ve güven değerlerini döndürür.

        true_labels: (h, w) uint8 gerçek etiketler (0 toprak, 1 ürün, 2 ot).
        rng: Ortamın dedektöre ayırdığı rastgele sayı üreteci; dedektör
            kendi rastgeleliğini yalnızca buradan almalıdır.
        Dönüş: (labels, confidence); ikisi de true_labels ile aynı şekilde,
            labels uint8, confidence [0, 1] aralığında float32.
        """
        ...


class PerfectDetector:
    """Gerçek etiketi güven 1.0 ile döndüren varsayılan dedektör."""

    def detect(self, true_labels, rng):
        return true_labels.copy(), np.ones(true_labels.shape, dtype=np.float32)
