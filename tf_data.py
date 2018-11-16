import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class TfData(ABC):
    @abstractmethod
    def random_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @abstractmethod
    def all_rows(self) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @abstractmethod
    def num_rows(self) -> int:
        ...
