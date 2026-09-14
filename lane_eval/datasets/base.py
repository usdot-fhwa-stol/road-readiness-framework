from __future__ import annotations
from typing import Iterator

from ..schema.sample import LaneSample


class LaneDatasetAdapter:
    """Abstract base for all dataset adapters.

    Subclasses implement __len__ and __getitem__ returning LaneSample.
    The name attribute is used by the registry and result JSON.
    """

    name: str = ""
    supported_tasks = {"lane"}

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> LaneSample:
        raise NotImplementedError

    def iter_samples(self) -> Iterator[LaneSample]:
        for i in range(len(self)):
            yield self[i]
