from __future__ import annotations
from dataclasses import dataclass, field

from .lane import LaneTarget


@dataclass
class LaneSample:
    image_id: str
    image_path: str
    width: int
    height: int
    target: LaneTarget
    meta: dict = field(default_factory=dict)
