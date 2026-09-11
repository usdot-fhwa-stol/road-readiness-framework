from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class LaneTarget:
    lanes: Optional[List[np.ndarray]] = None  # each [N,2] (x,y) image coords
    mask: Optional[np.ndarray] = None          # [H,W] uint8 binary {0,1}
    mask_path: Optional[str] = None
    meta: dict = field(default_factory=dict)


@dataclass
class LanePrediction:
    mask: Optional[np.ndarray] = None          # [H,W] uint8 binary {0,1}
    prob: Optional[np.ndarray] = None          # [H,W] float, lane probability
    lanes: Optional[List[np.ndarray]] = None   # predicted polylines if decoded
    scores: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)
