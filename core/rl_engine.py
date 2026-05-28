"""Q-learning progression engine for adaptive difficulty selection.

State space: (tier_idx ∈ {0,1,2})  ×  (accuracy_bucket ∈ {0,1,2})
  tier_idx:         0=Beginner, 1=Intermediate, 2=Advanced
  accuracy_bucket:  0=poor(<0.50), 1=medium(0.50–0.80), 2=good(>0.80)

Actions: 0=go easier  1=stay same  2=go harder
  → delta in tier: -1, 0, +1

Reward: delta in phoneme accuracy between consecutive phrase attempts.
  Positive reward → learning → Q-table pushes toward harder tiers.
  Negative reward → struggling → Q-table pushes toward easier tiers.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_N_TIERS = 3
_N_ACC_BUCKETS = 3
_N_ACTIONS = 3

_ALPHA = 0.30    # learning rate
_GAMMA = 0.90    # discount factor
_EPSILON = 0.15  # exploration probability (ε-greedy)

_QTABLE_PATH = Path(__file__).parent.parent / "data" / "qtable.json"


def _acc_bucket(accuracy: float) -> int:
    if accuracy < 0.50:
        return 0
    if accuracy < 0.80:
        return 1
    return 2


class RLEngine:
    """Tabular Q-learning agent for adaptive difficulty tier selection."""

    def __init__(self, qtable_path: Optional[Path] = None):
        self._path = Path(qtable_path or _QTABLE_PATH)
        self._q: np.ndarray = self._load_qtable()
        self._last_state: Optional[Tuple[int, int]] = None
        self._last_action: Optional[int] = None
        self._prev_accuracy: float = 0.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_qtable(self) -> np.ndarray:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    data = json.load(f)
                arr = np.array(data, dtype=float)
                if arr.shape == (_N_TIERS, _N_ACC_BUCKETS, _N_ACTIONS):
                    return arr
            except Exception:
                pass
        return np.zeros((_N_TIERS, _N_ACC_BUCKETS, _N_ACTIONS))

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._q.tolist(), f, indent=2)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def _state(self, tier: int, accuracy: float) -> Tuple[int, int]:
        tier_idx = max(0, min(_N_TIERS - 1, tier - 1))
        return (tier_idx, _acc_bucket(accuracy))

    def select_tier(self, current_tier: int, recent_accuracy: float) -> int:
        """Return next difficulty tier (1–3) using ε-greedy policy."""
        s = self._state(current_tier, recent_accuracy)
        self._last_state = s
        self._prev_accuracy = recent_accuracy

        if random.random() < _EPSILON:
            action = random.randint(0, _N_ACTIONS - 1)
        else:
            action = int(np.argmax(self._q[s[0], s[1]]))

        self._last_action = action
        delta = action - 1  # 0→-1, 1→0, 2→+1
        return max(1, min(_N_TIERS, current_tier + delta))

    def update(self, new_tier: int, new_accuracy: float) -> float:
        """Update Q-table and return the reward signal.

        Reward = delta in phoneme accuracy (positive = improved).
        """
        if self._last_state is None or self._last_action is None:
            return 0.0

        reward = new_accuracy - self._prev_accuracy

        s = self._last_state
        a = self._last_action
        s_prime = self._state(new_tier, new_accuracy)
        best_next = float(np.max(self._q[s_prime[0], s_prime[1]]))
        td_error = reward + _GAMMA * best_next - self._q[s[0], s[1], a]
        self._q[s[0], s[1], a] += _ALPHA * td_error

        self.save()
        return reward

    @property
    def q_values(self) -> np.ndarray:
        return self._q.copy()
