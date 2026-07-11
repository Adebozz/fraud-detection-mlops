"""Canonical feature schema — single source of truth for training AND serving.

Deliberately import-free (no pandas/sklearn) so the slim serving image can
use it without pulling in training dependencies.
"""

FEATURE_COLUMNS = [*[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET_COLUMN = "Class"
