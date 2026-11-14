"""
Preprocessing module for the HCRL Car-Hacking Dataset.

Parses raw variable-length CAN bus CSV logs (which have no header row),
safely pads payloads where DLC < 8 with '00' bytes, and engineers the
three core features:
  1. Normalized CAN_ID (converted from hexadecimal to integer, scaled to [0, 1])
  2. Normalized 8-byte payload DATA[0..7] (hex bytes 0-255 scaled to [0, 1])
  3. Message Frequency / Time Delta (delta_t: time difference between
     consecutive frames sharing the same CAN_ID)
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.preprocessing import MinMaxScaler


CLASS_NAMES = ["Normal", "DoS", "Fuzzy", "Spoofing"]
LABEL_MAP = {
    "Normal": 0,
    "DoS": 1,
    "Fuzzy": 2,
    "Spoofing": 3,
}

DATASET_FILES = {
    "DoS_dataset.csv": "DoS",
    "Fuzzy_dataset.csv": "Fuzzy",
    "RPM_dataset.csv": "Spoofing",
    "gear_dataset.csv": "Spoofing",
}


def parse_can_line(line: str) -> Optional[Tuple[float, int, List[int], str]]:
    """
    Parse a single CSV line from the HCRL Car-Hacking Dataset.

    Format per line:
        Timestamp, CAN_ID (hex), DLC (int), DATA[0..DLC-1] (hex), Flag ('R' or 'T')

    Packets with DLC < 8 have fewer data columns before the Flag column.
    This parser always extracts the last column as `Flag` and right-pads
    missing payload bytes with '00' (0) up to 8 bytes.
    """
    line = line.strip()
    if not line:
        return None

    parts = [p.strip() for p in line.split(",") if p.strip() != ""]
    if len(parts) < 4:
        return None

    try:
        timestamp = float(parts[0])
        can_id = int(parts[1], 16)
        dlc = int(parts[2])
        flag = parts[-1].upper()
        if flag not in ("R", "T"):
            return None

        # Extract payload bytes between index 3 and the last element (Flag)
        raw_data_hex = parts[3:-1]
        # Take up to min(dlc, 8) bytes from the row
        data_bytes = [int(b, 16) for b in raw_data_hex[:min(dlc, 8)]]
        # Right-pad with '00' (0) if DLC < 8 or fewer bytes present
        while len(data_bytes) < 8:
            data_bytes.append(0)

        return timestamp, can_id, data_bytes, flag
    except (ValueError, IndexError):
        return None


def parse_dataset_file(
    filepath: str,
    attack_type: str,
    sample_size: Optional[int] = 50000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read and parse a single HCRL Car-Hacking CSV file and compute raw features:
        [can_id, data_0, ..., data_7, delta_t]

    Args:
        filepath: Path to the CSV file.
        attack_type: Attack label assigned when Flag == 'T' ('DoS', 'Fuzzy', 'Spoofing').
        sample_size: Max number of valid rows to load per file (None for full dataset).

    Returns:
        features: np.ndarray of shape (N, 10) -> [can_id, d0..d7, delta_t]
        labels: np.ndarray of shape (N,) -> integer class labels (0..3)
    """
    features = []
    labels = []
    last_seen_ts: Dict[int, float] = {}

    attack_label = LABEL_MAP[attack_type]
    normal_label = LABEL_MAP["Normal"]

    count = 0
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_can_line(line)
            if parsed is None:
                continue

            timestamp, can_id, data_bytes, flag = parsed

            # Feature 3: Message Frequency / Time Delta (delta_t) between consecutive
            # frames sharing the same CAN_ID
            if can_id in last_seen_ts:
                delta_t = max(0.0, timestamp - last_seen_ts[can_id])
            else:
                delta_t = 0.0
            last_seen_ts[can_id] = timestamp

            row_features = [float(can_id)] + [float(b) for b in data_bytes] + [float(delta_t)]
            features.append(row_features)
            labels.append(normal_label if flag == "R" else attack_label)

            count += 1
            if sample_size is not None and count >= sample_size:
                break

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)


def load_and_preprocess_data(
    data_dir: str = "data",
    sample_size: Optional[int] = 50000,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Load all four HCRL Car-Hacking Dataset files, engineer the 10-dimensional
    feature vector (CAN_ID + 8 payload bytes + delta_t), and normalize all
    features into [0, 1].

    Args:
        data_dir: Directory containing DoS_dataset.csv, Fuzzy_dataset.csv,
                  RPM_dataset.csv, and gear_dataset.csv.
        sample_size: Number of rows to read per CSV file (default 50000; set to
                     None to train on the complete dataset).

    Returns:
        X_normalized: np.ndarray of shape (N, 10) normalized to [0, 1]
        y: np.ndarray of shape (N,) with integer labels 0..3
        scaler: Fitted MinMaxScaler instance
    """
    all_features = []
    all_labels = []

    for filename, attack_type in DATASET_FILES.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            # Also check one subdirectory level deep in case extracted into a subfolder
            found = False
            for root, _, files in os.walk(data_dir):
                if filename in files:
                    filepath = os.path.join(root, filename)
                    found = True
                    break
            if not found:
                raise FileNotFoundError(
                    f"Required dataset file '{filename}' not found in '{data_dir}'."
                )

        print(
            f"[Preprocess] Loading {filename} (attack class: {attack_type}, "
            f"sample_size={sample_size})..."
        )
        X_part, y_part = parse_dataset_file(filepath, attack_type, sample_size=sample_size)
        print(
            f"  -> Loaded {len(X_part):,} frames "
            f"(Normal: {np.sum(y_part == 0):,}, {attack_type}: {np.sum(y_part != 0):,})"
        )
        all_features.append(X_part)
        all_labels.append(y_part)

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    # Normalize features:
    # 1. CAN_ID normalized via MinMaxScaler (or max 0x7FF)
    # 2. DATA[0..7] converted from hex (0-255) and normalized to [0, 1]
    # 3. delta_t normalized via MinMaxScaler
    scaler = MinMaxScaler()
    # Explicitly scale payload bytes by 255.0 first, then fit MinMaxScaler across all columns
    # so that CAN_ID, DATA[0..7], and delta_t are cleanly normalized to [0, 1].
    X[:, 1:9] = X[:, 1:9] / 255.0
    X_normalized = scaler.fit_transform(X).astype(np.float32)

    print(f"[Preprocess] Total combined samples: {len(X_normalized):,} | Features shape: {X_normalized.shape}")
    return X_normalized, y, scaler


if __name__ == "__main__":
    X_data, y_data, _ = load_and_preprocess_data(data_dir="data", sample_size=50000)
    unique, counts = np.unique(y_data, return_counts=True)
    for cls_idx, cnt in zip(unique, counts):
        print(f"  Class {cls_idx} ({CLASS_NAMES[cls_idx]}): {cnt:,} samples")
