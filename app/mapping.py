
calibration_ranges = {
    "lighting/exposure": {"low": -0.010631025582551957, "high": 0.008518607914447762},
    "contrast": {"low": -0.01173621416091919, "high": 0.010204917192459105},
    "framing/perspective": {"low": -0.01035672277212143, "high": 0.012867729365825642},
    "visual focus": {"low": -0.020621787011623382, "high": -0.001886074244976044},
    "visual complexity": {"low": -0.00822569578886032, "high": 0.014318048208951925},
}

# Helper method ensuring the number stays within a fixed range
# This prevents scores going below 0 or above 10.
def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    elif value > hi:
        return hi
    return value

# Convert a raw CLIP delta into a user-friendly score from 0 to 10.
# CLIP deltas are relative and unbounded, so we rescale them.
def delta_to_score(delta: float, low_delta: float, high_delta: float) -> float:
    # Safety check to avoid division by zero
    if high_delta == low_delta:
        return 5.0 # Neutral score
    
    # Low delta -> score should be 0
    # High delta -> score should be 10
    # Values in between are scaled linearly
    if delta <= low_delta:
        return 0
    elif delta >= high_delta:
        return 10
    else:
        # Linear interpolation between low and high deltas
        scaled = (delta - low_delta) / (high_delta - low_delta) * 10
        return _clamp(scaled, 0, 10)     