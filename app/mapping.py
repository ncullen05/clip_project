
calibration_ranges = {
    "lighting/exposure": {"low": -0.01332082524895668, "high": 0.00823259949684143},
    "contrast": {"low": -0.01365853101015091, "high": 0.01057754531502721},
    "framing/perspective": {"low": -0.006390059739351272, "high": 0.011978773772716519},
    "visual focus": {"low": -0.004717288166284561, "high": 0.009776509553194045},
    "visual complexity": {"low": -0.009907539188861846, "high": 0.010813049226999275},
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