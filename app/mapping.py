
calibration_ranges = {
    "lighting/exposure": {"low": -0.028060731291770936, "high": 0.008780921995639795},
    "contrast": {"low": -0.015044403076171876, "high": 0.004521393775939941},
    "framing/perspective": {"low": -0.009384018927812576, "high": 0.0201389268040657},
    "visual focus": {"low": -0.011019904911518098, "high": 0.0087275892496109},
    "visual complexity": {"low": -0.011279307305812836, "high": 0.012673939019441601},
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