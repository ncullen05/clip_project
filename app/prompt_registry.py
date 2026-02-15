# app/prompt_registry.py
"""
Prompt Registry for loading aesthetic evaluation prompts.

"""

from app.prompts import lighting_exposure as le
from app.prompts import contrast as c
from app.prompts import framing_perspective as fp
from app.prompts import visual_focus as vf
from app.prompts import visual_complexity as vc

def get_prompt_sets():
    """
    Retrieve all positive and negative aesthetic evaluation prompts.
    
    Returns:
        tuple: A pair of dictionaries (positive_prompts, negative_prompts).
               Each maps aesthetic dimension names to lists of prompt strings.
    """
    # Dictionary of positive prompts describing high-quality aesthetic attributes
    positive_prompts = {
        "lighting/exposure": le.p_light_expo,
        "contrast": c.p_contrast,
        "framing/perspective": fp.p_frame_persp,
        "visual focus": vf.p_v_focus,
        "visual complexity": vc.p_v_complexity,
    }

    # Dictionary of negative prompts describing low-quality aesthetic attributes
    negative_prompts = {
        "lighting/exposure": le.n_light_expo,
        "contrast": c.n_contrast,
        "framing/perspective": fp.n_frame_persp,
        "visual focus": vf.n_v_focus,
        "visual complexity": vc.n_v_complexity,
    }

    # Return both sets for use in scoring
    return positive_prompts, negative_prompts
