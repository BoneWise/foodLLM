"""Render frozen task wording; no model, human outcomes, or tokenizer dependency."""
import math


def render_task_text(config, condition, kcal=None, prompt_id=None):
    if condition != 'V' and condition != 'T' and condition not in [f'P{i}' for i in range(1, 6)]:
        raise ValueError(f'Unknown condition: {condition}')
    if condition == 'V':
        nutrient = ''
    else:
        if kcal is None or not math.isfinite(float(kcal)) or float(kcal) < 0:
            raise ValueError('T/P require finite nonnegative kcal per 100 g')
        nutrient = config['nutrition_line_template'].format(kcal=float(kcal))
    if 'prompts' in config:
        if prompt_id not in config['prompts']:
            raise ValueError('An explicit frozen prompt_id is required')
        return nutrient + config['prompts'][prompt_id]['text']
    # Reproduction of historical v01 only; never inject its context into v02.
    return config['prefix'] + nutrient + config['suffix']
