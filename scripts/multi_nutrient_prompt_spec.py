"""Four single-attribute prompts; exact task wording comes from frozen v02."""
import math

def nutrition_line(config, row):
    condition=row['condition']
    if condition=='V': return ''
    if condition not in ['T',*[f'P{i}' for i in range(1,6)]]:raise ValueError(condition)
    nutrient=row['nutrient'];value=float(row['presented_value'])
    if nutrient not in config['nutrient_templates'] or not math.isfinite(value) or value<0:
        raise ValueError('A valid, nonnegative per-100-g nutrient value is required')
    return config['nutrient_templates'][nutrient].format(value=value)+'\n'

def render_task_text(config, row):
    return nutrition_line(config,row)+config['prompts'][row['prompt_id']]['text']
