"""Approved v09 unrelated text; frozen A/B task text is unchanged."""
def nutrition_line(config,row):
    if row['condition']=='V':return ''
    if row['condition'] not in ['N1','N2','N3'] or row['nutrient']!='unrelated_text':raise ValueError('Unexpected control identity')
    line=row['inserted_line']
    if not line.startswith('Reference words: ') or not line.endswith('.') or '\n' in line:raise ValueError('Malformed control line')
    return line+'\n'

def render_task_text(config,row):
    return nutrition_line(config,row)+config['prompts'][row['prompt_id']]['text']
