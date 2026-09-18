"""Sequential A/B technical gates and extraction, independent of scientific effects."""
import json,sys,subprocess,time
from pathlib import Path

def main():
    root=Path('/path/to/foodLLM');scripts=root/'scripts/controls_v09_v01';inp=root/'inputs/controls_v09_v01';outputs=root/'outputs'
    status=outputs/'controls_v09_sequence_v01.json';assert not status.exists()
    token=outputs/'controls_v09_token_audit_v01/audit.json';assert json.loads(token.read_text())['status']=='passed_before_N_features'
    baseline=json.loads((outputs/'multi_nutrient_A_features_v01/audit.json').read_text())
    def state(stage):status.write_text(json.dumps(dict(stage=stage,updated_unix=time.time(),scientific_effects_used_for_gate=False),indent=2))
    def run(stage,args):
        state(stage)
        with (root/'logs'/(stage+'.log')).open('x') as f:subprocess.run([sys.executable,*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        for pr in ['A','B']:
            common=['--model-dir',root/'models/Qwen3.5-9B-c202236','--prompt-config',inp/'control_prompt_v01.json']
            pilot=outputs/f'controls_{pr}_pilot_v01';features=outputs/f'controls_{pr}_features_v01';matrix=outputs/f'controls_{pr}_matrix_v01'
            run(f'controls_{pr}_pilot_v01',[scripts/'pilot_control_interface.py',*common,'--input-manifest',inp/f'{pr}_pilot.json','--output',pilot])
            report=json.loads((pilot/'audit.json').read_text());assert report['status']=='passed'
            for key in ['model_files','versions','image_processor','dtype','attention_implementation','seed','layer_types']:
                assert report[key]==baseline[key],('Historical environment mismatch',key)
            run(f'controls_{pr}_features_v01',[scripts/'pilot_control_interface.py',*common,'--mode','extract','--reference-dir',pilot,'--input-manifest',inp/f'{pr}_full.json','--output',features])
            run(f'controls_{pr}_matrix_v01',[scripts/'verify_build_controls.py','--features',features,'--manifest',inp/f'{pr}_full.json','--config',inp/'control_prompt_v01.json','--output',matrix])
            assert json.loads((matrix/'audit.json').read_text())['status']=='complete_verified'
        state('A_B_features_and_matrices_verified')
    except Exception as e:state('failed: '+repr(e));raise

if __name__=='__main__':main()
