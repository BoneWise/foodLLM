"""Wait for A extraction, verify it, then run and verify B regardless of effects."""
import json,subprocess,sys,time,argparse
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();root=Path(a.root)
    scripts=root/'scripts/multi_nutrient_v01';inputs=root/'inputs/multi_nutrient_v01';outputs=root/'outputs'
    status=outputs/'multi_nutrient_sequence_v01.json';assert not status.exists()
    def state(stage):status.write_text(json.dumps({'stage':stage,'updated_unix':time.time(),'scientific_effects_used_for_gate':False},indent=2))
    def run(stage,cmd):
        state(stage)
        with (root/'logs'/f'{stage}.log').open('x') as f:subprocess.run([sys.executable,*map(str,cmd)],stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        state('waiting_for_A_features');start=time.time();report=outputs/'multi_nutrient_A_features_v01/audit.json'
        while not report.exists():
            if time.time()-start>3600:raise TimeoutError('A audit not available within an hour')
            time.sleep(5)
        assert json.loads(report.read_text())['status']=='complete'
        for prompt in ['A','B']:
            if prompt=='B':
                common=['--model-dir',root/'models/Qwen3.5-9B-c202236','--prompt-config',inputs/'multi_nutrient_prompt_v01.json']
                run('multi_nutrient_B_pilot_v01',[scripts/'pilot_multi_nutrient_interface.py',*common,'--input-manifest',inputs/'B_pilot.json','--output',outputs/'multi_nutrient_B_pilot_v01'])
                assert json.loads((outputs/'multi_nutrient_B_pilot_v01/audit.json').read_text())['status']=='passed'
                run('multi_nutrient_B_features_v01',[scripts/'pilot_multi_nutrient_interface.py',*common,'--mode','extract','--reference-dir',outputs/'multi_nutrient_B_pilot_v01','--input-manifest',inputs/'B_full.json','--output',outputs/'multi_nutrient_B_features_v01'])
            run(f'multi_nutrient_{prompt}_matrix_v01',[scripts/'verify_build_multi_nutrients.py','--features',outputs/f'multi_nutrient_{prompt}_features_v01','--manifest',inputs/f'{prompt}_full.json','--config',inputs/'multi_nutrient_prompt_v01.json','--output',outputs/f'multi_nutrient_{prompt}_matrix_v01'])
            assert json.loads((outputs/f'multi_nutrient_{prompt}_matrix_v01/audit.json').read_text())['status']=='complete_verified'
        state('A_B_features_and_matrices_verified')
    except Exception as exc:
        state('failed: '+repr(exc));raise

if __name__=='__main__':main()
