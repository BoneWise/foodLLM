"""Attribute-aware Qwen interface and extraction; adapted from frozen kcal runner."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from control_prompt_spec import render_task_text, nutrition_line


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for name in ('model-dir', 'input-manifest', 'prompt-config', 'output'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--mode', choices=['pilot', 'extract'], default='pilot')
    ap.add_argument('--reference-dir', help='Passed same-prompt pilot required for full extraction')
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(Path(args.prompt_config).read_text())
    rows = json.loads(Path(args.input_manifest).read_text())['inputs']
    reference = None
    if args.mode == 'extract':
        if not args.reference_dir:
            raise ValueError('Extraction requires a passed pilot reference')
        reference = json.loads((Path(args.reference_dir)/'audit.json').read_text())
        assert reference['status'] == 'passed'
        assert reference['prompt_config_sha256'] == sha(args.prompt_config)
        assert reference['prompt_renderer_sha256'] == sha(Path(__file__).with_name('control_prompt_spec.py'))
        assert reference['model_revision'] == cfg['model_revision']
    keys = [(r.get('prompt_id'), r['nutrient'], r['image_id'], r['condition']) for r in rows]
    assert len(set(keys)) == len(rows), 'Duplicate input identities'
    torch.manual_seed(20260916)
    np.random.seed(20260916)
    torch.backends.cuda.matmul.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True,
        min_pixels=cfg['image_min_pixels'], max_pixels=cfg['image_max_pixels'])
    start = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir,
        local_files_only=True, dtype=torch.bfloat16, device_map='cuda:0',
        attn_implementation='sdpa').eval()
    print(json.dumps({'event': 'loaded', 'seconds': time.time()-start}), flush=True)
    text_models = [(n, m) for n, m in model.named_modules()
                   if m.__class__.__name__ == 'Qwen3_5TextModel']
    assert len(text_models) == 1
    text_name, text_model = text_models[0]
    assert len(text_model.layers) == 32
    captured = {}
    capture_active = [True]

    def hook(name):
        def save(module, inputs, output):
            if not capture_active[0]:
                return
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor[:, -1:, :].detach().clone()
            if name == 'final_norm':
                # Preserve the real tensor shape for a bitwise RMSNorm QA replay.
                # This tensor is temporary and is never a scientific feature cache.
                captured['_norm_input'] = inputs[0].detach().clone()
        return save

    handles = [m.register_forward_hook(hook(f'block_{i+1:02d}'))
               for i, m in enumerate(text_model.layers)]
    handles.append(text_model.norm.register_forward_hook(hook('final_norm')))
    records, repeat_records, reference_checks = [], [], []
    record_stream = (out/'records.jsonl').open('w')
    tokenizer_audits = {}
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        passes = (rows, list(reversed(rows))) if args.mode == 'pilot' else (rows,)
        for pass_id, ordered_rows in enumerate(passes):
            for row in ordered_rows:
                item_start = time.time()
                prompt_id = row.get('prompt_id')
                key = prompt_id + '_' + row['nutrient'] + '_' + row['image_id'] + '_' + row['condition']
                prompt = render_task_text(cfg, row)
                nutrition_ids = []
                if row['condition'] != 'V':
                    nutrition_text = nutrition_line(cfg, row).rstrip('\n')
                    nutrition_ids = processor.tokenizer.encode(nutrition_text, add_special_tokens=False)
                    assert processor.tokenizer.decode(nutrition_ids, clean_up_tokenization_spaces=False) == nutrition_text
                messages = [{'role': 'user', 'content': [
                    {'type': 'image'}, {'type': 'text', 'text': prompt}]}]
                rendered = processor.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                # Qwen's non-thinking template inserts an EMPTY, already closed think block.
                assert rendered.rsplit('<|im_start|>assistant', 1)[1].strip() == '<think>\n\n</think>'
                prefix_ids = processor.tokenizer.encode(rendered, add_special_tokens=False)
                candidates, variants = [], {}
                for c in cfg['candidate_continuations']:
                    extended = processor.tokenizer.encode(rendered+c, add_special_tokens=False)
                    assert extended[:len(prefix_ids)] == prefix_ids
                    suffix = extended[len(prefix_ids):]
                    assert len(suffix) == 1, 'Use sequence scoring for multi-token continuations'
                    candidates.append(suffix[0])
                    for variant in (c, ' '+c, '\n'+c):
                        ids = processor.tokenizer.encode(rendered+variant, add_special_tokens=False)
                        common = next((i for i, (a, b) in enumerate(zip(prefix_ids, ids))
                                       if a != b), min(len(prefix_ids), len(ids)))
                        variants[repr(variant)] = {
                            'preserves_prefix': common == len(prefix_ids),
                            'changed_prefix_tokens': len(prefix_ids)-common,
                            'suffix_token_ids': ids[common:]}
                tokenizer_audits[key] = variants
                image_path = Path(row['image_path'])
                assert sha(image_path) == row['image_sha256']
                with Image.open(image_path) as im:
                    inputs = processor(text=[rendered], images=[im.convert('RGB')], return_tensors='pt')
                assert inputs['attention_mask'].bool().all(), 'Pilot has no padding'
                assert inputs['input_ids'][0, -1].item() == prefix_ids[-1]
                input_id_sha = hashlib.sha256(inputs['input_ids'].numpy().tobytes()).hexdigest()
                token_count = inputs['input_ids'].shape[1]
                grid = inputs['image_grid_thw'].tolist()
                inputs = inputs.to('cuda:0')
                captured.clear()
                result = model(**inputs, use_cache=False, logits_to_keep=1)
                logits = result.logits[0, -1, :].float()
                assert len(captured) == 34
                hidden = torch.cat([captured[f'block_{i+1:02d}'] for i in range(32)], 1)[0].float().cpu().numpy()
                norm = captured['final_norm'][0, 0].float().cpu().numpy()
                assert hidden.shape == (32, 4096) and norm.shape == (4096,)
                assert np.isfinite(hidden).all() and np.isfinite(norm).all()
                saved_norm = captured['final_norm']
                assert torch.equal(captured['_norm_input'][:, -1:, :], captured['block_32'])
                capture_active[0] = False
                try:
                    norm_error = (text_model.norm(captured['_norm_input'])[:, -1:, :] - saved_norm).abs().max().item()
                    single_token_norm_error = (text_model.norm(captured['block_32']) - saved_norm).abs().max().item()
                    head_error = (model.lm_head(saved_norm)[0, 0].float() - logits).abs().max().item()
                finally:
                    capture_active[0] = True
                if norm_error != 0 or head_error != 0:
                    print(json.dumps({'event':'interface_error','key':key,'norm_error':norm_error,
                        'head_error':head_error,'token_count':token_count,'grid':grid}), flush=True)
                assert norm_error == 0 and head_error == 0
                logprobs = logits.log_softmax(-1)[candidates].cpu().numpy()
                probs = logits[candidates].softmax(-1).cpu().numpy()
                assert np.isfinite(logprobs).all() and abs(float(probs.sum())-1) < 1e-6
                cache = out / (key + '.npz')
                if pass_id == 0:
                    np.savez_compressed(cache, hidden=hidden, final_norm=norm,
                        candidate_logprobs=logprobs, rating_probs=probs,
                        input_ids=inputs['input_ids'].cpu().numpy())
                    (out / (key + '_rendered.txt')).write_text(rendered)
                    record = dict(row)
                    record.update(prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                        input_ids_sha256=input_id_sha, candidate_token_ids=candidates,
                        model_revision=cfg['model_revision'], tokenizer_revision=cfg['model_revision'],
                        enable_thinking=False, token_count=token_count, readout_index=token_count-1,
                        nutrition_token_ids=nutrition_ids,
                        readout_token_id=prefix_ids[-1], image_grid_thw=grid,
                        candidate_logprobs=logprobs.tolist(), rating_probs=probs.tolist(),
                        valid_answer_mass=float(np.exp(logprobs).sum()),
                        expected_rating=float(probs @ np.arange(1, 5)), hard_rating=int(probs.argmax()+1),
                        global_argmax_token=processor.tokenizer.decode([logits.argmax().item()]),
                        global_argmax_is_rating=logits.argmax().item() in candidates,
                        cache_path=cache.name, cache_sha256=sha(cache), norm_error=norm_error,
                        norm_single_token_replay_difference=single_token_norm_error,
                        lm_head_error=head_error, seconds=time.time()-item_start)
                    records.append(record)
                    record_stream.write(json.dumps(record)+'\n')
                    record_stream.flush()
                    reference_cache = Path(args.reference_dir)/(key+'.npz') if reference else None
                    if reference_cache and reference_cache.exists():
                        with np.load(reference_cache) as original:
                            assert np.array_equal(hidden, original['hidden'])
                            assert np.array_equal(norm, original['final_norm'])
                            assert np.array_equal(logprobs, original['candidate_logprobs'])
                            assert np.array_equal(probs, original['rating_probs'])
                            assert np.array_equal(inputs['input_ids'].cpu().numpy(), original['input_ids'])
                        reference_checks.append(key)
                else:
                    with np.load(cache) as original:
                        errors = {name: float(np.max(np.abs(original[name]-current)))
                            for name, current in [('hidden', hidden), ('final_norm', norm),
                                ('candidate_logprobs', logprobs), ('rating_probs', probs)]}
                    original_record = next(r for r in records if r['image_id']==row['image_id'] and r['condition']==row['condition'] and r.get('prompt_id')==prompt_id and r['nutrient']==row['nutrient'])
                    assert input_id_sha == original_record['input_ids_sha256']
                    repeat_records.append({'key': key, 'max_absolute_errors': errors,
                                           'exact': all(v == 0 for v in errors.values())})
                if args.mode == 'pilot' or len(records) % 100 == 0 or len(records) == len(rows):
                    print(json.dumps({'event': 'forward', 'key': key, 'pass': pass_id,
                                      'completed_inputs': len(records), 'total_inputs': len(rows),
                                      'seconds': time.time()-item_start}), flush=True)
                del result, logits, inputs
    for h in handles:
        h.remove()
    record_stream.close()
    (out/'records.json').write_text(json.dumps(records, indent=2)+'\n')
    if reference:
        assert len(reference_checks) == reference['inputs']
    report = {
        'status': ('complete' if reference else 'passed') if all(r['exact'] for r in repeat_records) else 'repeat_difference_needs_review',
        'mode': args.mode,
        'inputs': len(records), 'forwards': len(records)+len(repeat_records),
        'repeat_order': 'reversed_all_inputs' if args.mode == 'pilot' else 'no_full_repeat_pilot_reference_checked',
        'repeat_checks': repeat_records, 'pilot_reference_exact_matches': reference_checks,
        'pilot_reference_audit_sha256': sha(Path(args.reference_dir)/'audit.json') if reference else None,
        'candidate_tokenization': tokenizer_audits, 'text_module': text_name,
        'text_layers': len(text_model.layers), 'hidden_size': 4096,
        'norm_QA': 'actual_full_shape_replay_and_block32_input_identity; single_token_replay_difference_is_diagnostic_only',
        'layer_types': model.config.text_config.layer_types,
        'image_processor': processor.image_processor.to_dict(),
        'attention_implementation': 'sdpa', 'dtype': 'bfloat16',
        'seed': 20260916, 'use_cache': False, 'eval': not model.training,
        'gpu': torch.cuda.get_device_name(0),
        'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_GiB': torch.cuda.max_memory_reserved()/2**30,
        'versions': {p: importlib.metadata.version(p) for p in
            ['torch', 'transformers', 'accelerate', 'huggingface_hub', 'numpy', 'pillow', 'tokenizers']},
        'python': platform.python_version(), 'platform': platform.platform(),
        'script_sha256': sha(__file__), 'prompt_config_sha256': sha(args.prompt_config),
        'prompt_renderer_sha256': sha(Path(__file__).with_name('control_prompt_spec.py')),
        'input_manifest_sha256': sha(args.input_manifest),
        'model_repo': cfg['model_repo'], 'model_revision': cfg['model_revision'],
        'model_files': {p.name: sha(p) for p in sorted(Path(args.model_dir).iterdir())
            if p.is_file() and p.suffix in ('.json', '.jinja', '.txt', '.safetensors')},
        'seconds': time.time()-start,
        'limitations': [('technical pilot only' if args.mode == 'pilot' else 'feature extraction only'),
                       'no human or brain targets read', 'value and inferential analyses not executed']}
    if reference:
        assert report['model_files'] == reference['model_files']
        assert report['versions'] == reference['versions']
    (out/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    subprocess.run(['nvidia-smi'], stdout=(out/'nvidia_smi.txt').open('w'), check=True)
    (out/'requirements.lock.txt').write_text(subprocess.check_output(
        [__import__('sys').executable, '-m', 'pip', 'freeze'], text=True))
    with (out/'audit_table.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        for r in records:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    print(json.dumps({k: report[k] for k in ['status','inputs','forwards','peak_allocated_GiB','seconds']}), flush=True)


if __name__ == '__main__':
    main()
