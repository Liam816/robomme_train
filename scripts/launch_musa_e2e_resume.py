"""Validate the step7 numerical repair, then resume the original 80k experiment."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

def main():
    repo = Path(__file__).resolve().parents[1]
    root = Path('/data/heping/robomme_musa/e2e')
    cli = argparse.ArgumentParser()
    cli.add_argument('--run-name', required=True)
    args = cli.parse_args()
    lock = (root / 'training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.sched_setaffinity(0, set(range(64, 96)))
    os.environ.update(MUSA_VISIBLE_DEVICES='4,5,6,7', CUDA_VISIBLE_DEVICES='4,5,6,7',
        ROBOMME_MUSA_ROOT=str(root), ROBOMME_MUSA_REPO=str(repo),
        ROBOMME_MUSA_CONFIG=str(repo / 'configs/train/robomme_percmem_v1_e2e_repro.yaml'),
        ROBOMME_MUSA_RUNTIME=str(repo / 'configs/train/robomme_musa_worker36021.yaml'))
    sys.path.insert(0, str(repo / 'src'))
    from mme_vla_suite.musa.runtime import load_profile
    profile = load_profile()
    if profile['musa']['micro_batch_size'] != 16 or profile['musa']['gradient_accumulation_steps'] != 1:
        raise ValueError('This resume requires micro16, accumulation1, global64/four ranks')
    if profile['musa']['gradient_loss_scale'] != 2**20:
        raise ValueError('Use the validated exact-power-of-two backward scale')
    source = repo / 'configs/train/robomme_percmem_v1_e2e_repro.yaml'
    expected = subprocess.check_output(['git', '-C', str(repo), 'show',
        'b4432b1ef2ab746b13eec32f25bcc5862c582164:configs/train/robomme_percmem_v1_e2e_repro.yaml'])
    if source.read_bytes() != expected:
        raise ValueError('The source experiment YAML changed')
    probe = root / 'profiles/inference20_gradrepair_20261104'
    inference = json.loads((probe / 'inference20.json').read_text())
    gradient = json.loads((probe / 'gradient_diagnostic.json').read_text())
    if not inference.get('passed') or inference['iterations'] != 20 or not gradient.get('passed'):
        raise ValueError('Inference and real-FM numerical checks must pass')
    if not json.loads((root / 'numerics_repair_tests.json').read_text()).get('passed'):
        raise ValueError('Numerical/resume restriction unit checks must pass')
    checkpoint = root / 'runs/ckpts/mme_vla_suite/percmem-v1-e2e-lerobot-s42/profile_resume_e2e_20261104/step_000007'
    if json.loads((checkpoint / 'complete.json').read_text())['step'] != 7:
        raise ValueError('Expected the verified completed update7 checkpoint')
    logs = root / 'logs' / args.run_name
    logs.mkdir(parents=True, exist_ok=False)
    parent = root / 'runs/ckpts/mme_vla_suite/percmem-v1-e2e-lerobot-s42'
    state = {'pid': os.getpid(), 'process_group': os.getpgrp(), 'worktree': str(repo),
        'branch': subprocess.check_output(['git', '-C', str(repo), 'branch', '--show-current'], text=True).strip(),
        'commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'run_name': args.run_name, 'source_commit': 'b4432b1ef2ab746b13eec32f25bcc5862c582164',
        'source_profile_sha256': hashlib.sha256(expected).hexdigest(), 'resume_checkpoint': str(checkpoint),
        'resume_step': 7, 'target_optimizer_updates': 80000, 'physical_gpus': [4, 5, 6, 7],
        'micro_batch_per_rank': 16, 'accumulation': 1, 'global_batch': 64, 'gradient_loss_scale': 2**20,
        'started_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'original_tasks_alive_at_launch': {str(p): Path('/proc', str(p)).exists() for p in [2531277,2531278,3238230,3238231]},
        'train_log': str(logs / 'train.log'), 'preflight_log': str(logs / 'preflight.log')}
    previous = root / 'training.status.json'
    (logs / 'previous_supervisor_state.json').write_text(previous.read_text())
    def update(**values):
        state.update(values)
        temporary = root / 'training.status.json.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n'); temporary.replace(root / 'training.status.json')
    def run(stage, name, extra, logfile):
        command = ['bash', str(repo / 'scripts/run_musa_percmem_e2e.sh'), 'train',
                   '--resume-from', str(checkpoint), '--run-name', name, *extra]
        with logfile.open('a') as log:
            child = subprocess.Popen(command, cwd=repo, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            update(stage=stage, child_pid=child.pid, command=command, train_status=str(parent / name / 'status.json'))
            code = child.wait()
            if code:
                raise RuntimeError(stage + ' exited ' + str(code))
        return json.loads((parent / name / 'status.json').read_text())
    try:
        short = run('repair_preflight', args.run_name + '_preflight', ['--steps', '3'], logs / 'preflight.log')
        if short['stage'] != 'complete' or not short['checkpoint_reload_fixed_input_identical'] or not short['frozen_hashes_unchanged']:
            raise ValueError('Four-rank restored checkpoint validation failed')
        if not short.get('scorer_gradient_observed') or short['records'][-1]['scorer_parameter_delta_l2'] <= 0:
            raise ValueError('The restored real-FM scorer must have a nonzero gradient and parameter update')
        if not all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) for row in short['records']):
            raise ValueError('Preflight contains a non-finite result')
        update(preflight_result=short)
        # Start the authorized long continuation from exactly update7, not the diagnostic checkpoint.
        result = run('training', args.run_name, [], logs / 'train.log')
        if result['stage'] == 'paused':
            update(stage='paused', resume_checkpoint=result['checkpoint'], resume_step=result['steps'])
        elif result['stage'] == 'complete' and result['steps'] == 80000:
            update(stage='complete', finished_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        else:
            raise ValueError('Unexpected terminal training state')
    except BaseException as error:
        update(stage='failed', error=repr(error)); raise

if __name__ == '__main__': main()
