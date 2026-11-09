"""Restore a complete four-rank checkpoint, validate updates, and continue to80k."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--resume-from', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = Path('/data/heping/robomme_musa/e2e')
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
        raise ValueError('Requires micro16, accumulation1, global64')
    expected = subprocess.check_output(['git', '-C', str(repo), 'show',
        'b4432b1ef2ab746b13eec32f25bcc5862c582164:configs/train/robomme_percmem_v1_e2e_repro.yaml'])
    if Path(os.environ['ROBOMME_MUSA_CONFIG']).read_bytes() != expected:
        raise ValueError('Source experiment YAML differs')
    checkpoint = args.resume_from.resolve()
    complete = json.loads((checkpoint / 'complete.json').read_text())
    if not all((checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0 for name in complete['files']):
        raise ValueError('Incomplete checkpoint files')
    if complete['step'] >= 79990:
        raise ValueError('Not enough updates for this bounded recovery validation')
    gmi = subprocess.check_output(['mthreads-gmi'], text=True)
    for index in range(4, 8):
        line = re.search(r'^' + str(index) + r'\s+MTT[^\n]+', gmi, re.MULTILINE)
        if line is None or re.search(r'\s0MiB\(', line.group()) is None:
            raise RuntimeError('Physical GPU' + str(index) + ' is occupied')
    logs = root / 'logs' / args.run_name
    logs.mkdir(parents=True, exist_ok=False)
    (logs / 'gpu_before.txt').write_text(gmi)
    previous = root / 'training.status.json'
    (logs / 'previous_supervisor_state.json').write_text(previous.read_text())
    parent = checkpoint.parent.parent
    state = {
        'pid': os.getpid(), 'process_group': os.getpgrp(), 'worktree': str(repo),
        'branch': subprocess.check_output(['git', '-C', str(repo), 'branch', '--show-current'], text=True).strip(),
        'commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'run_name': args.run_name, 'source_commit': 'b4432b1ef2ab746b13eec32f25bcc5862c582164',
        'source_profile_sha256': hashlib.sha256(expected).hexdigest(),
        'initial_resume_checkpoint': str(checkpoint), 'resume_checkpoint': str(checkpoint),
        'resume_step': complete['step'], 'target_optimizer_updates': 80000,
        'physical_gpus': [4, 5, 6, 7], 'micro_batch_per_rank': 16, 'accumulation': 1,
        'global_batch': 64, 'gradient_loss_scale': profile['musa']['gradient_loss_scale'],
        'loss_scale_policy': 'halve and replay identical batch/RNG on nonfinite gradients',
        'started_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'train_log': str(logs / 'train.log'), 'preflight_log': str(logs / 'preflight.log'),
    }
    def update(**values):
        state.update(values)
        temp = root / 'training.status.json.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(root / 'training.status.json')
    def run(stage, name, extra, logfile):
        command = ['bash', str(repo / 'scripts/run_musa_percmem_e2e.sh'), 'train',
            '--resume-from', str(checkpoint), '--run-name', name, *extra]
        with logfile.open('a') as log:
            child = subprocess.Popen(command, cwd=repo, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            update(stage=stage, child_pid=child.pid, command=command,
                   train_status=str(parent / name / 'status.json'))
            code = child.wait()
        if code:
            raise RuntimeError(stage + ' exited ' + str(code))
        return json.loads((parent / name / 'status.json').read_text())
    try:
        short = run('resume_preflight', args.run_name + '_preflight', ['--steps', '10'], logs / 'preflight.log')
        if (short['stage'] != 'complete' or not short['checkpoint_reload_fixed_input_identical']
                or not short['frozen_hashes_unchanged'] or not short['scorer_gradient_observed']):
            raise ValueError('Restored four-rank preflight did not pass')
        if not all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm'])
                   and row['scorer_grad_norm'] > 0 for row in short['records']):
            raise ValueError('Invalid preflight metric')
        if short['records'][-1]['scorer_parameter_delta_l2'] <= 0:
            raise ValueError('Scorer weights did not update')
        checkpoint = Path(short['checkpoint'])
        update(preflight_result=short, resume_checkpoint=str(checkpoint), resume_step=short['steps'])
        result = run('training', args.run_name, [], logs / 'train.log')
        if result['stage'] == 'paused':
            update(stage='paused', resume_checkpoint=result['checkpoint'], resume_step=result['steps'])
        elif result['stage'] == 'complete' and result['steps'] == 80000:
            update(stage='complete', finished_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        else:
            raise ValueError('Unexpected terminal training state')
    except BaseException as error:
        update(stage='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
