#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


# GPU helpers

def _gpu_free_mb(env):
    """Return free GPU memory in MB on the device selected by CUDA_VISIBLE_DEVICES."""
    try:
        idx = env.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0].strip()
        out = subprocess.check_output(
            ['nvidia-smi', f'--id={idx}',
             '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return int(out.splitlines()[0])
    except Exception:
        return 99999  # unknown → assume plenty


def _wait_for_gpu(env, min_free_mb=8192, poll_secs=60):
    """Block until the selected GPU has at least min_free_mb MB free."""
    while True:
        free = _gpu_free_mb(env)
        if free >= min_free_mb:
            return
        print(f"  [GPU] {free} MB free, need {min_free_mb} MB. Polling in {poll_secs}s ...",
              flush=True)
        time.sleep(poll_secs)


# Lock / status helpers

def _lock_path(runs_dir, exp_name):
    return Path(runs_dir) / exp_name / 'run.lock'


def _status_path(runs_dir, exp_name):
    return Path(runs_dir) / exp_name / 'run_status.json'


def _write_lock(runs_dir, exp_name):
    lp = _lock_path(runs_dir, exp_name)
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(str(os.getpid()))


def _clear_lock(runs_dir, exp_name):
    _lock_path(runs_dir, exp_name).unlink(missing_ok=True)


# CLI

def parse_args():
    p = argparse.ArgumentParser(description="Thesis experiment orchestrator")
    p.add_argument('--config',         default='experiment_config.yaml',
                   help="Path to experiment config YAML (default: experiment_config.yaml)")
    p.add_argument('--profile',        default='smoke', choices=['smoke', 'full'],
                   help="Which profile to run (smoke | full)")
    p.add_argument('--suite_id',       default=None,
                   help="Reuse an existing suite_id for resumption (default: generate new)")
    p.add_argument('--skip_ablations', action='store_true',
                   help="Skip ablation runs; run only primary method comparisons")
    return p.parse_args()


# Config helpers

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def git_info():
    info = {}
    try:
        info['commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.check_output(
            ['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL
        ).decode().strip()
        info['dirty'] = bool(dirty)
        info['dirty_files'] = dirty.splitlines()[:10]   # cap at 10 for readability
    except Exception as e:
        info['error'] = str(e)

    info['library_versions'] = {}
    for lib in ('torch', 'numpy', 'lpips', 'skimage', 'scipy', 'yaml'):
        try:
            mod = __import__(lib)
            info['library_versions'][lib] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            pass
    try:
        import torch
        info['library_versions']['cuda'] = torch.version.cuda or 'n/a'
        info['library_versions']['cudnn'] = str(torch.backends.cudnn.version())
    except Exception:
        pass
    info['python'] = sys.version
    return info


# Run-spec expansion

def expand_runs(profile_cfg, suite_id, skip_ablations):
    """
    Return a list of dicts, each describing one train.py invocation.

    Keys: exp_name, method, scene, seed, ablation_tag, train_args (list of CLI args)
    """
    runs = []
    training   = profile_cfg['training']
    seeds      = profile_cfg['seeds']
    blender    = profile_cfg.get('blender_scenes', [])
    methods    = profile_cfg['methods']
    ablations  = profile_cfg.get('ablations', [])
    abl_scenes = profile_cfg.get('ablation_scenes', blender)

    def _build(method_cfg, scene, seed, ablation_tag, extra_flags=None):
        tag = method_cfg['tag']
        exp_name = f"suite_{suite_id}_{tag}_{scene}_s{seed}_{ablation_tag}"
        args = [
            '--exp_name',    exp_name,
            '--data_root',   f"{profile_cfg.get('__data_root__', '')}/{scene}",
            '--iterations',  str(training['iterations']),
            '--val_interval', str(training['val_interval']),
            '--t',           str(training['hash_table_size']),
            '--seed',        str(seed),
            '--use_warp',    # Warp backend forced for all suite runs
            '--occ_threshold', str(method_cfg.get('occ_threshold', 0.1)),
        ]
        if training.get('use_amp', True):
            args.append('--use_amp')
        if method_cfg.get('use_zip', False):
            args.append('--use_zip')
        nwd = method_cfg.get('zip_nwd_lambda', 0.0)
        if nwd > 0:
            args += ['--zip_nwd_lambda', str(nwd)]
        if 'zip_sigma_scale' in method_cfg:
            args += ['--zip_sigma_scale', str(method_cfg['zip_sigma_scale'])]
        if method_cfg.get('zip_nwd_per_level', False):
            args.append('--zip_nwd_per_level')
        if 'zip_eval_n_samples' in method_cfg:
            args += ['--zip_eval_n_samples', str(method_cfg['zip_eval_n_samples'])]
        if extra_flags:
            args.extend(extra_flags)
        return {
            'exp_name':     exp_name,
            'method':       tag,
            'scene':        scene,
            'seed':         seed,
            'ablation_tag': ablation_tag,
            'train_args':   args,
        }

    # Primary runs (all methods × all scenes × all seeds)
    for method_cfg in methods:
        for scene in blender:
            for seed in seeds:
                runs.append(_build(method_cfg, scene, seed, 'main'))

    # Ablation runs (ablation configs × ablation_scenes × all seeds)
    if not skip_ablations:
        for abl in ablations:
            extra = []
            if abl.get('zip_collapse_samples', False):
                extra.append('--zip_collapse_samples')
            if abl.get('zip_no_downweighting', False):
                extra.append('--zip_no_downweighting')
            k = abl.get('zip_n_samples', 6)
            if k != 6:
                extra += ['--zip_n_samples', str(k)]
            if 'zip_sigma_scale' in abl:
                extra += ['--zip_sigma_scale', str(abl['zip_sigma_scale'])]
            if abl.get('zip_nwd_per_level', False):
                extra.append('--zip_nwd_per_level')
            if 'zip_eval_n_samples' in abl:
                extra += ['--zip_eval_n_samples', str(abl['zip_eval_n_samples'])]
            nwd = abl.get('zip_nwd_lambda', 0.0)
            abl_method = {
                'tag':              abl['tag'],
                'use_zip':          abl.get('use_zip', True),
                'occ_threshold':    abl.get('occ_threshold', 0.5),
                'zip_nwd_lambda':   nwd,
            }
            for scene in abl_scenes:
                for seed in seeds:
                    runs.append(_build(abl_method, scene, seed, abl['tag'], extra))

    # Profiling run (timing + memory; separate exp_name, never merged into CSV)
    prof = profile_cfg.get('profiling', {})
    if prof.get('enabled', False):
        prof_method_tag = prof.get('method', 'baseline')
        prof_scene      = prof.get('scene', 'lego')
        prof_seed       = prof.get('seed', 42)
        prof_interval   = prof.get('profile_interval', 5000)
        matched = next((m for m in methods if m['tag'] == prof_method_tag), methods[0])
        runs.append(_build(matched, prof_scene, prof_seed, 'profile',
                           extra_flags=['--profile_interval', str(prof_interval)]))

    return runs


# Completion check

def is_complete(run_spec, runs_dir):
    """A run is complete when run_status.json reports status=='complete'."""
    sp = _status_path(runs_dir, run_spec['exp_name'])
    if sp.exists():
        try:
            with open(sp) as f:
                return json.load(f).get('status') == 'complete'
        except Exception:
            pass
    # Fallback for runs that completed before run_status.json was introduced
    return (Path(runs_dir) / run_spec['exp_name'] / 'metrics.json').exists()


# Single run executor

def run_one(run_spec, work_dir, env, runs_dir):
    """
    Execute train.py for one run.  Returns (success: bool, stderr_excerpt: str).
    Prints live stdout from the subprocess so the user can watch progress.
    Holds a PID-based lock file for the duration of the run.
    """
    cmd = [sys.executable, '-u', 'train.py'] + run_spec['train_args']
    print(f"\n{'─'*70}", flush=True)
    print(f"  [{run_spec['method']}] {run_spec['scene']}  seed={run_spec['seed']}  "
          f"ablation={run_spec['ablation_tag']}", flush=True)
    print(f"  exp: {run_spec['exp_name']}", flush=True)
    print(f"{'─'*70}", flush=True)

    t0 = time.time()
    child_env = {**env, 'PYTHONUNBUFFERED': '1'}
    _write_lock(runs_dir, run_spec['exp_name'])
    try:
        proc = subprocess.Popen(
            cmd, cwd=work_dir, env=child_env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        stderr_lines = []
        import threading

        def _drain_stderr():
            for line in proc.stderr:
                stderr_lines.append(line)

        t = threading.Thread(target=_drain_stderr, daemon=True)
        t.start()

        for line in proc.stdout:
            print(line, end='', flush=True)

        proc.wait()
        t.join()

        elapsed = time.time() - t0
        if proc.returncode == 0:
            print(f"  ✓  Completed in {elapsed/60:.1f} min")
            return True, ''
        else:
            excerpt = ''.join(stderr_lines[-40:])  # last 40 lines
            print(f"  ✗  FAILED (exit {proc.returncode}) after {elapsed/60:.1f} min")
            print(f"     stderr (last 40 lines):\n{excerpt}")
            return False, excerpt

    except Exception as e:
        return False, str(e)
    finally:
        _clear_lock(runs_dir, run_spec['exp_name'])


# Manifest helpers

def append_manifest(path, record):
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')


# Main

def main():
    sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    cfg  = load_config(args.config)

    paths      = cfg['paths']
    work_dir   = paths['work_dir']
    data_root  = paths['data_root']
    runs_dir   = os.path.join(work_dir, paths['runs_dir'])
    results_dir= os.path.join(work_dir, paths['results_dir'])

    profile_cfg = cfg['profiles'][args.profile]
    profile_cfg['__data_root__'] = data_root

    suite_id = args.suite_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    suite_dir = os.path.join(results_dir, suite_id)
    os.makedirs(suite_dir, exist_ok=True)

    manifest_path  = os.path.join(suite_dir, 'manifest.jsonl')
    failures_path  = os.path.join(suite_dir, 'failures.jsonl')
    git_info_path  = os.path.join(suite_dir, 'git_info.json')
    config_snap_path = os.path.join(suite_dir, 'config_snapshot.yaml')

    print(f"\n{'='*70}")
    print(f"  Thesis Experiment Suite")
    print(f"  Profile:  {args.profile}")
    print(f"  Suite ID: {suite_id}")
    print(f"  Results:  {suite_dir}")
    print(f"  Resume?   {'YES — skipping completed runs' if args.suite_id else 'NO — fresh start'}")
    print(f"  Ablations: {'SKIP' if args.skip_ablations else 'INCLUDE'}")
    print(f"{'='*70}\n")

    if not os.path.exists(git_info_path):
        with open(git_info_path, 'w') as f:
            json.dump(git_info(), f, indent=2)
        print(f"Git info saved to {git_info_path}")

    if not os.path.exists(config_snap_path):
        shutil.copy(args.config, config_snap_path)
        print(f"Config snapshot saved to {config_snap_path}")

    all_runs = expand_runs(profile_cfg, suite_id, args.skip_ablations)
    n_total  = len(all_runs)
    n_done   = sum(1 for r in all_runs if is_complete(r, runs_dir))
    print(f"Total runs: {n_total}  |  Already complete: {n_done}  |  Remaining: {n_total - n_done}")
    print(f"Estimated remaining time: ~{(n_total - n_done) * 37:.0f} min "
          f"(assuming ~37 min/run at 75K iters)\n")

    env = os.environ.copy()
    if 'CUDA_VISIBLE_DEVICES' not in env:
        print("WARNING: CUDA_VISIBLE_DEVICES not set. Defaulting to GPU 0.")
        print("         Set it before running: export CUDA_VISIBLE_DEVICES=2")

    n_success = 0
    n_fail    = 0
    failed_runs = []

    for i, run_spec in enumerate(all_runs):
        # Skip completed runs (resumption support)
        if is_complete(run_spec, runs_dir):
            print(f"[{i+1}/{n_total}] SKIP (already complete): {run_spec['exp_name']}")
            n_success += 1
            continue

        # Skip if a live process already owns the lock. remove stale locks
        lp = _lock_path(runs_dir, run_spec['exp_name'])
        if lp.exists():
            try:
                pid = int(lp.read_text().strip())
                os.kill(pid, 0)  # signal 0 = liveness probe
                print(f"[{i+1}/{n_total}] SKIP (PID {pid} running): {run_spec['exp_name']}",
                      flush=True)
                continue
            except (OSError, ValueError):
                print(f"  [lock] Stale lock for {run_spec['exp_name']} — removing.", flush=True)
                lp.unlink(missing_ok=True)

        _wait_for_gpu(env)

        print(f"\n[{i+1}/{n_total}]", flush=True)
        success, stderr_excerpt = run_one(run_spec, work_dir, env, runs_dir)

        record = {
            'exp_name':     run_spec['exp_name'],
            'method':       run_spec['method'],
            'scene':        run_spec['scene'],
            'seed':         run_spec['seed'],
            'ablation_tag': run_spec['ablation_tag'],
            'status':       'complete' if success else 'failed',
            'timestamp':    datetime.now().isoformat(),
        }
        append_manifest(manifest_path, record)

        if success:
            n_success += 1
        else:
            n_fail += 1
            failed_runs.append(run_spec['exp_name'])
            append_manifest(failures_path, {**record, 'stderr_excerpt': stderr_excerpt[-2000:]})

    print(f"\n{'='*70}")
    print(f"  Suite complete: {n_success} succeeded, {n_fail} failed")
    if failed_runs:
        print(f"  Failed runs:")
        for name in failed_runs:
            print(f"    - {name}")
        print(f"  Full failure log: {failures_path}")
    print(f"{'='*70}\n")

    print("Running aggregate_suite.py to produce CSV + Markdown summary...")
    agg_cmd = [
        sys.executable, 'aggregate_suite.py',
        '--suite_dir',  suite_dir,
        '--runs_dir',   runs_dir,
        '--manifest',   manifest_path,
    ]
    try:
        result = subprocess.run(agg_cmd, cwd=work_dir, capture_output=False, text=True)
        if result.returncode != 0:
            print("WARNING: aggregate_suite.py exited with errors. Check output above.")
    except Exception as e:
        print(f"WARNING: Could not run aggregate_suite.py: {e}")

    print(f"\nResults directory: {suite_dir}")
    print(f"  manifest.jsonl  — per-run completion log")
    print(f"  results.csv     — flat table (all metrics, timing, memory)")
    print(f"  summary.md      — LaTeX-ready tables with significance tests")
    print(f"  git_info.json   — commit hash + library versions")


if __name__ == '__main__':
    main()
