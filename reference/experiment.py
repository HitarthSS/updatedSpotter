"""One finite, frozen paired evaluation; no tuning or automatic seed replacement."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import threading
import time
import urllib.request


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def load_queue(root):
    lines = (root/'improved/holdout_500_ep5_9.txt').read_text().splitlines()
    cases = [tuple([s.split(':')[0]] + list(map(int, s.split(':')[1:])))
             for s in lines if s.strip()]
    tasks = {r['task'] for r in csv.DictReader((root/'task_budgets.csv').read_text().splitlines())}
    expected = {(t, 500, e) for t in tasks for e in range(5, 10)}
    assert len(tasks) == 24 and len(cases) == 120 and len(set(cases)) == 120
    assert set(cases) == expected, 'Queue must contain exactly five fresh cases per task'
    seen = set()
    for path in (root/'improved/development-records').glob('*.json'):
        r = read_json(path); seen.add((r['task'], r['seed'], r['episode']))
    reference = root/'results.jsonl'
    if not reference.exists():
        reference = root/'Spotter/recovery_explore/runs_paper_24x5_s500_pi05/results.jsonl'
    for line in reference.read_text().splitlines():
        r = json.loads(line); seen.add((r['task'], r['seed'], r['episode']))
    assert not (set(cases) & seen), 'Fresh evaluation overlaps inspected cases'
    return cases


def prepare(root):
    final = root/'improved/final'
    assert not (final/'freeze.json').exists(), 'Never overwrite a frozen evaluation'
    load_queue(root)
    replay = read_json(final/'development-replay.json')
    assert replay['cases'] == 24 and not replay['masked_decisive_verdicts']
    # Both arms get precisely the same read-only audit outside model evidence.
    marker = '  TREAT_T0=$(date +%s)\n'
    audit = '''  printf '%s\\n' "$BEG2" > "$TD/initial_begin.json"
  $H state > "$TD/initial_state.json" || { log "initial audit failed"; exit 10; }
  $H meta > "$TD/initial_meta.json" || { log "initial audit failed"; exit 10; }
'''
    for arm, source in [('authors', root/'Spotter/recovery_explore/judge_driver_v7.sh'),
                        ('my', root/'Spotter/recovery_explore/judge_driver_my_v1.sh')]:
        text = source.read_text(); assert text.count(marker) == 1
        if arm == 'my':
            assert text.count('cli/my_gate_v1.py') == 3
            text = text.replace('cli/my_gate_v1.py', 'cli/my_gate_v2.py')
        text = text.replace(marker, audit+marker)
        path = root/'Spotter/recovery_explore'/f'judge_driver_final_{arm}.sh'
        path.write_text(text)
        subprocess.run(['bash', '-n', str(path)], check=True)
    cli = root/'Spotter/recovery_explore/cli/my_gate_v2.py'
    cli.write_text((final/'my_gate_v2.py').read_text())
    metered = (root/'improved/serve_qwen_metered.py').read_text()
    metered = metered.replace("ROOT=Path('/home/hitarth/spotter-reproduction/improved')",
                              "ROOT=Path('/home/hitarth/spotter-reproduction/improved/final')")
    marker = '    tag=uuid.uuid4().hex; started=time.monotonic()'
    assert metered.count(marker) == 1
    metered = metered.replace(marker, '    if request.seed is None: request.seed=0\n'+marker)
    metered = metered.replace("'kind':'screen' if request.temperature==0 else 'judge'",
                               "'sampling_seed':request.seed,'kind':'screen' if request.temperature==0 else 'judge'")
    (final/'serve_qwen_final.py').write_text(metered)
    launch = (root/'improved/launch_qwen_metered.sh').read_text()
    launch = launch.replace('"$R/improved/serve_qwen_metered.py"', '"$R/improved/final/serve_qwen_final.py"')
    launch = launch.replace('export CUDA_VISIBLE_DEVICES=3 ', 'export VLLM_SERVER_DEV_MODE=1 CUDA_VISIBLE_DEVICES=3 ')
    (final/'launch_qwen_final.sh').write_text(launch)
    print('Prepared separate drivers and shared instrumentation; evaluation not started')


def source_hashes(root):
    paths = {root/'Spotter'/p.decode() for p in subprocess.check_output(
        ['git', '-C', str(root/'Spotter'), 'ls-files', '-z']).split(b'\0') if p}
    paths |= {root/'Spotter/recovery_explore'/name for name in (
        'judge_driver_final_authors.sh', 'judge_driver_final_my.sh', 'judge_driver_my_v1.sh',
        'cli/my_gate_v1.py', 'cli/my_gate_v2.py', 'cli/screen_window_my_v1.py')}
    paths |= {root/name for name in (
        'serve_policy_strict.py', 'protocol.json', 'source-manifest.json',
        'policy-weight-verification.json', 'task_budgets.csv',
        'improved/runtime_env.sh', 'improved/holdout_500_ep5_9.txt',
        'improved/launch_qwen_metered.sh', 'improved/serve_qwen_metered.py')}
    paths |= set((root/'improved/final').glob('*.py'))
    paths |= set((root/'improved/final').glob('*.sh'))
    paths |= {p for p in (root/'Spotter/recovery_explore/memory_bank').rglob('*') if p.is_file()}
    paths |= {root/'improved/final/PROTOCOL.md', root/'improved/final/development-replay.json'}
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}


def freeze(root):
    final = root/'improved/final'; target = final/'freeze.json'
    assert not target.exists(), 'Never overwrite the evaluation freeze'
    cases = load_queue(root)
    weights = read_json(root/'policy-weight-verification.json')
    assert weights['all_tensors_match'] and weights['tensors_verified'] == 813
    write_json(target, dict(phase='fresh paired pilot', version='my harness v2',
                           frozen_unix=time.time(), pairs=len(cases), episodes=240,
                           gpus={'qwen':3, 'policy_simulator':0}, horizon_multiplier=1.8,
                           sha256=source_hashes(root)))


def verify(root):
    final=root/'improved/final'; frozen=final/'freeze.json'
    expected=read_json(frozen)['sha256']
    allowed={'improved/final/experiment.py','improved/final/test_experiment.py'}
    for path in sorted(final.glob('accounting-amendment-*.json')):
        amendment=read_json(path)
        assert amendment['original_freeze_sha256']==hashlib.sha256(frozen.read_bytes()).hexdigest()
        assert set(amendment['changes']) <= allowed, 'Amendment cannot change the harness or model setup'
        for name, change in amendment['changes'].items():
            assert expected[name]==change['before']
            expected[name]=change['after']
    assert expected == source_hashes(root), \
        'Frozen implementation/settings changed; stop and diagnose'


def http(path, body=None):
    req = urllib.request.Request('http://127.0.0.1:8301'+path,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=120) as response:
        data = response.read()
        return json.loads(data) if data else None


def reset_cache():
    response = http('/reset_prefix_cache', {})
    assert isinstance(response, dict) and response.get('success') is True, 'Prefix cache reset failed'
    http('/reset_mm_cache', {})
    http('/reset_encoder_cache', {})


def model_context(final, run, arm, spec):
    write_json(final/'metrics-context.json', dict(run=run, arm=arm, case=spec))


def usage_for(final, run):
    starts, ends = {}, {}
    path = final/'vlm-usage.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            try: entry = json.loads(line)
            except ValueError: continue
            if entry.get('context', {}).get('run') != run: continue
            (starts if entry['event']=='start' else ends)[entry['request']] = entry
    return starts, ends


def drain(final, run):
    # A finished/cancelled client must not leak generation into the next arm.
    for _ in range(30):
        starts, ends = usage_for(final, run)
        if set(starts) <= set(ends): return
        time.sleep(1)
    raise RuntimeError('Outstanding model requests remain; stop before changing arm/context')


def rpc_audit_records(audit):
    """Render-context diagnostics share the log but are not action RPCs."""
    rpc=[]; diagnostics=0
    for entry in audit:
        if 'method' in entry:
            assert isinstance(entry['method'],str) and isinstance(entry.get('ok'),bool)
            assert entry.get('elapsed_ms',-1)>=0
            rpc.append(entry)
        else:
            assert entry.get('render_guard') is True and isinstance(entry.get('event'),str), \
                'Unrecognized simulator log entry; do not silently omit it'
            diagnostics+=1
    assert any(x['method']=='harness_advance' for x in rpc), 'Missing policy-advance audit'
    forbidden={'reset_to_failure','reset_step_counter','debug_reset','restore_snapshot','harness_round_begin'}
    assert not any(x['method'] in forbidden and x['ok'] for x in rpc), 'Reset/rewind detected'
    return rpc,diagnostics


def collect(root, name, arm, case, wall_seconds, audit_offset):
    final = root/'improved/final'; folder = root/'Spotter/recovery_explore'/f'runs_{name}_pi05'
    rows = [json.loads(s) for s in (folder/'results.jsonl').read_text().splitlines() if s.strip()]
    assert len(rows)==1, 'Missing/duplicate result: preserve this case, do not replace'
    r = rows[0]; assert (r['task'], r['seed'], r['episode']) == case
    assert r['policy_family']=='pi05' and r['lane']==0 and r['control_success'] is None
    data = folder/'lane0'/r['cell']; truth = folder/'_truth/lane0'/r['cell']
    initial = read_json(truth/'initial_state.json'); initial_meta = read_json(truth/'initial_meta.json')
    assert set(initial)=={'state', 'oracle'}
    budget_row = next(x for x in csv.DictReader((root/'task_budgets.csv').read_text().splitlines()) if x['task']==case[0])
    budget = int(budget_row['authorized_steps_1_8x']); cutoff = int(budget_row['official_steps'])*3//2
    meta = read_json(final/'terminal-meta'/f'{name}.json')
    assert meta['step_budget_scale']==1.8 and meta['policy_endpoint']=='http://127.0.0.1:8900'
    assert initial_meta['episode']['failure_committed_timestep']==0
    ep = meta['episode']; assert ep['max_steps']==budget
    step = int(ep['failure_committed_timestep']); assert 0 <= step <= budget
    advances = list((data/'async').glob('adv_*.json')); assert advances
    for path in advances:
        a = read_json(path); assert a['max_steps']==budget and 0<=a['committed_timestep']<=step
    for path in data.glob('judge_w*.json'):
        blob = read_json(path); payload = blob.get('result')
        if isinstance(payload, str):
            try: payload = json.loads(payload)
            except ValueError: payload = None
        assert not (isinstance(payload, dict) and payload.get('type')=='error'), \
            f'Unhandled model error: {path}'
    audit_path = root/'Spotter/recovery_explore_runs/rounds_8450/env_service_calls.jsonl'
    with audit_path.open() as stream:
        stream.seek(audit_offset)
        audit = [json.loads(s) for s in stream if s.strip()]
    assert audit, 'Missing environment action audit'
    rpc,render_diagnostics=rpc_audit_records(audit)
    starts, ends = usage_for(final, name)
    assert all(d.get('sampling_seed')==0 for d in starts.values()), 'Wrong common Qwen seed'
    known = [d for d in ends.values() if d.get('usage')]
    source_cached = [(d['usage'].get('prompt_tokens_details') or {}).get('cached_tokens') for d in known]
    digest = hashlib.sha256(json.dumps(initial, sort_keys=True, separators=(',',':')).encode()).hexdigest()
    record = dict(run=name, arm=arm, task=case[0], seed=case[1], episode=case[2],
                  success=bool(r['treatment_success']), treatment_seconds=r['treat_s'],
                  total_wall_seconds=wall_seconds, terminal_steps=step, horizon_limit=budget,
                  horizon_violations=0, success_within_1_5x=bool(r['treatment_success']) and step<=cutoff,
                  initial_observable_state_sha256=digest, initial_audit_path=str(truth/'initial_state.json'),
                  vlm_requests=len(starts), screen_requests=sum(d['kind']=='screen' for d in starts.values()),
                  judge_and_repair_requests=sum(d['kind']=='judge' for d in starts.values()),
                  requests_without_usage=len(starts)-len(known),
                  input_tokens_known=sum(d['usage'].get('prompt_tokens',0) for d in known),
                  output_tokens_known=sum(d['usage'].get('completion_tokens',0) for d in known),
                  cached_input_tokens_known=sum(x for x in source_cached if x is not None),
                  requests_with_unknown_cached_token_count=sum(x is None for x in source_cached),
                  vlm_service_request_seconds=sum(d.get('elapsed',0) for d in ends.values()),
                  environment_rpc_seconds=sum(d['elapsed_ms']/1000 for d in rpc if d['ok']),
                  environment_rpc_records=len(rpc),render_diagnostic_records=render_diagnostics,
                  environment_rpc_error_records=sum(not d['ok'] for d in rpc),interventions=r['interventions'],
                  reused_windows=sum(e.get('judge')=='my_reuse' for e in r['events']),
                  asynchronous_stale_steps=r.get('stale_steps',0), result_path=str(folder/'results.jsonl'))
    write_json(final/'records'/f'{name}.json', record)
    print(json.dumps(record), flush=True)
    return record


def wilson(k, n):
    if not n: return None
    z=1.96; p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [round(100*(center-half),2), round(100*(center+half),2)]


def summarize(root):
    final=root/'improved/final'; groups=defaultdict(dict)
    for path in (final/'records').glob('*.json'):
        r=read_json(path); key=(r['task'],r['seed'],r['episode'])
        assert r['arm'] not in groups[key], 'Duplicate case arm'
        groups[key][r['arm']]=r
    paired=[(k,v) for k,v in groups.items() if set(v)=={'authors','my'}]
    for key, pair in paired:
        assert pair['authors']['initial_observable_state_sha256']==pair['my']['initial_observable_state_sha256'], \
            f'Initial state mismatch: {key}; stop rather than comparing unmatched cases'
    out=dict(phase='fresh paired pilot', target_pairs=120, paired_completed=len(paired),
             completed_episodes=sum(map(len,groups.values())),
             horizon_violations=sum(r['horizon_violations'] for v in groups.values() for r in v.values()),
             initial_observable_state_mismatches=0, frozen_version='my harness v2',
             paper_full_evaluation=False)
    for arm in ('authors','my'):
        records=[v[arm] for k,v in paired]; n=len(records); wins=sum(r['success'] for r in records)
        metrics={field:sum(r[field] for r in records) for field in (
            'vlm_requests','screen_requests','judge_and_repair_requests','input_tokens_known',
            'output_tokens_known','requests_without_usage','treatment_seconds','total_wall_seconds',
            'vlm_service_request_seconds','environment_rpc_seconds','interventions','reused_windows')}
        out[arm]=dict(successes=wins, episodes=n, success_percent=round(100*wins/n,2) if n else None,
                      success_wilson_95_percent=wilson(wins,n),
                      successes_within_1_5x=sum(r['success_within_1_5x'] for r in records), **metrics)
    helped=sum(v['my']['success'] and not v['authors']['success'] for k,v in paired)
    harmed=sum(v['authors']['success'] and not v['my']['success'] for k,v in paired)
    d=helped+harmed
    out.update(helped=helped,harmed=harmed,
               mcnemar_exact_p=min(1.,2*sum(math.comb(d,i) for i in range(min(helped,harmed)+1))/2**d) if d else 1.,
               paired_success_difference_95_percent=None)
    timing_pairs=[v for k,v in paired if not any(r.get('recovered_after_collector_failure') for r in v.values())]
    out['timing_on_uninterrupted_pairs']={
        'paired_cases':len(timing_pairs),'excluded_restart_boundary_pairs':len(paired)-len(timing_pairs),
        'shared_hardware':True,
        **{arm:{field:sum(v[arm][field] for v in timing_pairs) for field in (
            'treatment_seconds','total_wall_seconds','vlm_service_request_seconds','environment_rpc_seconds')}
           for arm in ('authors','my')}}
    if len(paired)==120:
        tasks=defaultdict(list)
        for key,v in paired: tasks[key[0]].append(int(v['my']['success'])-int(v['authors']['success']))
        assert len(tasks)==24 and all(len(x)==5 for x in tasks.values())
        rng=random.Random(8173); names=list(tasks); draws=[]
        for _ in range(5000):
            draw=[rng.choice(tasks[t]) for t in rng.choices(names,k=24) for __ in range(5)]
            draws.append(100*sum(draw)/len(draw))
        draws.sort(); out['paired_success_difference_95_percent']=[draws[125],draws[4874]]
    per_task=[]
    task_names=[r['task'] for r in csv.DictReader((root/'task_budgets.csv').read_text().splitlines())]
    for task in task_names:
        row={'task':task}
        for arm in ('authors','my'):
            rs=[v[arm] for k,v in groups.items() if k[0]==task and arm in v]
            wins=sum(r['success'] for r in rs)
            row.update({arm+'_successes':wins,arm+'_completed':len(rs),
                        arm+'_percent':100*wins/5 if len(rs)==5 else ''})
        per_task.append(row)
    with (final/'per-task.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_task[0]));writer.writeheader();writer.writerows(per_task)
    write_json(final/'summary.json',out)
    print(f"Fresh pilot: {out['completed_episodes']}/240 episodes, {len(paired)}/120 pairs; "
          f"authors {out['authors']['successes']}/{len(paired)}, my harness {out['my']['successes']}/{len(paired)}",flush=True)
    return out


def gpu_snapshot():
    value=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total',
                                   '--format=csv,noheader,nounits','-i','0,3'],text=True,timeout=10)
    return dict(time=time.time(), csv=value.strip())


def sample_gpus(final, stop):
    while not stop.is_set():
        try:
            with (final/'gpu-load.jsonl').open('a') as f: f.write(json.dumps(gpu_snapshot())+'\n')
        except Exception as exc:
            with (final/'gpu-load.jsonl').open('a') as f: f.write(json.dumps(dict(time=time.time(),error=type(exc).__name__))+'\n')
        stop.wait(15)


def run(root):
    final=root/'improved/final'; verify(root); cases=load_queue(root)
    resume=os.environ.get('SPOTTER_RESUME')=='1'
    if resume:
        assert (final/'job-start.json').exists() and read_json(final/'job-status.json')['exit_code']!=0
        assert not (final/'resume-001.json').exists(), 'Inspect existing resume before attempting another'
        write_json(final/'resume-001.json',dict(pid=os.getpid(),time=time.time(),
                                               reason='Collector schema fix; preserve completed episodes'))
    else:
        assert not (final/'job-start.json').exists(), 'Inspect existing run before attempting a resume'
        write_json(final/'job-start.json',dict(pid=os.getpid(),time=time.time(),target_episodes=240,target_my_episodes=120))
    for name in ('records','logs','terminal-meta','executions'): (final/name).mkdir(exist_ok=True)
    stop=threading.Event(); threading.Thread(target=sample_gpus,args=(final,stop),daemon=True).start()
    code=1; failure=None
    try:
        for i,case in enumerate(cases):
            arms=('my','authors') if i%2==0 else ('authors','my')
            for arm in arms:
                verify(root)
                name=f'fresh_v2_{i:03d}_{arm}'; spec=':'.join(map(str,case))
                record_path=final/'records'/f'{name}.json'
                if resume and record_path.exists():
                    completed=read_json(record_path)
                    assert (completed['task'],completed['seed'],completed['episode'],completed['arm'])==(*case,arm)
                    print('PRESERVE completed episode',name,flush=True)
                    continue
                folder=root/'Spotter/recovery_explore'/f'runs_{name}_pi05'
                assert not folder.exists(), f'Existing partial case: {name}; never replace automatically'
                queue=root/'Spotter/recovery_explore'/f'runs_{name}'
                queue.mkdir(); (queue/'episodes_pi05.txt').write_text(spec+'\n')
                reset_cache(); model_context(final,name,arm,spec)
                env=os.environ.copy()
                for key in ('OPENAI_API_KEY','OPENAI_API_TOKEN_FILE','OPENAI_RESPONSES_URL','ANTHROPIC_API_KEY'):
                    env.pop(key,None)
                env.update(RUN=name,TARGET='1',ON_POD='1',NGPU='1',FAMILY_OVERRIDE='pi05',
                           SIM_HOST='127.0.0.1',SIM_MANAGED='0',POLICY_URL='http://127.0.0.1:8900',
                           CTRL_ONLY='0',SKIP_CONTROL='1',MY_CACHE='1',MY_WATCHDOG='1',
                           MY_FIX_PROPRIO='1',MY_DROP_UNACTIONABLE='1')
                env['MEMORY_BANK']=str(root/'Spotter/recovery_explore/memory_bank')
                audit=root/'Spotter/recovery_explore_runs/rounds_8450/env_service_calls.jsonl'
                offset=audit.stat().st_size if audit.exists() else 0
                write_json(final/'current.json',dict(index=i,arm=arm,case=spec,run=name,
                                                     started_unix=time.time(),gpu=gpu_snapshot()))
                print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'START',name,spec,flush=True)
                driver=root/'Spotter/recovery_explore'/f'judge_driver_final_{arm}.sh'
                command=['bash','-c','source "$1"; shift; exec bash "$@"','spotter',
                         str(root/'improved/runtime_env.sh'),str(driver),'0']
                start=time.monotonic()
                with (final/'logs'/f'{name}.log').open('wb') as log:
                    subprocess.run(command,env=env,cwd=root/'Spotter',stdout=log,stderr=subprocess.STDOUT,check=True)
                wall=time.monotonic()-start
                write_json(final/'executions'/f'{name}.json',dict(run=name,arm=arm,case=spec,
                                                                wall_seconds=wall,audit_offset=offset,
                                                                ended_unix=time.time()))
                # No action is taken by this evaluator-side terminal audit.
                cli=[str(root/'cosmos-runtime/envs/robocasa/bin/python'),
                     str(root/'Spotter/recovery_explore/cli/harness.py'),'--endpoint','http://127.0.0.1:8450','meta']
                meta=subprocess.check_output(cli,cwd=root/'Spotter',text=True,timeout=30)
                write_json(final/'terminal-meta'/f'{name}.json',json.loads(meta.splitlines()[-1]))
                drain(final,name); collect(root,name,arm,case,wall,offset); summarize(root)
        code=0
    except BaseException as exc:
        failure=f'{type(exc).__name__}: {exc}'; raise
    finally:
        stop.set()
        write_json(final/'job-status.json',dict(exit_code=code,error=failure,ended_unix=time.time(),
                                               completed_episodes=len(list((final/'records').glob('*.json')))))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','freeze','verify','summarize','run'))
    p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    globals()[a.command](a.root)


if __name__=='__main__': main()
