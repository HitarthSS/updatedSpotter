"""Run exactly the finite paired pilot, then release this job's owned models."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root=Path('/home/hitarth/spotter-reproduction'); final=root/'improved/final'
models=json.loads((final/'model-start-final.json').read_text())['pids']
code=1
try:
    code=subprocess.call(['python3',str(final/'experiment.py'),'run','--root',str(root)],cwd=root)
finally:
    stopped=[]
    expected={int(models['qwen']):str(final/'serve_qwen_final.py').encode(),
              int(models['policy']):str(root/'serve_policy_strict.py').encode()}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            if proc.stat().st_uid!=os.getuid(): continue
            pid=int(proc.name);args=(proc/'cmdline').read_bytes().split(b'\0')
            own=(pid in expected and expected[pid] in args)
            own=own or (b'recovery_explore.env_service' in args
                        and b'8450' in args and (proc/'cwd').resolve()==root/'Spotter')
            if own:
                os.kill(pid,signal.SIGTERM);stopped.append(pid)
        except (OSError,ValueError): pass
    (final/'cleanup.json').write_text(json.dumps(dict(exit_code=code,stopped_pids=stopped,
                                                     ended_unix=time.time()),indent=2)+'\n')
    print('Finite paired pilot cleanup',stopped,'exit',code,flush=True)
raise SystemExit(code)
