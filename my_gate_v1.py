"""Training-free supervision scheduling from RGB and robot proprioception only.

No simulator predicates, object poses, task identifiers, seeds, or outcome labels
are inputs. Cached decisions have a fixed expiry and are never renewed by hits.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path

NUMBER = r"[-+]?\d+(?:\.\d+)?"
ROW = re.compile(r"^(\d+)\s+(\d+)-(\d+)\s+(\[[^]]+\])\s+(\[[^]]+\])\s+("+NUMBER+r")\s+("+NUMBER+r")\s+("+NUMBER+r")\s+(\[[^]]+\])\s+(\[[^]]+\])\s*$")
POSE = re.compile(r"End-effector is now at (\[[^]]+\]), aperture ("+NUMBER+r") mm\.")
ANGLE = re.compile(r"Wrist orientation now: roll/pitch/yaw (\[[^]]+\])")

def parse_proprio(text):
    rows = []
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            rows.append({'step':int(m[3]), 'cmd':ast.literal_eval(m[4]),
                         'grip':float(m[6]), 'moved_cm':float(m[7]),
                         'width':float(m[8]), 'xyz':ast.literal_eval(m[9]),
                         'rpy':ast.literal_eval(m[10])})
    pose, angle = POSE.search(text), ANGLE.search(text)
    if not pose or not angle or not rows:
        raise ValueError('public proprioception is incomplete')
    xyz, rpy = ast.literal_eval(pose[1]), ast.literal_eval(angle[1])
    values = list(xyz)+list(rpy)+[float(pose[2])]
    if len(xyz)!=3 or len(rpy)!=3 or not all(math.isfinite(v) for v in values):
        raise ValueError('invalid public proprioception')
    return {'xyz':xyz,'rpy':rpy,'width':float(pose[2]),
            'close_seen':any(r['grip']>0.5 or r['width']<30 for r in rows),
            'rows':rows}

def rgb_features(path):
    from PIL import Image
    with Image.open(path) as source:
        im=source.convert('RGB'); w,h=im.size
        if w<256 or w%256 or h<768:
            raise ValueError('unexpected montage shape')
        # A batch ends with the newest 3-camera montage. Read its newest column.
        strip=im.crop((w-256,h-768,w,h))
        return [list(strip.crop((0,c*256,256,(c+1)*256)).resize((32,32)).tobytes()) for c in range(3)]

def visual_distance(a,b):
    means=[]; patches=[]
    for left,right in zip(a,b):
        if len(left)!=3072 or len(right)!=3072:
            raise ValueError('bad RGB signature')
        delta=[abs(x-y)/255 for x,y in zip(left,right)]
        means.append(sum(delta)/len(delta))
        for by in range(0,32,4):
            for bx in range(0,32,4):
                ids=[(y*32+x)*3+c for y in range(by,by+4) for x in range(bx,bx+4) for c in range(3)]
                patches.append(sum(delta[i] for i in ids)/len(ids))
    return max(means),max(patches)

def unchanged(a,b):
    mean,patch=visual_distance(a['rgb'],b['rgb'])
    angles=[abs((x-y+180)%360-180) for x,y in zip(a['rpy'],b['rpy'])]
    return (math.dist(a['xyz'],b['xyz'])<=0.005 and max(angles)<=5
            and abs(a['width']-b['width'])<=2 and a['grasp_count']==b['grasp_count']
            and a['close_seen']==b['close_seen'] and mean<=0.008 and patch<=0.04)

def decide(state, observation, *, step, budget, kind, interventions, verify=False,
           gave_up=False, exhausted=False, cache=True, watchdog=True):
    result={'decision':'check','reason':'fresh_observation','force_judge':False,
            'width_mm':observation['width'] if observation else 99,
            'close_seen':observation['close_seen'] if observation else True}
    if gave_up or exhausted:
        result.update(decision='handoff',reason='no_intervention_authority')
        return result
    if observation is None or not 0<=step<=budget or budget<=0:
        state.pop('credit',None)
        result['reason']='invalid_observation_no_reuse'
        return result
    old=state.get('history',[])
    if old and step<=old[-1]['step']:
        state.clear(); old=[]
    observation=dict(observation,close_seen=bool(state.get('ever_closed')) or observation['close_seen'])
    state['ever_closed']=observation['close_seen']
    result['close_seen']=observation['close_seen']
    current={k:observation[k] for k in ('xyz','rpy','width','close_seen','grasp_count','rgb')}
    current['step']=step
    history=[x for x in old if step-x['step']<=150]+[current]
    state['history']=history[-12:]
    state['current']=current
    state['interventions']=interventions
    # Never suppress the required first check after a repair.
    if verify:
        state.pop('credit',None);result['reason']='post_repair_verification';return result
    # A permanent pre-grasp gate must not hide a stalled, open-handed approach.
    long=[x for x in history if step-x['step']>=100]
    if watchdog and kind=='pnp' and not observation['close_seen'] and step>=0.35*budget and long:
        origin=long[-1]
        recent=[x for x in history if x['step']>=origin['step']]
        stationary=all(math.dist(x['xyz'],origin['xyz'])<=0.015 for x in recent)
        mean,patch=visual_distance(origin['rgb'],current['rgb'])
        if stationary and mean<=0.015 and patch<=0.06:
            state.pop('credit',None)
            result.update(force_judge=True,reason='stalled_pregrasp_100_steps')
            return result
    credit=state.get('credit')
    if cache and credit and credit['interventions']==interventions:
        age=step-credit['observation']['step']
        if 0<age<=50 and unchanged(credit['observation'],current):
            result.update(decision='reuse',reason=credit['source']+'_unchanged_age_'+str(age))
    return result

def remember(state, source):
    if source in ('judge_ok','screen_pass') and state.get('current'):
        state['credit']={'source':source,'observation':state['current'],
                         'interventions':state.get('interventions',0)}
    else:
        state.pop('credit',None)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['observe','remember'])
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--source',default='')
    p.add_argument('--telemetry',type=Path);p.add_argument('--montage',type=Path)
    p.add_argument('--step',type=int,default=0);p.add_argument('--budget',type=int,default=0)
    p.add_argument('--kind',default='mech');p.add_argument('--grasp-count',type=int,default=0)
    p.add_argument('--interventions',type=int,default=0);p.add_argument('--verify',type=int,default=0)
    p.add_argument('--gave-up',type=int,default=0);p.add_argument('--exhausted',type=int,default=0)
    p.add_argument('--cache',type=int,default=1);p.add_argument('--watchdog',type=int,default=1)
    a=p.parse_args()
    try: state=json.loads(a.state.read_text())
    except (OSError,ValueError): state={}
    if a.command=='remember':
        remember(state,a.source);out={'recorded':a.source}
    else:
        try:
            obs=parse_proprio(a.telemetry.read_text())
            obs.update(rgb=rgb_features(a.montage),grasp_count=a.grasp_count)
        except (OSError,ValueError,TypeError,SyntaxError): obs=None
        out=decide(state,obs,step=a.step,budget=a.budget,kind=a.kind,
                   interventions=a.interventions,verify=bool(a.verify),gave_up=bool(a.gave_up),
                   exhausted=bool(a.exhausted),cache=bool(a.cache),watchdog=bool(a.watchdog))
    a.state.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.state.with_suffix('.tmp');tmp.write_text(json.dumps(state));tmp.replace(a.state)
    print(json.dumps(out))

if __name__=='__main__': main()
