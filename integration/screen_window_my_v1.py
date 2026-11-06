"""Avoid a model call when the authors' deterministic rule already flags it."""
import os
from pathlib import Path
import runpy
import sys

def install_rule_shortcut(module):
    original=module.screen
    def screen(image_bytes,telemetry_text,task,prev_verdicts=None,**kwargs):
        if os.environ.get('MY_SHORTCIRCUIT_RULE','1')=='1' and module.rule_hit(telemetry_text,task):
            # The final flag is rule OR VLM. The VLM cannot change a True rule.
            return {'flag':True,'risk':None,'rule_hit':True,'why':'deterministic empty-grasp rule',
                    'port':None,'dt':0.,'source':'my_rule_shortcut','model_call_skipped':True}
        return original(image_bytes,telemetry_text,task,prev_verdicts,**kwargs)
    module.screen=screen

def main():
    here=Path(__file__).resolve().parent
    sys.path.insert(0,str(here))
    import screener
    install_rule_shortcut(screener)
    runpy.run_path(str(here/'screen_window.py'),run_name='__main__')

if __name__=='__main__': main()
