import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('experiment',Path(__file__).with_name('experiment.py'))
experiment=importlib.util.module_from_spec(spec);spec.loader.exec_module(experiment)


class EvaluationGuardTests(unittest.TestCase):
    def fixture(self,root):
        final=root/'improved/final';final.mkdir(parents=True)
        tasks=[f'Task{i}' for i in range(24)]
        (root/'task_budgets.csv').write_text('task,official_steps\n'+''.join(f'{t},500\n' for t in tasks))
        (root/'improved/holdout_500_ep5_9.txt').write_text(''.join(f'{t}:500:{e}\n' for t in tasks for e in range(5,10)))
        (root/'improved/development-records').mkdir()
        (root/'results.jsonl').write_text('')
        return tasks

    def test_exactly_five_unique_fresh_cases_per_task(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            self.assertEqual(len(experiment.load_queue(root)),120)
            path=root/'improved/holdout_500_ep5_9.txt'
            path.write_text(path.read_text().replace('Task0:500:5','Task0:500:6'))
            with self.assertRaises(AssertionError): experiment.load_queue(root)

    def test_overlap_with_inspected_cases_stops_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            (root/'results.jsonl').write_text(json.dumps(dict(task='Task0',seed=500,episode=5))+'\n')
            with self.assertRaises(AssertionError): experiment.load_queue(root)

    def test_failed_cache_reset_cannot_silently_advantage_an_arm(self):
        with patch.object(experiment,'http',return_value={'success':False}):
            with self.assertRaises(AssertionError): experiment.reset_cache()

    def test_initial_state_mismatch_stops_paired_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root);out=root/'improved/final/records';out.mkdir()
            for arm in ('authors','my'):
                row=dict(arm=arm,task='Task0',seed=500,episode=5,initial_observable_state_sha256=arm)
                (out/f'{arm}.json').write_text(json.dumps(row))
            with self.assertRaises(AssertionError): experiment.summarize(root)

    def test_render_diagnostics_are_separate_from_action_rpcs(self):
        rpc=dict(method='harness_advance',ok=True,elapsed_ms=125)
        diagnostic=dict(event='freed_render_context',render_guard=True,ctx='unused')
        records,n=experiment.rpc_audit_records([diagnostic,rpc])
        self.assertEqual(records,[rpc]);self.assertEqual(n,1)

    def test_mixed_log_still_detects_reset_and_unknown_records(self):
        advance=dict(method='harness_advance',ok=True,elapsed_ms=125)
        reset=dict(method='reset_step_counter',ok=True,elapsed_ms=2)
        diagnostic=dict(event='reaped_leaked_contexts',render_guard=True)
        with self.assertRaises(AssertionError): experiment.rpc_audit_records([advance,diagnostic,reset])
        with self.assertRaises(AssertionError): experiment.rpc_audit_records([advance,{'event':'unknown'}])

    def test_accounting_amendment_cannot_change_the_frozen_harness(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root);final=root/'improved/final'
            frozen=final/'freeze.json';frozen.write_text(json.dumps({'sha256':{'harness.py':'original'}}))
            change=dict(original_freeze_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                        changes={'harness.py':dict(before='original',after='changed')})
            (final/'accounting-amendment-001.json').write_text(json.dumps(change))
            with self.assertRaises(AssertionError): experiment.verify(root)


if __name__=='__main__': unittest.main()
