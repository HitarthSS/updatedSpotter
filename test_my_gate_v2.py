import copy
import unittest
import my_gate_v2 as gate


def obs(**changes):
    value = dict(xyz=[0., 0., 0.], rpy=[0., 0., 0.], width=55.,
                 close_seen=False, grasp_count=0, rgb=[[100]*3072 for _ in range(3)],
                 rows=[dict(grip=-0.9)])
    value.update(changes)
    return value


def observe(state, step, observation=None, **changes):
    args = dict(step=step, budget=900, kind='mech', interventions=0)
    args.update(changes)
    return gate.decide(state, observation or obs(), **args)


def two_oks():
    state = {}
    observe(state, 100)
    gate.remember(state, 'judge_ok')
    observe(state, 175)
    gate.remember(state, 'judge_ok')
    return state


class CooldownTests(unittest.TestCase):
    def test_one_ok_does_not_enable_motion_cooldown(self):
        state = {}; observe(state, 100); gate.remember(state, 'judge_ok')
        self.assertEqual(observe(state, 125, obs(xyz=[.02, 0, 0]))['decision'], 'check')

    def test_two_oks_allow_small_motion_but_do_not_renew_expiry(self):
        state = two_oks()
        for step in (200, 225):
            result = observe(state, step, obs(xyz=[.02, 0, 0]))
            self.assertEqual(result['decision'], 'reuse')
        self.assertEqual(state['credit']['observation']['step'], 175)
        self.assertEqual(observe(state, 250, obs(xyz=[.02, 0, 0]))['decision'], 'check')

    def test_contact_events_and_changes_force_a_fresh_check(self):
        for changed in (obs(grasp_count=1), obs(width=40), obs(rows=[dict(grip=.9)]),
                        obs(xyz=[.04, 0, 0]), obs(rpy=[11, 0, 0])):
            self.assertEqual(observe(two_oks(), 200, changed)['decision'], 'check')

    def test_local_rgb_change_cannot_hide_inside_global_mean(self):
        changed = obs(xyz=[.02, 0, 0])
        for y in range(4):
            for x in range(4):
                start = (y*32+x)*3
                changed['rgb'][0][start:start+3] = [255]*3
        self.assertEqual(observe(two_oks(), 200, changed)['decision'], 'check')

    def test_repair_verification_and_end_budget_bypass_motion_cooldown(self):
        for changes in (dict(verify=True), dict(interventions=1), dict(budget=220)):
            self.assertEqual(observe(two_oks(), 200, obs(xyz=[.02, 0, 0]), **changes)['decision'], 'check')

    def test_screen_or_invalid_verdict_cannot_build_explicit_ok_streak(self):
        for source in ('screen_pass', 'invalid'):
            state = two_oks(); gate.remember(state, source)
            self.assertEqual(state['explicit_ok_streak'], 0)
            self.assertEqual(observe(state, 200, obs(xyz=[.02, 0, 0]))['decision'], 'check')

    def test_oracle_labels_do_not_change_decisions(self):
        clean = obs(xyz=[.02, 0, 0]); poisoned = copy.deepcopy(clean)
        poisoned.update(task_success=True, object_xyz=[0, 0, 0], is_grasped=True)
        self.assertEqual(observe(two_oks(), 200, clean), observe(two_oks(), 200, poisoned))

    def test_invalid_observation_cannot_recredit_a_stale_observation(self):
        state=two_oks()
        gate.decide(state,None,step=200,budget=900,kind='mech',interventions=0)
        gate.remember(state,'judge_ok')
        self.assertEqual(state['explicit_ok_streak'],0)
        self.assertNotIn('credit',state)


if __name__ == '__main__':
    unittest.main()
