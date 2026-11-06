"""Bound repeated explicit OK reviews without hiding new contact events.

This adds a short cooldown after two consecutive explicit judge OKs. The
original v1 gate, its strict RGB cache, and repair verification remain intact.
No task names, simulator predicates, seeds, or success labels are consulted.
"""
import math
import my_gate_v1 as base

BASE_DECIDE = base.decide
BASE_REMEMBER = base.remember


def grip_phase(observation):
    rows = observation.get('rows') or []
    if not rows:
        return None
    value = rows[-1]['grip']
    return 1 if value > 0.5 else -1 if value < -0.5 else 0


def decide(state, observation, **kwargs):
    previous = state.get('current')
    contact_changed = bool(observation and previous and (
        observation['grasp_count'] != previous['grasp_count']
        or grip_phase(observation) != previous.get('grip_phase')
        or abs(observation['width'] - previous['width']) > 2))
    repair_changed = state.get('interventions', 0) != kwargs['interventions']
    if contact_changed or repair_changed:
        state['explicit_ok_streak'] = 0
        state.pop('credit', None)
    if observation is None:
        state['explicit_ok_streak'] = 0
        state.pop('current', None)
    result = BASE_DECIDE(state, observation, **kwargs)
    if observation and state.get('current'):
        state['current']['grip_phase'] = grip_phase(observation)
    if result['decision'] != 'check' or result['force_judge']:
        return result
    if (observation is None or kwargs.get('verify') or kwargs.get('gave_up')
            or kwargs.get('exhausted') or not kwargs.get('cache', True)
            or kwargs['step'] >= 0.9 * kwargs['budget']):
        return result
    credit = state.get('credit')
    if (state.get('explicit_ok_streak', 0) < 2 or not credit
            or credit['source'] != 'judge_ok'
            or credit['interventions'] != kwargs['interventions']):
        return result
    before, current = credit['observation'], state['current']
    age = current['step'] - before['step']
    if not (0 < age <= 50 and current.get('grip_phase') is not None
            and current['grip_phase'] == before.get('grip_phase')
            and current['grasp_count'] == before['grasp_count']
            and current['close_seen'] == before['close_seen']
            and abs(current['width'] - before['width']) <= 2
            and math.dist(current['xyz'], before['xyz']) <= 0.03
            and max(abs((x-y+180) % 360-180)
                    for x, y in zip(current['rpy'], before['rpy'])) <= 10):
        return result
    mean, patch = base.visual_distance(before['rgb'], current['rgb'])
    if mean <= 0.02 and patch <= 0.10:
        result.update(decision='reuse', reason='two_explicit_oks_cooldown_age_' + str(age))
    return result


def remember(state, source):
    if not state.get('current'):
        source = 'invalid'
    state['explicit_ok_streak'] = (
        state.get('explicit_ok_streak', 0) + 1 if source == 'judge_ok' else 0)
    BASE_REMEMBER(state, source)


if __name__ == '__main__':
    base.decide = decide
    base.remember = remember
    base.main()
