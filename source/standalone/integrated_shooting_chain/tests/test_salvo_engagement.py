"""Miss-based engagement regressions independent of OpenCV and Isaac."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from integrated_shooting_chain.salvo_engagement import SalvoEngagementConfig, SalvoEngagementScheduler


def test_fifth_actual_miss_switches_but_time_and_four_misses_do_not():
    scheduler=SalvoEngagementScheduler(SalvoEngagementConfig(switch_policy='misses'))
    candidates={0:80.,1:100.}
    assert scheduler.choose(0.,candidates)==0
    assert scheduler.choose(60.,candidates)==0  # Waiting/aiming alone never consumes a shot.
    for i in range(4):
        scheduler.record_shot(0,hit=False)
        assert scheduler.choose(61.+i,candidates)==0
    assert scheduler.consecutive_misses==4
    scheduler.record_shot(0,hit=False)
    assert scheduler.choose(70.,candidates)==1
    event=scheduler.events[-2]
    assert event['reason']=='consecutive_misses'
    assert event['consecutive_misses']==event['attempt_shots']==5
    assert scheduler.attempt_shots==scheduler.consecutive_misses==0
    assert not scheduler.removed


def test_fifth_shot_hit_takes_priority_over_miss_switch():
    scheduler=SalvoEngagementScheduler()
    candidates={0:80.,1:100.}
    scheduler.choose(0.,candidates)
    for _ in range(4):scheduler.record_shot(0,hit=False)
    scheduler.record_shot(0,hit=True)
    assert scheduler.attempt_shots==5 and scheduler.consecutive_misses==0
    assert scheduler.choose(1.,candidates,removed_ids=[0])==1
    assert [e['event'] for e in scheduler.events]==['selected','destroyed','selected']
    assert scheduler.summary()['deferred_count']==0


def test_retry_resets_misses_and_exhaustion_does_not_destroy_target():
    scheduler=SalvoEngagementScheduler(SalvoEngagementConfig(retry_delay_s=1.,max_attempts=2))
    candidates={0:80.,1:100.}
    scheduler.choose(0.,candidates)
    for _ in range(5):scheduler.record_shot(0,hit=False)
    assert scheduler.choose(1.,candidates)==1
    scheduler.record_shot(1,hit=True)
    assert scheduler.choose(1.1,candidates,removed_ids=[1]) is None
    assert scheduler.terminal_reason(candidates) is None
    assert scheduler.choose(2.,candidates)==0
    assert scheduler.consecutive_misses==scheduler.attempt_shots==0
    assert scheduler.summary()['exhausted_target_ids']==[]
    for _ in range(5):scheduler.record_shot(0,hit=False)
    assert scheduler.choose(3.,candidates) is None
    assert scheduler.summary()['exhausted_target_ids']==[0]
    assert scheduler.terminal_reason(candidates)=='attempts_exhausted'
    assert scheduler.removed=={1}  # Failed shots never score a kill.
    scheduler.reset()
    assert scheduler.current_id is None and not scheduler.attempts
    assert scheduler.consecutive_misses==scheduler.attempt_shots==0


def test_tracking_loss_still_switches_without_any_shots():
    scheduler=SalvoEngagementScheduler()
    assert scheduler.choose(0.,{0:80.,1:100.})==0
    assert scheduler.choose(.02,{1:100.})==1
    assert scheduler.events[-2]['reason']=='tracking_lost'
    assert scheduler.events[-2]['attempt_shots']==0


def test_other_target_or_no_target_shot_cannot_consume_current_budget():
    scheduler=SalvoEngagementScheduler(SalvoEngagementConfig(switch_policy='misses'))
    scheduler.record_shot(0,hit=False)
    scheduler.choose(0.,{0:80.})
    for _ in range(10):scheduler.record_shot(9,hit=False)
    assert scheduler.attempt_shots==scheduler.consecutive_misses==0
    assert scheduler.choose(100.,{0:80.})==0


def test_legacy_timeout_is_explicit_and_separate_from_miss_policy():
    scheduler=SalvoEngagementScheduler(SalvoEngagementConfig(stall_timeout_s=3.,switch_policy='timeout'))
    scheduler.choose(0.,{0:80.,1:100.})
    for _ in range(5):scheduler.record_shot(0,hit=False)
    assert scheduler.choose(1.,{0:80.,1:100.})==0
    assert scheduler.choose(3.,{0:80.,1:100.})==1
    assert scheduler.events[-2]['reason']=='no_hit'


def test_default_both_exits_after_three_seconds_without_a_single_shot():
    scheduler=SalvoEngagementScheduler()
    assert scheduler.config.switch_policy=='both'
    candidates={0:80.,1:100.}
    assert scheduler.choose(0.,candidates)==0
    assert scheduler.choose(2.99,candidates)==0
    assert scheduler.choose(3.,candidates)==1
    event=scheduler.events[-2]
    assert event['reason']=='no_hit' and event['attempt_shots']==0
    assert scheduler.attempts=={0:1,1:1}


def test_default_both_exits_on_fifth_miss_before_three_seconds():
    scheduler=SalvoEngagementScheduler();candidates={0:80.,1:100.}
    scheduler.choose(0.,candidates)
    for shot in range(1,6):
        scheduler.record_shot(0,hit=False)
        assert scheduler.choose(shot*.12,candidates)==(0 if shot<5 else 1)
    assert scheduler.events[-2]['reason']=='consecutive_misses'


def test_default_both_hit_on_timeout_boundary_wins_over_both_failure_triggers():
    scheduler=SalvoEngagementScheduler();candidates={0:80.,1:100.}
    scheduler.choose(0.,candidates)
    for _ in range(4):scheduler.record_shot(0,hit=False)
    scheduler.record_shot(0,hit=True)
    assert scheduler.choose(3.,candidates,removed_ids=[0])==1
    assert scheduler.events[-2]['event']=='destroyed'
    assert scheduler.summary()['deferred_count']==0


@pytest.mark.parametrize('kwargs',[
    dict(switch_policy='invalid'),dict(switch_policy='both',stall_timeout_s=None),
    dict(switch_policy='timeout',stall_timeout_s=None),
])
def test_invalid_combined_policy(kwargs):
    with pytest.raises(ValueError):SalvoEngagementConfig(**kwargs)


@pytest.mark.parametrize('limit',[0,-1,2.5,True])
def test_miss_limit_must_be_positive_integer(limit):
    with pytest.raises(ValueError,match='miss_limit'):
        SalvoEngagementConfig(miss_limit=limit)


@pytest.mark.parametrize('switch_policy', ['both', 'misses', 'timeout'])
def test_all_100_targets_get_a_turn_before_nearest_target_is_retried(switch_policy):
    scheduler = SalvoEngagementScheduler(SalvoEngagementConfig(switch_policy=switch_policy))
    candidates = {i: 80. + i for i in reversed(range(100))}
    assert scheduler.choose(0., candidates) == 0
    for target_id in range(100):
        # Both deadlines and the retry cooldown have elapsed on every switch.
        # A fresh distance-based selection would keep bouncing between 0 and 1.
        for _ in range(5):
            scheduler.record_shot(target_id, hit=False)
        assert scheduler.choose(3. * (target_id + 1), candidates) == (target_id + 1) % 100
    assert scheduler.attempts[0] == 2
    assert all(scheduler.attempts[i] == 1 for i in range(1, 100))
    assert scheduler.pending_ids == list(range(1, 100))


def test_latest_distance_reorders_untried_targets_but_cannot_promote_failed_target():
    scheduler = SalvoEngagementScheduler()
    candidates = {0: 80., 1: 100., 2: 120., 3: 140.}
    assert scheduler.choose(0., candidates) == 0
    assert scheduler.choose(3., candidates) == 1  # 0 moves behind 2 and 3.
    changed_ranges = {0: 10., 1: 100., 2: 200., 3: 20.}
    assert scheduler.choose(4.1, changed_ranges, removed_ids=[1]) == 3
    assert scheduler.choose(4.2, changed_ranges, removed_ids=[1, 3]) == 2
    assert scheduler.choose(4.3, changed_ranges, removed_ids=[1, 2, 3]) == 0
    assert scheduler.attempts == {0: 2, 1: 1, 2: 1, 3: 1}
    summary = scheduler.summary()
    assert summary['selection'] == 'dynamic_nearest_per_round_deferred_retries'
    assert summary['selection_round'] == 2
    assert summary['pending_target_ids'] == []
    assert summary['round_pending_target_ids'] == summary['next_round_target_ids'] == []


def test_missing_tracks_do_not_block_available_current_round_targets():
    scheduler = SalvoEngagementScheduler()
    assert scheduler.choose(0., {0: 80., 1: 100., 2: 120., 3: 140.}) == 0
    # 0 is lost and requeued; 1 is temporarily unavailable and does not block 2.
    assert scheduler.choose(.02, {2: 120., 3: 140.}) == 2
    assert set(scheduler.pending_ids) == {1, 3}
    assert scheduler.retry_ids == [0]
    assert scheduler.events[-2]['reason'] == 'tracking_lost'
    assert scheduler.choose(1.1, {0: 80., 1: 100., 3: 140.}, removed_ids=[2]) == 1
    assert scheduler.pending_ids == [3]
    assert scheduler.retry_ids == [0]


def test_cooling_target_is_skipped_and_new_target_joins_tail():
    scheduler = SalvoEngagementScheduler(SalvoEngagementConfig(retry_delay_s=10.))
    candidates = {0: 80., 1: 100.}
    assert scheduler.choose(0., candidates) == 0
    assert scheduler.choose(3., candidates) == 1
    assert scheduler.choose(6., candidates) is None
    assert scheduler.pending_ids == [0, 1]
    assert scheduler.terminal_reason(candidates) is None
    assert scheduler.choose(6.1, {**candidates, 2: 50.}) == 2
    assert scheduler.pending_ids == [0, 1]
    assert scheduler.choose(6.2, candidates, removed_ids=[2]) is None
    assert scheduler.choose(13., candidates) == 0
    assert scheduler.pending_ids == [1]


def test_removed_and_exhausted_targets_leave_queue_and_reset_clears_it():
    scheduler = SalvoEngagementScheduler(SalvoEngagementConfig(max_attempts=1))
    candidates = {0: 80., 1: 100., 2: 120.}
    scheduler.choose(0., candidates)
    assert scheduler.choose(3., candidates, removed_ids=[1]) == 2
    assert scheduler.pending_ids == []  # 0 exhausted; 1 destroyed out of turn.
    assert scheduler.choose(6., candidates) is None
    assert scheduler.terminal_reason(candidates) == 'attempts_exhausted'
    scheduler.reset()
    assert scheduler.pending_ids == []
    assert scheduler.retry_ids == [] and scheduler.selection_round == 1
    assert scheduler.choose(0., {0: 120., 1: 80., 2: 100.}) == 1
    assert scheduler.pending_ids == [2, 0]


def test_distance_changes_do_not_preempt_current_target_but_apply_at_next_selection():
    scheduler = SalvoEngagementScheduler()
    assert scheduler.choose(0., {0: 80., 1: 100., 2: 120.}) == 0
    changed = {0: 80., 1: 100., 2: 60.}
    assert scheduler.choose(.1, changed) == 0
    assert scheduler.choose(.2, changed, removed_ids=[0]) == 2
    assert scheduler.attempts == {0: 1, 2: 1}
    assert scheduler.selection_round == 1


def test_second_round_uses_latest_ranges_and_never_retries_same_round_failure():
    scheduler = SalvoEngagementScheduler()
    candidates = {0: 80., 1: 100., 2: 120.}
    assert scheduler.choose(0., candidates) == 0
    assert scheduler.choose(3., candidates) == 1
    assert scheduler.choose(6., candidates) == 2
    # Round 2 starts: 2 is nearest but cooling; 1 is nearer than 0.
    changed = {0: 140., 1: 50., 2: 10.}
    assert scheduler.choose(9., changed) == 1
    assert scheduler.selection_round == 2
    assert scheduler.attempts == {0: 1, 1: 2, 2: 1}
    # Even after 1's cooldown expires, it waits for 0 and 2 to get a turn.
    assert scheduler.choose(12., changed) == 2
    changed[1] = 1.
    assert scheduler.choose(15., changed) == 0
    summary = scheduler.summary()
    assert summary['selection_round'] == 2
    assert summary['round_pending_target_ids'] == []
    assert summary['next_round_target_ids'] == [1, 2]
    assert scheduler.attempts == {0: 2, 1: 2, 2: 2}
    assert scheduler.choose(18., changed) == 1
    assert scheduler.selection_round == 3


def test_wait_for_cooling_current_round_target_before_promoting_failures_again():
    scheduler = SalvoEngagementScheduler(SalvoEngagementConfig(retry_delay_s=10.))
    candidates = {0: 80., 1: 100.}
    assert scheduler.choose(0., candidates) == 0
    assert scheduler.choose(3., candidates) == 1
    assert scheduler.choose(6., candidates) is None  # Round 2, both cooling.
    assert scheduler.selection_round == 2
    assert scheduler.choose(13., candidates) == 0
    assert scheduler.choose(16., candidates) == 1
    assert scheduler.choose(19., candidates) is None  # Round 3, both cooling.
    assert scheduler.selection_round == 3
    assert scheduler.choose(26., candidates) == 0
    assert scheduler.choose(29., candidates) == 1
    assert scheduler.choose(32., candidates) is None
    assert scheduler.terminal_reason(candidates) == 'attempts_exhausted'
    assert scheduler.attempts == {0: 3, 1: 3}


def test_unavailable_current_round_targets_can_recover_without_early_retry():
    scheduler = SalvoEngagementScheduler()
    assert scheduler.choose(0., {0: 80., 1: 100., 2: 120.}) == 0
    assert scheduler.choose(3., {0: 80., 2: 120.}) == 2
    # Lost ID 1 cannot block a new round; both failures may be retried.
    assert scheduler.choose(6., {0: 80., 2: 120.}) == 0
    assert scheduler.selection_round == 2
    # Recovered ID 1 retains its outstanding chance and can be selected now.
    assert scheduler.choose(9., {0: 80., 1: 50., 2: 120.}) == 1
    assert scheduler.attempts[1] == 1
    assert scheduler.retry_ids == [0]


def test_newly_visible_target_joins_current_round_before_deferred_retry():
    scheduler = SalvoEngagementScheduler()
    assert scheduler.choose(0., {0: 80., 1: 100.}) == 0
    assert scheduler.choose(3., {0: 80., 1: 100.}) == 1
    assert scheduler.choose(6., {0: 80., 1: 100., 9: 150.}) == 9
    assert scheduler.selection_round == 1
    assert scheduler.retry_ids == [0, 1]
    # Hit removes 9 and starts the retry round without waiting another 3 s.
    assert scheduler.choose(6.1, {0: 80., 1: 100.}, removed_ids=[9]) == 0
    assert scheduler.selection_round == 2


def test_distance_ties_are_deterministic_and_removed_retry_never_reappears():
    scheduler = SalvoEngagementScheduler()
    candidates = {9: 80., 3: 80., 7: 100.}
    assert scheduler.choose(0., candidates) == 3
    assert scheduler.choose(3., candidates) == 9
    assert scheduler.choose(6., candidates, removed_ids=[3]) == 7
    assert scheduler.retry_ids == [9]
    assert scheduler.choose(6.1, {9: 80.}, removed_ids=[3, 7]) is None
    assert scheduler.choose(7., {9: 80.}) == 9
    assert scheduler.selection_round == 2 and scheduler.attempts[9] == 2
    assert 3 not in scheduler.summary()['pending_target_ids']
