"""Real scene geometry plus regression tests for pool/actor/vision boundaries."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
from integrated_shooting_chain.mujoco_sim.salvo import (
    ASSET, SalvoActorGate, SalvoPerceptionPipeline, make_report, main, project_path,
    RecurrentShootingPolicy, MujocoCameraSensor, mujoco, np, torch,
)
from integrated_shooting_chain.mujoco_sim.salvo_scene import (
    build_salvo_model, sample_layout, MujocoStaticTargetPool,
)
from integrated_shooting_chain.salvo_engagement import SalvoEngagementConfig, SalvoEngagementScheduler
from integrated_shooting_chain.salvo_ablation import load_layout, save_layout


@pytest.fixture
def scene():
    model = build_salvo_model(ASSET)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('stand').id)
    mujoco.mj_forward(model, data)
    camera = MujocoCameraSensor(model, ASSET/'camera_config.json')
    pool = MujocoStaticTargetPool(model, data)
    points = sample_layout(*camera.pose(data), 100)
    pool.place(points)
    yield model, data, camera, pool
    camera.close()


def test_100_targets_are_world_static_and_bootstrap_once(scene):
    model, data, camera, pool = scene
    assert model.nmocap == 100
    boxes = pool.bootstrap_boxes(camera)
    assert set(boxes) == set(range(100))
    assert camera.project_world(data, pool.positions)[2].all()
    before = data.mocap_pos.copy()
    for _ in range(10):
        mujoco.mj_step(model, data)
    np.testing.assert_array_equal(before, data.mocap_pos)
    with pytest.raises(RuntimeError, match='only once'):
        pool.bootstrap_boxes(camera)
    with pytest.raises(RuntimeError, match='only be placed once'):
        pool.place(pool.positions)


def test_selection_preserves_visual_pool_and_hit_removes_one(scene):
    model, data, camera, pool = scene
    pool.select(42)
    assert (model.geom_group[pool.geom_ids] == 4).sum() == 1
    assert (model.geom_group[pool.geom_ids] == 5).sum() == 99
    assert model.geom_rgba[pool.geom_ids, 3].sum() == 100
    untouched = data.mocap_pos[pool.mocap_ids[41]].copy()
    pool.remove(42)
    assert pool.removed_ids == {42}
    assert pool.alive.sum() == 99
    assert model.geom_rgba[pool.geom_ids[42], 3] == 0
    assert data.mocap_pos[pool.mocap_ids[42], 2] < -900
    np.testing.assert_array_equal(untouched, data.mocap_pos[pool.mocap_ids[41]])
    pool.select(41)
    with pytest.raises(ValueError):
        pool.select(42)
    with pytest.raises(ValueError):
        pool.remove(42)


def test_layout_seed_and_shared_isaac_format(scene, tmp_path):
    _, data, camera, pool = scene
    points = sample_layout(*camera.pose(data), 100)
    np.testing.assert_array_equal(points, pool.positions)
    assert not np.array_equal(points, sample_layout(*camera.pose(data), 100, seed=62))
    path = tmp_path/'layout.json'
    digest = save_layout(path, torch.tensor(points, dtype=torch.float32), .35)
    loaded, actual = load_layout(path, 100, .35)
    assert actual == digest
    np.testing.assert_array_equal(loaded.numpy(), points)
    with pytest.raises(ValueError):
        load_layout(path, 99, .35)
    record = json.loads(path.read_text())
    record['positions_env_local_m'][0][0] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='checksum'):
        load_layout(path, 100, .35)


def test_synthetic_100_hit_accounting(scene):
    _, _, _, pool = scene
    scheduler = SalvoEngagementScheduler()
    # Unit-test answers only; this does not claim 100 policy-generated hits.
    distances = {i: 100.+i for i in range(100)}
    shots = []
    for i in range(100):
        selected = scheduler.choose(i*.1, distances, removed_ids=pool.removed_ids)
        assert selected == i
        scheduler.record_shot(selected, hit=True)
        pool.remove(selected)
        shots.append(dict(target_id=selected, hit=True))
    report = make_report(pool, shots, reason='cleared')
    assert report['cleared'] and report['destroyed'] == 100
    assert report['remaining'] == 0 and report['accuracy'] == 1
    assert len(report['destroyed_target_ids']) == 100


def test_retry_round_delay_and_attempt_limits():
    scheduler = SalvoEngagementScheduler(SalvoEngagementConfig(max_attempts=2))
    distances = {0: 10., 1: 20.}
    assert scheduler.choose(0., distances) == 0
    for _ in range(5):
        scheduler.record_shot(0, hit=False)
    assert scheduler.choose(.1, distances) == 1  # first round before deferred ID
    assert scheduler.choose(3.1, distances) == 0  # no-shot stall triggers a switch
    assert scheduler.choose(6.1, distances) == 1
    assert scheduler.choose(9.1, distances) is None
    assert scheduler.terminal_reason(distances) == 'attempts_exhausted'
    single = SalvoEngagementScheduler()
    assert single.choose(0., {0: 10.}) == 0
    assert single.choose(3., {0: 10.}) is None
    assert single.terminal_reason({0: 10.}) is None
    assert single.choose(4., {0: 10.}) == 0


def test_track_loss_keeps_original_denominator(scene):
    report = make_report(scene[3], [], reason='tracking_lost')
    assert report['targets_per_episode'] == 100 and report['remaining'] == 100
    assert report['destroyed'] == 0 and not report['cleared']


def test_actor_gate_wait_first_target_and_identity_switch():
    class Policy:
        calls = resets = 0
        def __call__(self, obs):
            self.calls += 1
            return np.zeros(3)
        def reset_actor(self):
            self.resets += 1
    policy = Policy()
    policy(np.zeros(28))  # stand warmup
    gate = SalvoActorGate(policy)
    gate.action(7, False, np.zeros(28))
    assert policy.calls == 1 and policy.resets == 0
    gate.action(7, True, np.zeros(28))
    gate.action(None, False, np.zeros(28))
    gate.action(7, True, np.zeros(28))
    assert policy.resets == 0
    gate.action(8, False, np.zeros(28))
    gate.action(8, True, np.zeros(28))
    assert policy.resets == 1 and gate.resets == 1
    assert policy.calls == 4


def test_actor_reset_works_for_inference_tensors_and_preserves_critic():
    policy = object.__new__(RecurrentShootingPolicy)
    with torch.inference_mode():
        policy.states = [torch.ones(1, 1, 4), torch.full((1, 1, 4), 9.)]
    policy.calls = 17
    policy.reset_actor()
    assert policy.states[0].count_nonzero() == 0
    assert (policy.states[1] == 9).all() and policy.calls == 17


def test_visual_tracks_survive_selected_removal_and_forbid_gt_rebootstrap():
    class Tracker:
        def __init__(self):
            self.ids, self.init_calls, self.updates = set(), 0, 0
        def result(self):
            return SimpleNamespace(targets=[SimpleNamespace(target_id=i, status='tracked') for i in self.ids])
        def initialize(self, image, boxes, timestamp_s):
            self.ids = set(boxes)
            self.init_calls += 1
            return self.result()
        def selectable_target_distances(self):
            return {i: 10.+i for i in self.ids}
        def set_selection_candidates(self, ids):
            self.candidates = ids
        def remove_targets(self, ids):
            self.ids.difference_update(ids)
        def update(self, image, dt_s, timestamp_s):
            self.updates += 1
            return self.result()
    pipeline = object.__new__(SalvoPerceptionPipeline)
    resets = []
    pipeline.tracker = Tracker()
    pipeline.velocity = SimpleNamespace(reset=lambda i: resets.append(i))
    pipeline.stabilize_camera = False
    pipeline.scheduler = SalvoEngagementScheduler()
    pipeline.bootstrap_calls = 0
    pipeline.removed = set()
    pipeline.selected_id = pipeline.previous_time = None
    frame = SimpleNamespace(rgb=np.zeros((8, 8, 3), dtype=np.uint8), time=2.)
    boxes = {i: (0., 0., 4., 4.) for i in range(100)}
    assert pipeline.track(frame, initial_boxes=boxes).target_id == 0
    frame.time += .02
    assert pipeline.track(frame, removed_ids={0}).target_id == 1
    assert pipeline.tracker.init_calls == 1 and len(pipeline.tracker.ids) == 99
    assert pipeline.tracker.updates == 1 and resets == [0, 0]
    frame.time += .02
    with pytest.raises(RuntimeError, match='rebootstrap'):
        pipeline.track(frame, initial_boxes=boxes)


def test_dry_run_cli_builds_100_without_claiming_clearance(tmp_path):
    output = tmp_path/'preview'
    result = subprocess.run([sys.executable, str(ROOT/'tools/eval_mujoco_salvo.py'),
        '--dry-run', '--output', str(output)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    report = json.loads((output/'salvo_report.json').read_text())
    assert report['mode'] == 'dry_run' and not report['completed'] and not report['cleared']
    assert report['remaining'] == 100 and len(report['initial_boxes']) == 100
    assert len(load_layout(output/'layout.json', 100, .35)[0]) == 100


def test_cli_does_not_overwrite_previous_run(tmp_path):
    with pytest.raises(FileExistsError):
        main(['--dry-run', '--output', str(tmp_path)])


def test_migrated_model_path_uses_current_workspace():
    assert project_path('/home/jby/IsaacLab-2.3.2/runs/example.pth') == ROOT.parents[3]/'runs/example.pth'


@pytest.fixture
def model_config(tmp_path, monkeypatch):
    from integrated_shooting_chain.mujoco_sim import salvo
    code, assets = tmp_path/'system_project', tmp_path/'data_project'
    code.mkdir()
    assets.mkdir()
    monkeypatch.setattr(salvo, 'WORKSPACE', code)
    config = dict(model_root=str(assets), decision_checkpoint='runs/old/nn/policy.pt',
        velocity_checkpoint='models/velocity.pt', ostrack_checkpoint='models/ostrack.pt',
        ostrack_repo='tracker', locomotion_policy='models/walk.pt')
    for key in ('decision_checkpoint', 'velocity_checkpoint', 'ostrack_checkpoint', 'locomotion_policy'):
        path = assets/config[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (assets/'tracker').mkdir()
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    args = SimpleNamespace(config=path, decision_checkpoint=None, locomotion_policy=None,
                           model_root=None, dry_run=False)
    return salvo, args, code, assets, config


def test_all_models_use_configured_root_and_results_keep_project_root(model_config):
    salvo, args, code, assets, config = model_config
    actual = salvo.prepare_config(args)
    for key in ('decision_checkpoint', 'velocity_checkpoint', 'ostrack_checkpoint', 'ostrack_repo', 'locomotion_policy'):
        assert actual[key] == str(assets/config[key])
    assert salvo.project_path('outputs/result') == code/'outputs/result'


def test_paired_velocity_absolute_system_path_rebases_to_data(model_config):
    salvo, args, code, assets, config = model_config
    decision = assets/'runs/new/nn/policy.pt'
    decision.parent.mkdir(parents=True)
    decision.touch()
    velocity = assets/'models/paired.pt'
    velocity.touch()
    (decision.parent.parent/'training_settings.json').write_text(json.dumps({
        'arguments': {'velocity_model': str(code/'models/paired.pt')}}))
    args.decision_checkpoint = Path('runs/new/nn/policy.pt')
    actual = salvo.prepare_config(args)
    assert actual['decision_checkpoint'] == str(decision)
    assert actual['velocity_checkpoint'] == str(velocity)
    assert actual['training_settings'] == str(decision.parent.parent/'training_settings.json')


def test_missing_data_model_does_not_use_system_copy(model_config):
    salvo, args, code, assets, config = model_config
    relative = config['velocity_checkpoint']
    (assets/relative).unlink()
    fallback = code/relative
    fallback.parent.mkdir(parents=True)
    fallback.touch()
    with pytest.raises(FileNotFoundError, match='velocity_checkpoint'):
        salvo.prepare_config(args)


def test_model_root_override_and_locomotion_override(model_config, tmp_path):
    salvo, args, code, assets, config = model_config
    override = tmp_path/'alternate'
    import shutil
    shutil.copytree(assets, override)
    (override/'models/custom_walk.pt').touch()
    args.model_root = override
    args.locomotion_policy = Path('models/custom_walk.pt')
    actual = salvo.prepare_config(args)
    assert actual['model_root'] == str(override)
    assert actual['decision_checkpoint'] == str(override/config['decision_checkpoint'])
    assert actual['locomotion_policy'] == str(override/'models/custom_walk.pt')
