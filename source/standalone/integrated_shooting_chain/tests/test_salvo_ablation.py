from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from integrated_shooting_chain.salvo_ablation import (
    SalvoAblation, SalvoAblationConfig, install_actor_switch_reset,
    install_salvo_ablation, load_layout, save_layout,
)


def env_fixture():
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=.02,
                          episode_length_buf=torch.tensor([200, 300]),
                          _static_salvo=SimpleNamespace(active=torch.tensor([0, 5])),
                          _rl_pred_ready=torch.tensor([False, True]),
                          _rl_bridge_training_ready=torch.tensor([False, True]),
                          _rl_bridge_velocity_ready=torch.tensor([False, True]),
                          _aim_warmup_left_steps=torch.tensor([0, 0]),
                          _turret_pos_targets=torch.tensor([[.1, .2], [.3, .4]]),
                          _delta_pos=torch.zeros(2, 2))
    def pre(actions):
        env._actions = actions.clone()
        env._delta_pos = actions[:, :2]*.03
        env._turret_pos_targets += env._delta_pos
    env._pre_physics_step = pre
    env._reset_idx = lambda ids: None
    return env


@pytest.mark.parametrize("group,hold,reset", [("A",False,False),("B",True,False),("C",False,True),("D",True,True)])
def test_group_mapping_and_hold_preserves_command_and_proposed_action(group, hold, reset):
    env=env_fixture(); old=env._turret_pos_targets.clone();actions=torch.ones(2,3)
    exp=install_salvo_ablation(env,group)
    env._pre_physics_step(actions)
    assert exp.config.reset_actor_on_switch == reset
    assert torch.allclose(env._turret_pos_targets[0], old[0] if hold else old[0]+.03)
    assert torch.allclose(env._turret_pos_targets[1], old[1]+.03)
    assert torch.equal(actions, torch.ones(2,3))
    assert torch.equal(env._actions,actions)
    assert exp.hold_steps == ([1,0] if hold else [0,0])
    assert torch.allclose(exp.proposed_delta,torch.full((2,2),.03))
    assert torch.allclose(env._delta_pos[0],torch.zeros(2) if hold else torch.full((2,),.03))


def test_hold_exits_when_ready_and_ignores_warmup():
    env=env_fixture();exp=install_salvo_ablation(env,"B")
    env._aim_warmup_left_steps[0]=1
    env._pre_physics_step(torch.ones(2,3));assert not exp.hold_applied.any()
    env._aim_warmup_left_steps[0]=0
    env._pre_physics_step(torch.ones(2,3));assert exp.hold_applied.tolist()==[True,False]
    env._rl_pred_ready[:]=True;env._rl_bridge_training_ready[:]=True
    env._pre_physics_step(torch.ones(2,3));assert not exp.hold_applied.any()


@pytest.mark.parametrize("group",list("ABCD"))
def test_actor_reset_only_changed_env_before_inference_and_never_critic(group):
    env=env_fixture();exp=install_salvo_ablation(env,group,wait_for_velocity=False)
    actor=torch.ones(1,2,4);critic=torch.full((1,2,4),7.)
    seen=[]
    player=SimpleNamespace(states=[actor,critic],get_action=lambda obs:seen.append(actor.clone()))
    install_actor_switch_reset(player,exp)
    player.get_action(None)  # First acquisition preserves shared warmup history.
    assert torch.equal(actor,torch.ones_like(actor))
    env._static_salvo.active[0]=-1;player.get_action(None)
    env._static_salvo.active[0]=9;player.get_action(None)
    reset=group in "CD"
    assert seen[-1][0,0].tolist()==([0.]*4 if reset else [1.]*4)
    assert seen[-1][0,1].tolist()==[1.]*4
    assert torch.equal(critic,torch.full_like(critic,7.))
    assert exp.actor_resets==([1,0] if reset else [0,0])
    actor.fill_(3.);player.get_action(None)  # Same ID must not reset again.
    assert torch.equal(actor,torch.full_like(actor,3.))
    env._reset_idx(torch.tensor([0]));player.get_action(None)
    assert torch.equal(actor,torch.full_like(actor,3.))


@pytest.mark.parametrize("dictionary_obs", [False, True])
def test_default_d_skips_unready_actor_with_real_gru_and_resumes_selected_env(dictionary_obs):
    # NaN observations must never reach either network branch while waiting.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"test7_2_weight"))
    from asym_central_value_gru_network_gt import AsymDualGRUNet
    env=env_fixture();exp=install_salvo_ablation(env)
    model=AsymDualGRUNet(actions_num=3).eval()
    calls=[]
    class Player:
        has_batch_dimension=True
        states=[torch.ones(1,2,128),torch.full((1,2,128),7.)]
        def get_action(self, obs, is_deterministic=False):
            obs=obs['obs'] if isinstance(obs,dict) else obs
            calls.append((obs.clone(),self.states[0].clone(),is_deterministic))
            with torch.no_grad():
                mu,_,_,states=model(dict(obs=obs,rnn_states=self.states))
            self.states=states
            return mu.tanh()
    player=Player();install_actor_switch_reset(player,exp)
    observation=torch.zeros(2,59);observation[0]=float('nan')
    wrap=lambda: {'obs':observation} if dictionary_obs else observation
    actions=player.get_action(wrap(),is_deterministic=True)
    assert len(calls)==1 and calls[0][0].shape==(1,59) and calls[0][2] is True
    assert actions[0].tolist()==[0.,0.,-1.] and torch.isfinite(actions).all()
    assert torch.equal(player.states[0][:,0],torch.ones(1,128))
    assert torch.equal(player.states[1][:,0],torch.full((1,128),7.))

    # New identity resets only Actor once, before waiting; zero state stays zero.
    env._static_salvo.active[0]=90
    player.get_action(wrap());assert exp.actor_resets==[1,0]
    assert torch.count_nonzero(player.states[0][:,0])==0
    env._rl_pred_ready[1]=env._rl_bridge_training_ready[1]=env._rl_bridge_velocity_ready[1]=False
    saved=[s.clone() for s in player.states];call_count=len(calls)
    assert player.get_action(wrap()).tolist()==[[0.,0.,-1.],[0.,0.,-1.]]
    assert len(calls)==call_count
    assert all(torch.equal(s,old) for s,old in zip(player.states,saved))
    assert exp.actor_resets==[1,0]

    # Position readiness alone must not release Actor while velocity is invalid.
    env._rl_pred_ready[0]=env._rl_bridge_training_ready[0]=True
    player.get_action(wrap());assert len(calls)==call_count
    observation[0]=0;env._rl_bridge_velocity_ready[0]=True
    player.get_action(wrap())
    assert len(calls)==call_count+1 and torch.count_nonzero(calls[-1][1])==0
    assert exp.actor_inference_skipped.tolist()==[False,True]
    assert exp.actor_resets==[1,0]
    assert torch.equal(player.states[1][:,1],saved[1][:,1])

    # Loss of perception freezes the latest state instead of resetting it again.
    saved=[s.clone() for s in player.states]
    env._rl_bridge_velocity_ready[0]=False
    player.get_action(wrap())
    assert all(torch.equal(s,old) for s,old in zip(player.states,saved))
    assert exp.actor_resets==[1,0]
    summary=exp.summary(0)
    assert summary['wait_for_velocity'] and summary['actor_wait_steps']==5
    assert summary['actor_wait_sim_s']==pytest.approx(.1)
    env._reset_idx(torch.tensor([0]))
    assert exp.actor_wait_steps[0]==0 and not exp.actor_inference_skipped[0]


@pytest.mark.parametrize("batched", [False, True])
def test_waiting_single_env_preserves_warmup_and_action_shape(batched):
    env=env_fixture();env.num_envs=1
    for name in ('_rl_pred_ready','_rl_bridge_training_ready','_rl_bridge_velocity_ready',
                 '_aim_warmup_left_steps','_turret_pos_targets','_delta_pos','episode_length_buf'):
        setattr(env,name,getattr(env,name)[:1])
    env._static_salvo.active=env._static_salvo.active[:1]
    exp=install_salvo_ablation(env)
    calls=[]
    def infer(obs):
        calls.append(obs)
        player.states[0].add_(1)
        return torch.ones(1,3) if batched else torch.ones(3)
    player=SimpleNamespace(states=[torch.ones(1,1,4),torch.ones(1,1,4)],
                           has_batch_dimension=batched,get_action=infer)
    install_actor_switch_reset(player,exp)
    env._aim_warmup_left_steps[0]=1
    player.get_action(torch.zeros(1,59) if batched else torch.zeros(59))
    assert len(calls)==1 and torch.equal(player.states[0],torch.full((1,1,4),2.))
    env._aim_warmup_left_steps[0]=0
    idle=player.get_action(None)
    assert idle.shape==((1,3) if batched else (3,))
    assert idle.reshape(-1).tolist()==[0.,0.,-1.] and len(calls)==1
    env._pre_physics_step(idle.reshape(1,3))
    assert exp.hold_applied.item() and env._actions[0,2]==-1
    assert exp.proposed_delta[0].tolist()==[0.,0.]


def test_partial_inference_failure_restores_full_player_state():
    env=env_fixture();exp=install_salvo_ablation(env)
    states=[torch.ones(1,2,4),torch.full((1,2,4),7.)]
    def fail(obs):
        player.states[0].zero_()
        raise RuntimeError('forward failed')
    player=SimpleNamespace(states=states,has_batch_dimension=True,get_action=fail)
    install_actor_switch_reset(player,exp)
    with pytest.raises(RuntimeError,match='forward failed'):
        player.get_action(torch.zeros(2,59))
    assert player.states is states and player.has_batch_dimension
    assert torch.equal(states[0],torch.ones(1,2,4))
    assert torch.equal(states[1],torch.full((1,2,4),7.))


def test_layout_exact_roundtrip_checksum_and_reject_overwrite(tmp_path):
    p=tmp_path/"layout.json";points=torch.tensor([[80., 1., .35],[120., -3., 2.]])
    digest=save_layout(p,points,.35);got,d=load_layout(p,2,.35)
    assert torch.equal(got,points) and digest==d
    with pytest.raises(FileExistsError):save_layout(p,points,.35)
    doc=json.loads(p.read_text());doc['positions_env_local_m'][0][0]=90.;p.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="checksum"):load_layout(p,2,.35)


def test_layout_rejects_wrong_radius_count_and_invalid_coordinates(tmp_path):
    p=tmp_path/"layout.json";save_layout(p,torch.tensor([[80.,1.,1.]]),.35)
    with pytest.raises(ValueError,match="radius"):load_layout(p,1,.4)
    with pytest.raises(ValueError,match="count"):load_layout(p,2,.35)
    with pytest.raises(ValueError):SalvoAblationConfig("Z")


def test_campaign_commands_fix_models_seed_gpu_and_share_one_layout(tmp_path):
    import runpy
    campaign=runpy.run_path(str(Path(__file__).resolve().parents[1]/"tools/run_salvo_ablation.py"))
    commands={g:campaign["group_command"](g,tmp_path,gpu=1,seed=61) for g in "ABCD"}
    def option(command,key):return command[command.index(key)+1]
    for key in ("--checkpoint","--full_chain_velocity_checkpoint","--salvo_seed","--device","--salvo_timeout_s"):
        assert len({option(c,key) for c in commands.values()})==1
    assert option(commands['A'],'--salvo_layout_out')==str(tmp_path/'layout.json')
    for g in "BCD":assert option(commands[g],'--salvo_layout_in')==str(tmp_path/'layout.json')
    assert all('--record_video' not in c for c in commands.values())
    assert all('--no-salvo_wait_for_velocity' in c for c in commands.values())
    assert all(option(c,'--salvo_switch_policy')=='timeout' for c in commands.values())


def test_resume_copies_only_completed_groups_preserves_source_and_checks_conditions(tmp_path):
    import runpy
    campaign=runpy.run_path(str(Path(__file__).resolve().parents[1]/"tools/run_salvo_ablation.py"))
    source=tmp_path/'source';source.mkdir();(source/'A').mkdir();(source/'B').mkdir()
    (source/'status.json').write_text(json.dumps(dict(completed=['A'],state='interrupted')))
    commands={g:campaign['group_command'](g,source,gpu=1,seed=61) for g in 'ABCD'}
    (source/'commands.json').write_text(json.dumps(commands))
    (source/'layout.json').write_text(json.dumps(dict(sha256='same-layout')))
    old_report=dict(layout_sha256='same-layout',episodes=[dict(reason='timeout')],
                    diagnostics_csv=str(source/'A'/'salvo_report.steps.csv'))
    (source/'A'/'salvo_report.json').write_text(json.dumps(old_report))
    (source/'A'/'salvo_report.steps.csv').write_text('env_id\n0\n')
    (source/'B'/'run.log').write_text('interrupted log')
    output=tmp_path/'resumed';output.mkdir()
    completed,_=campaign['resume_completed'](source,output,gpu=1,timeout_s=30.,seed=61,video=False)
    assert completed==['A']
    assert not (output/'B').exists()
    assert (source/'B'/'run.log').read_text()=='interrupted log'
    assert json.loads((source/'A'/'salvo_report.json').read_text())==old_report
    assert json.loads((output/'A'/'salvo_report.json').read_text())['diagnostics_csv']==str(output/'A'/'salvo_report.steps.csv')
    with pytest.raises(ValueError,match='salvo_seed'):
        campaign['resume_completed'](source,tmp_path/'bad',gpu=1,timeout_s=30.,seed=62,video=False)
