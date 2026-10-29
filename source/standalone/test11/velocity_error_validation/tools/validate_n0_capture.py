"""Require actual complete acquisition; an exit code or marker alone is insufficient."""
import argparse
import json
from pathlib import Path
import torch


def validate(folder, log, envs, steps):
    if '[taq-collect] complete:' not in log.read_text(errors='replace'):
        raise ValueError('Collector did not write its final completion record')
    streams=list(folder.rglob('timed_sensor_packets.pt'))
    frames=0;seen_envs=set()
    for path in streams:
        payload=torch.load(path,map_location='cpu',weights_only=False)
        stream=torch.load(path.with_name('taq_velocity_episode.taqstream'),map_location='cpu',weights_only=False)
        packets=payload['packets'];n=len(packets)
        audit=path.with_name('velocity_error_audit.jsonl').read_text().splitlines()
        if not n or n!=len(audit) or n!=len(stream['timestamps_s']):
            raise ValueError('Packet / label / audit count mismatch')
        times=torch.tensor([p['capture_time_s'] for p in packets],dtype=torch.float64)
        torch.testing.assert_close(times,stream['timestamps_s'],atol=1e-10,rtol=0)
        if len({p['sample_id'] for p in packets})!=n:
            raise ValueError('Duplicate capture identity')
        frames+=n;seen_envs.add(path.parents[1].name)
    if len(seen_envs)!=envs or frames<envs*(steps-4):
        raise ValueError(f'Incomplete acquisition: envs={len(seen_envs)}, frames={frames}, expected >= {envs*(steps-4)}')
    result=dict(environments=len(seen_envs),frames=frames,streams=len(streams),requested_steps=steps)
    (folder/'capture_validation.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--log',type=Path,required=True)
    parser.add_argument('--envs',type=int,required=True)
    parser.add_argument('--steps',type=int,required=True)
    args=parser.parse_args()
    print(validate(args.folder,args.log,args.envs,args.steps))
