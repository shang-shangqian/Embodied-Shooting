"""Audit saved walking trajectories for body contacts and world heading drift."""
import argparse
from collections import Counter
import json
from pathlib import Path

from play_mujoco_lite3 import mujoco, np, resolve_model_path


def audit(folder):
    report = json.loads((folder / 'report.json').read_text())
    trace = np.load(folder / 'trajectory.npz')
    model = mujoco.MjModel.from_xml_path(str(resolve_model_path(report['model'])))
    override = report.get('contact_time_constant_override_s')
    if override is not None:
        active = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        model.geom_solref[active, 0] = override
    data = mujoco.MjData(model)
    contacts = Counter()
    deepest = 0.
    penetrations = []
    contact_geoms = Counter()
    sampled = 0
    yaws = []
    indices = np.arange(int(round(2/model.opt.timestep)), len(trace['physics_time']), 8)
    if not len(indices):
        indices = np.array([len(trace['physics_time'])-1])
    for index in indices:
        data.qpos[:] = trace['physics_qpos'][index]
        data.qvel[:] = trace['physics_qvel'][index]
        mujoco.mj_forward(model, data)
        touched = set()
        for contact in data.contact:
            if contact.dist > 0:
                continue
            deepest = min(deepest, float(contact.dist))
            penetrations.append(-float(contact.dist))
            for geom in contact.geom:
                body = int(model.geom_bodyid[geom])
                if body:
                    touched.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body))
                    contact_geoms[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom))] += 1
        contacts.update(touched)
        sampled += 1
        rotation = data.body('TORSO').xmat.reshape(3, 3)
        yaws.append(float(np.arctan2(rotation[1, 0], rotation[0, 0])))
    heading_error = None
    if report.get('heading_target_rad') is not None:
        errors = np.arctan2(np.sin(np.asarray(yaws)-report['heading_target_rad']),
                            np.cos(np.asarray(yaws)-report['heading_target_rad']))
        heading_error = float(np.rad2deg(np.abs(errors).max()))
    result = {'sampled_frames_after_2s': sampled, 'sampling_period_s': 8*model.opt.timestep,
              'contact_frame_fraction_by_body': {k: v/sampled for k, v in sorted(contacts.items())},
              'deepest_contact_distance_m': deepest, 'final_yaw_deg': float(np.rad2deg(yaws[-1])),
              'mean_contact_penetration_m': float(np.mean(penetrations)) if penetrations else 0.,
              'p95_contact_penetration_m': float(np.percentile(penetrations, 95)) if penetrations else 0.,
              'contact_point_counts_by_geom': dict(sorted(contact_geoms.items())),
              'maximum_heading_target_error_after_2s_deg': heading_error,
              'scope': 'Geometry/contact replay at sampled states, not a force or friction-equivalence validation.'}
    (folder / 'contact_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'run': str(folder), **result}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', type=Path, nargs='+')
    for path in parser.parse_args().folders:
        audit(path.resolve())
