"""Physical CV state output contract; no labels, world poses or future inputs.

Local state order is [px, py, pz, vx, vy, vz]. Velocities in this fixed local
frame are target absolute velocities. Body outputs use the current yaw-only
body frame and subtract the estimated base translation/linear velocity.
Covariances are conditional on the supplied base pose/velocity and clock.
"""
from __future__ import annotations
import numpy as np

STATE_ORDER = ('px', 'py', 'pz', 'vx', 'vy', 'vz')
FRAME = 'local_inertial_yaw_odometry'


def predict_cv_state(mean, covariance, process_intensity, elapsed_s):
    """Constant-velocity propagation, including all position/velocity cross terms.

The current acceleration spectral intensity is held constant for this forecast;
no future measurement or future noise-network output is used.
"""
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    q = np.asarray(process_intensity, dtype=np.float64)
    h = float(elapsed_s)
    if mean.shape != (6,) or covariance.shape != (6, 6) or q.shape != (3,):
        raise ValueError('Expected six-state mean, 6x6 covariance and three Q intensities')
    if not np.isfinite(h) or h < 0 or not all(np.isfinite(a).all() for a in (mean, covariance, q)) or (q < 0).any():
        raise ValueError('Nonfinite state or negative forecast interval/process intensity')
    # Accept float32 Joseph-update roundoff; never hide a material asymmetry.
    if not np.allclose(covariance, covariance.T, rtol=2e-6, atol=1e-9):
        raise ValueError('Covariance must be symmetric')
    covariance = .5*(covariance+covariance.T)
    if np.linalg.eigvalsh(covariance).min() < -1e-8:
        raise ValueError('Covariance must be positive semidefinite')
    f = np.eye(6); f[:3, 3:] = np.eye(3)*h
    a = np.diag(q)
    process = np.block([[a*h**3/3, a*h**2/2], [a*h**2/2, a*h]])
    predicted = f@covariance@f.T+process
    return f@mean, .5*(predicted+predicted.T)


def filter_state_output(mean, covariance, *, state_time_s, heading_local_rad,
                        base_position_local_m, base_velocity_base_mps,
                        process_intensity, initialized, reference_epoch_s,
                        reference_generation=0, output_time_s=None,
                        last_direct_measurement_time_s=None,
                        last_observation_local_m=None, direct_update=False):
    """Return a detached physical snapshot at output_time_s (defaults to state time).

When output time is later, first propagate the target state over the full gap.
    Base pose/heading/velocity must describe that output time. `posterior_time_s`
    identifies the supplied state; `state_time_s` identifies the returned state.
    A delayed observation's acquisition time is separate from both fields.
"""
    posterior_time = float(state_time_s)
    output_time = posterior_time if output_time_s is None else float(output_time_s)
    epoch = float(reference_epoch_s)
    heading = float(heading_local_rad)
    if not all(np.isfinite(v) for v in (posterior_time, output_time, epoch, heading)) or epoch > posterior_time+1e-9:
        raise ValueError('Invalid state/output/reference clock')
    mean, cov = predict_cv_state(mean, covariance, process_intensity, output_time-posterior_time)
    base = np.asarray(base_position_local_m, dtype=float)
    robot = np.asarray(base_velocity_base_mps, dtype=float)
    if base.shape != (3,) or robot.shape != (3,) or not np.isfinite(base).all() or not np.isfinite(robot).all():
        raise ValueError('Expected finite base position and velocity')
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    transform = np.zeros((6, 6)); transform[:3, :3] = transform[3:, 3:] = rot.T
    body_cov = transform@cov@transform.T
    body_mean = np.r_[rot.T@(mean[:3]-base), rot.T@mean[3:]-robot]
    measurement_time = None if last_direct_measurement_time_s is None else float(last_direct_measurement_time_s)
    if measurement_time is not None and (not np.isfinite(measurement_time) or measurement_time > posterior_time+1e-9 or measurement_time < epoch-1e-9):
        raise ValueError('Direct observation acquisition time is outside causal history')
    if last_observation_local_m is not None:
        observation = np.asarray(last_observation_local_m, dtype=float)
        if observation.shape != (3,) or not np.isfinite(observation).all() or measurement_time is None:
            raise ValueError('Raw observation requires a finite position and acquisition time')
        observation = observation.copy()
    else:
        observation = None
    return dict(
        contract_version='cv_state_output_v1', state_order=STATE_ORDER,
        state_time_s=output_time, posterior_time_s=posterior_time,
        output_time_s=output_time, reference_epoch_s=epoch,
        reference_generation=int(reference_generation), reference_frame=FRAME,
        reference_origin='base_position_at_episode_start',
        position_local_m=mean[:3].copy(), velocity_local_mps=mean[3:].copy(),
        state_local=mean.copy(), covariance_local=cov.copy(),
        position_base_m=body_mean[:3].copy(), velocity_base_mps=body_mean[3:].copy(),
        state_base=body_mean.copy(), covariance_base=body_cov.copy(),
        covariance_position_local=cov[:3, :3].copy(),
        covariance_position_velocity_local=cov[:3, 3:].copy(),
        covariance_velocity_local=cov[3:, 3:].copy(),
        sigma_base_mps=np.sqrt(np.maximum(np.diag(body_cov)[3:], 0)),
        base_position_local_m=base.copy(), base_velocity_base_mps=robot.copy(),
        heading_local_rad=heading, process_intensity_local=np.asarray(process_intensity, dtype=float).copy(),
        last_direct_measurement_time_s=measurement_time,
        measurement_age_s=None if measurement_time is None else output_time-measurement_time,
        last_observation_local_m=observation,
        initialized=bool(initialized), position_available=bool(initialized),
        forecast_valid=bool(initialized), direct_update=bool(direct_update),
        status='uninitialized' if not initialized else ('updated' if direct_update else 'predicted'),
        covariance_scope='conditional_target_model; base_pose_velocity_clock_uncertainty_excluded',
    )


def forecast_local(estimate, horizon_s, *, allow_uninitialized=False):
    """Forecast from the published state's time, in its unchanged local frame.

An uninitialized state has no acquired target position. Explicit prior requests
are useful for scoring, but do not convert that prior into a valid localization.
"""
    if not estimate['initialized'] and not allow_uninitialized:
        raise ValueError('Target position is uninitialized; explicitly request a prior for diagnostic scoring')
    mean, cov = predict_cv_state(estimate['state_local'], estimate['covariance_local'],
                                 estimate['process_intensity_local'], horizon_s)
    return dict(prediction_time_s=estimate['state_time_s']+float(horizon_s),
                initialized=estimate['initialized'],valid=estimate['initialized'],
                reference_frame=estimate['reference_frame'],
                reference_epoch_s=estimate['reference_epoch_s'],
                reference_generation=estimate['reference_generation'],
                position_local_m=mean[:3], velocity_local_mps=mean[3:],
                covariance_local=cov, covariance_position_local=cov[:3, :3].copy())
