"""Sampling and delivery contract, with no target labels or GT world poses.

Capture time uses a monotonic reference clock. Device-reported time is separate.
Transport delay is applied after acquisition, never by changing physical dt.
"""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import math
import numpy as np

TIME_CONTRACT_VERSION = 'physical_capture_v1'


class EpisodeCollectionWarmup:
    """Exclude startup captures using each environment's resettable step count.

    Physics and tracking still run during warmup. The first saved capture is
    strictly after the requested interval; this also handles fractional steps.
    """
    def __init__(self, warmup_s: float, step_dt_s: float):
        if not math.isfinite(warmup_s) or warmup_s < 0:
            raise ValueError('Episode collection warmup must be finite and non-negative')
        if not math.isfinite(step_dt_s) or step_dt_s <= 0:
            raise ValueError('Physics step duration must be finite and positive')
        self.steps = int(math.ceil(warmup_s / step_dt_s - 1e-9))
        self.effective_s = self.steps * step_dt_s

    def ready(self, episode_step: int) -> bool:
        return episode_step > self.steps


class PhysicalSamplingGate:
    """Skip camera consumption while physics and proprioception keep advancing."""
    def __init__(self, probability, seed):
        if not 0 <= probability < 1:
            raise ValueError('Invalid sampling skip probability')
        self.probability = probability
        self.rng = np.random.default_rng(seed)
        self.skip_until = -1

    def take(self, step):
        if step == 0:
            return True
        if step >= self.skip_until and self.rng.random() < self.probability:
            self.skip_until = step+int(self.rng.integers(2, 7))
        return step >= self.skip_until


def quaternion_rotation(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def yaw_rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def capture_packet(*, sample_id, sample_time_s, yaw_pitch, range_m, range_valid,
                   source_id, quality, linear, angular, camera_position_world,
                   camera_quaternion_world, base_position_world,
                   base_quaternion_world, tracker_reset):
    """World robot poses cancel to a camera-to-yaw-body sensor extrinsic.

Returned values contain simulated sensor readings and extrinsic calibration,
never the target position or absolute robot position.
"""
    root = quaternion_rotation(base_quaternion_world)
    unyaw = yaw_rotation(-math.atan2(root[1, 0], root[0, 0]))
    rotation = unyaw @ quaternion_rotation(camera_quaternion_world)
    origin = unyaw @ (np.asarray(camera_position_world)-np.asarray(base_position_world))
    valid = bool(range_valid) and math.isfinite(float(range_m)) and range_m > 0
    return dict(version=TIME_CONTRACT_VERSION, sample_id=int(sample_id),
                capture_time_s=float(sample_time_s), reported_clock_time_s=float(sample_time_s),
                direction_sample_time_s=float(sample_time_s), range_sample_time_s=float(sample_time_s),
                proprio_sample_time_s=float(sample_time_s), tracker_result_available_time_s=float(sample_time_s),
                direction_yaw_pitch=[float(v) for v in yaw_pitch],
                camera_rotation_base=rotation.tolist(), camera_origin_base=origin.tolist(),
                range_m=float(range_m) if valid else None, range_valid=valid,
                source_id=int(source_id), quality=[float(v) for v in quality],
                linear_velocity_base_mps=[float(v) for v in linear],
                angular_velocity_base_rps=[float(v) for v in angular], tracker_reset=bool(tracker_reset))


def position_from_pair(direction, distance):
    if direction['sample_id'] != distance['sample_id']:
        raise ValueError('Unmatched direction/range sample identity')
    if abs(direction['capture_time_s']-distance['capture_time_s']) > 1e-9:
        raise ValueError('Unmatched direction/range sampling time')
    if not distance['range_valid'] or direction['source_id'] == 4:
        return None
    yaw, pitch = direction['direction_yaw_pitch']
    rho = distance['range_m']
    camera = rho*np.array([np.sin(yaw)*np.cos(pitch), -np.sin(pitch), np.cos(yaw)*np.cos(pitch)])
    return np.asarray(direction['camera_rotation_base'])@camera + direction['camera_origin_base']


@dataclass(frozen=True)
class Delivery:
    kind: str
    sample_id: int
    sample_time_s: float
    arrival_time_s: float
    value: dict


class DeliveryQueue:
    def __init__(self):
        self.heap = []
        self.identities = set()
        self.counter = 0

    def push(self, packet):
        if packet.kind not in ('direction', 'range', 'proprio'):
            raise ValueError('Unknown sensor kind')
        if not (math.isfinite(packet.sample_time_s) and math.isfinite(packet.arrival_time_s)):
            raise ValueError('Non-finite timestamp')
        if packet.arrival_time_s < packet.sample_time_s-1e-9:
            raise ValueError('Arrival precedes acquisition')
        if packet.value['sample_id'] != packet.sample_id or abs(packet.value['capture_time_s']-packet.sample_time_s)>1e-9:
            raise ValueError('Packet payload and delivery sampling metadata disagree')
        key = packet.kind, packet.sample_id
        if key in self.identities:
            raise ValueError('Duplicate sensor packet')
        self.identities.add(key)
        heapq.heappush(self.heap, (packet.arrival_time_s, self.counter, packet))
        self.counter += 1

    def arrived(self, now):
        ready = []
        while self.heap and self.heap[0][0] <= now+1e-9:
            ready.append(heapq.heappop(self.heap)[2])
        return ready


class MatchedMeasurements:
    def __init__(self):
        self.pending = {}
        self.consumed = set()

    def receive(self, delivery):
        if delivery.kind == 'proprio':
            return None
        key = delivery.sample_id
        if key in self.consumed:
            raise ValueError('Measurement already consumed')
        parts = self.pending.setdefault(key, {})
        if delivery.kind in parts:
            raise ValueError('Duplicate component')
        parts[delivery.kind] = delivery
        if 'direction' not in parts or 'range' not in parts:
            return None
        direction, distance = parts['direction'], parts['range']
        if abs(direction.sample_time_s-distance.sample_time_s) > 1e-9:
            raise ValueError('Sampling time mismatch')
        position = position_from_pair(direction.value, distance.value)
        self.consumed.add(key)
        del self.pending[key]
        return dict(sample_id=key, sample_time_s=direction.sample_time_s,
                    position_base_m=position, direction=direction.value)

    def expire(self, oldest_sample_time_s):
        expired = [key for key, parts in self.pending.items()
                   if min(v.sample_time_s for v in parts.values()) < oldest_sample_time_s-1e-9]
        for key in expired:
            del self.pending[key]
        return len(expired)
