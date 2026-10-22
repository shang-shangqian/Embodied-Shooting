"""Load the original recurrent actor and OSTrack without starting Isaac Sim."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
import torch


def ensure_opencv():
    """Reuse an existing ABI3 OpenCV build without installing into an environment."""
    try:
        return importlib.import_module('cv2')
    except ModuleNotFoundError as exc:
        if exc.name != 'cv2':
            raise
    envs = Path(sys.prefix).parent
    for name in ('ostrack', 'sam2', 'yolo11_eval'):
        for root in sorted((envs/name/'lib').glob('python*/site-packages')):
            if not list((root/'cv2').glob('cv2.abi3.so')):
                continue
            sys.path.append(str(root))
            try:
                return importlib.import_module('cv2')
            finally:
                sys.path.remove(str(root))
    raise RuntimeError('OpenCV is required for LK/OSTrack; no compatible local ABI3 build was found.')


class RecurrentShootingPolicy:
    """Use the original RL-Games model, saved normalization, GRU and tanh mean.

    The unused 31-D critic branch receives zero placeholders. Its separate GRU
    cannot affect the actor; no target truth is supplied to either branch.
    """
    def __init__(self, checkpoint, *, device='cpu'):
        from rl_games.algos_torch.models import ModelA2CContinuousTanh
        standalone = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(standalone/'test7_2_weight'))
        from asym_central_value_gru_network_gt import AsymDualGRUNet
        self.device = torch.device(device)
        self.model = ModelA2CContinuousTanh.Network(
            AsymDualGRUNet(actions_num=3), obs_shape=(59,),
            normalize_value=True, normalize_input=True, value_size=1).to(self.device)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.model.load_state_dict(payload['model'], strict=True)
        self.model.eval().requires_grad_(False)
        self.epoch = payload.get('epoch')
        self.reset()

    def reset(self):
        self.states = [x.to(self.device) for x in self.model.get_default_rnn_state()]
        self.calls = 0

    @torch.inference_mode()
    def __call__(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (28,) or not np.isfinite(observation).all():
            raise ValueError('Actor requires a finite 28-D observation.')
        full = np.concatenate([np.clip(observation, -100., 100.), np.zeros(31, dtype=np.float32)])
        result = self.model({'is_train': False, 'prev_actions': None,
                             'obs': torch.as_tensor(full[None], device=self.device), 'rnn_states': self.states})
        self.states = result['rnn_states']
        self.calls += 1
        return torch.tanh(result['mus'])[0].cpu().numpy()


def make_device_ostrack(repo, checkpoint, *, config='vitb_256_mae_ce_32x4_ep300', device='cpu'):
    """Keep official initialize/track methods; make only construction device-aware.

    The repository's official constructor hardcodes .cuda(). This local adapter
    replaces its device allocations, without modifying the original tracker.
    """
    ensure_opencv()
    from test_lk_ostrack.ostrack_tracker import OSTrackFrameTracker
    adapter = OSTrackFrameTracker(repo, checkpoint, config=config, device=device)
    params = adapter.adapter._build_params()
    from lib.models.ostrack import build_ostrack
    from lib.test.tracker.ostrack import OSTrack
    from lib.test.tracker.basetracker import BaseTracker
    from lib.test.utils.hann import hann2d
    from lib.utils.misc import NestedTensor

    class DevicePreprocessor:
        def __init__(self):
            self.mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
            self.std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)

        def process(self, image, mask):
            tensor = torch.as_tensor(np.ascontiguousarray(image), device=device).float().permute(2, 0, 1)[None]
            tensor = (tensor/255. - self.mean)/self.std
            return NestedTensor(tensor, torch.as_tensor(mask, device=device, dtype=torch.bool)[None])

    network = build_ostrack(params.cfg, training=False)
    network.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['net'], strict=True)
    tracker = OSTrack.__new__(OSTrack)
    BaseTracker.__init__(tracker, params)
    tracker.cfg = params.cfg
    tracker.network = network.to(device).eval().requires_grad_(False)
    tracker.preprocessor = DevicePreprocessor()
    tracker.state = None
    tracker.feat_sz = tracker.cfg.TEST.SEARCH_SIZE // tracker.cfg.MODEL.BACKBONE.STRIDE
    tracker.output_window = hann2d(torch.tensor([tracker.feat_sz, tracker.feat_sz]).long(), centered=True).to(device)
    tracker.debug = tracker.use_visdom = 0
    tracker.frame_id = 0
    tracker.save_all_boxes = params.save_all_boxes
    tracker.z_dict1 = {}
    adapter._tracker = tracker
    return adapter
