import os, sys

current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

import numpy as np
import torch

from model import ReCoNet
from utils import preprocess_for_reconet, postprocess_reconet, Dummy, nhwc_to_nchw, nchw_to_nhwc

sys.path.remove(current_dir)


class ReCoNetModel:

    def __init__(self, state_dict_path, use_gpu=True, gpu_device=None, frn=False):
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device

        with self.device():
            self.model = ReCoNet(frn=frn)
            self.model.load_state_dict(torch.load(state_dict_path))
            self.model = self.to_device(self.model)
            self.model.eval()

    def run(self, images):
        assert images.dtype == np.uint8
        assert 3 <= images.ndim <= 4

        orig_ndim = images.ndim
        if images.ndim == 3:
            images = images[None, ...]

        images = torch.from_numpy(images)
        images = nhwc_to_nchw(images)
        images = images.to(torch.float32) / 255

        with self.device():
            with torch.no_grad():
                images = self.to_device(images)
                images = preprocess_for_reconet(images)
                styled_images = self.model(images)
                styled_images = postprocess_reconet(styled_images)
                styled_images = styled_images.cpu()
                styled_images = torch.clamp(styled_images * 255, 0, 255).to(torch.uint8)
                styled_images = nchw_to_nhwc(styled_images)
                styled_images = styled_images.numpy()
                if orig_ndim == 3:
                    styled_images = styled_images[0]
                return styled_images

    def to_device(self, x):
        if self.use_gpu:
            with self.device():
                return x.cuda()
        else:
            return x

    def device(self):
        if self.use_gpu and self.gpu_device is not None:
            return torch.cuda.device(self.gpu_device)
        else:
            return Dummy()
