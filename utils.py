from collections import defaultdict

import torch


def tensors_sum(tensors):
    result = 0
    for tensor in tensors:
        result += tensor
    return result


def magnitude_squared(x):
    return x.pow(2).sum(-1)


def nhwc_to_nchw(x):
    return x.permute(0, 3, 1, 2)


def nchw_to_nhwc(x):
    return x.permute(0, 2, 3, 1)


def warp_optical_flow(source, reverse_flow):
    n, h, w, _ = reverse_flow.shape

    reverse_flow = reverse_flow.clone()
    reverse_flow[..., 0] += torch.arange(w).view(1, 1, w).cuda()
    reverse_flow[..., 0] *= 2 / w
    reverse_flow[..., 0] -= 1
    reverse_flow[..., 1] += torch.arange(h).view(1, h, 1).cuda()
    reverse_flow[..., 1] *= 2 / h
    reverse_flow[..., 1] -= 1

    return torch.nn.functional.grid_sample(source, reverse_flow, padding_mode='border')


def occlusion_mask_from_flow(optical_flow, reverse_optical_flow, motion_boundaries):
    # "Dense Point Trajectories by GPU-accelerated Large Displacement Optical Flow"
    # Page 7

    optical_flow = nhwc_to_nchw(optical_flow)
    optical_flow = warp_optical_flow(optical_flow, reverse_optical_flow)
    optical_flow = nchw_to_nhwc(optical_flow)

    forward_magnitude = magnitude_squared(optical_flow)
    reverse_magnitude = magnitude_squared(reverse_optical_flow)
    sum_magnitude = magnitude_squared(optical_flow + reverse_optical_flow)

    occlusion_mask = sum_magnitude < (0.01 * (forward_magnitude + reverse_magnitude) + 0.5)
    occlusion_mask &= ~motion_boundaries
    return occlusion_mask.to(torch.float32).unsqueeze_(1)


def rgb_to_luminance(x):
    return x[:, 0, ...] * 0.2126 + x[:, 1, ...] * 0.7512 + x[:, 2, ...] * 0.0722


def l2_squared(x):
    return x.pow(2).sum()


def mean_l2_squared(x):
    return x.pow(2).mean()


def resize_optical_flow(optical_flow, h, w):
    optical_flow_nchw = nhwc_to_nchw(optical_flow)
    optical_flow_resized_nchw = torch.nn.functional.interpolate(optical_flow_nchw, size=(h, w), mode='bilinear')
    optical_flow_resized = nchw_to_nhwc(optical_flow_resized_nchw)

    old_h, old_w = optical_flow_nchw.shape[-2:]
    h_scale, w_scale = h / old_h, w / old_w
    optical_flow_resized[..., 0] *= w_scale
    optical_flow_resized[..., 1] *= h_scale
    return optical_flow_resized


def gram_matrix(feature_map):
    n, c, h, w = feature_map.shape
    feature_map = feature_map.reshape((n, c, h * w))
    return feature_map.bmm(feature_map.transpose(1, 2)) / (c * h * w)


def normalize_batch(batch, mean, std):
    dtype = batch.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=batch.device)
    std = torch.as_tensor(std, dtype=dtype, device=batch.device)
    return (batch - mean[None, :, None, None]) / std[None, :, None, None]


def preprocess_for_vgg(images_batch):
    return normalize_batch(images_batch,
                           mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])


def preprocess_for_reconet(images_batch):
    images_batch = images_batch.clone()
    return images_batch * 2 - 1


def postprocess_reconet(images_batch):
    images_batch = images_batch.clone()
    return (images_batch + 1) / 2


class RunningLossesContainer:

    def __init__(self):
        self.values = defaultdict(lambda: 0)
        self.counters = defaultdict(lambda: 0)

    def update(self, losses):
        for key, value in losses.items():
            self.values[key] += value.item()
            self.counters[key] += 1

    def get(self):
        return {key: self.values[key] / self.counters[key] for key in self.values}

    def reset(self):
        self.values.clear()
        self.counters.clear()


class Dummy:

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattribute__(self, item):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return
