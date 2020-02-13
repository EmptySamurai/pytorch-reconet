import random

from torchvision import transforms
import torch
import cv2
import numpy as np
from PIL import Image


class ToTensor:
    def __call__(self, sample):
        return {
            "frame": self.image_to_tensor(sample["frame"]),
            "previous_frame": self.image_to_tensor(sample["previous_frame"]),
            "optical_flow": torch.from_numpy(sample["optical_flow"]),
            "reverse_optical_flow": torch.from_numpy(sample["reverse_optical_flow"]),
            "motion_boundaries": torch.from_numpy(np.array(sample["motion_boundaries"]).astype(np.bool)),
            "index": sample["index"]
        }

    @staticmethod
    def image_to_tensor(image):
        return transforms.ToTensor()(image)


class Resize:

    def __init__(self, new_width, new_height):
        self.new_width = new_width
        self.new_height = new_height

    def resize_image(self, image):
        return image.resize((self.new_width, self.new_height))

    def resize_optical_flow(self, optical_flow):
        orig_height, orig_width = optical_flow.shape[:2]
        optical_flow_resized = cv2.resize(optical_flow, (self.new_width, self.new_height))
        h_scale, w_scale = self.new_height / orig_height, self.new_width / orig_width
        optical_flow_resized[..., 0] *= w_scale
        optical_flow_resized[..., 1] *= h_scale
        return optical_flow_resized

    def __call__(self, sample):
        return {
            "frame": self.resize_image(sample["frame"]),
            "previous_frame": self.resize_image(sample["previous_frame"]),
            "optical_flow": self.resize_optical_flow(sample["optical_flow"]),
            "reverse_optical_flow": self.resize_optical_flow(sample["reverse_optical_flow"]),
            "motion_boundaries": self.resize_image(sample["motion_boundaries"]),
            "index": sample["index"]
        }


class RandomHorizontalFlip:

    def __init__(self, p=0.5):
        self.p = p

    @staticmethod
    def flip_image(image):
        return image.transpose(Image.FLIP_LEFT_RIGHT)

    @staticmethod
    def flip_optical_flow(optical_flow):
        optical_flow = np.flip(optical_flow, axis=1).copy()
        optical_flow[..., 0] *= -1
        return optical_flow

    def __call__(self, sample):
        if random.random() < self.p:
            return {
                "frame": self.flip_image(sample["frame"]),
                "previous_frame": self.flip_image(sample["previous_frame"]),
                "optical_flow": self.flip_optical_flow(sample["optical_flow"]),
                "reverse_optical_flow": self.flip_optical_flow(sample["reverse_optical_flow"]),
                "motion_boundaries": self.flip_image(sample["motion_boundaries"]),
                "index": sample["index"]
            }
        else:
            return sample
