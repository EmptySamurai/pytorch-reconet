import os
from collections import namedtuple
from abc import abstractmethod

from torch.utils.data import Dataset
from PIL import Image

from IO import readFlow

SceneFlowEntry = namedtuple("SceneFlowEntry",
                            ("frame", "previous_frame", "optical_flow", "reverse_optical_flow", "motion_boundaries"))


class SceneFlowDataset(Dataset):
    def __init__(self, root_dir, transform, use_both_sides=False):
        self.entries = list(self.iterate_entries(root_dir, use_both_sides))
        self.transform = transform

    @abstractmethod
    def iterate_entries(self, root_dir, use_both_sides):
        pass

    def __getitem__(self, index):
        entry = self.entries[index]
        sample = {
            "frame": Image.open(entry.frame).convert("RGB"),
            "previous_frame": Image.open(entry.previous_frame).convert("RGB"),
            "optical_flow": readFlow(entry.optical_flow).copy(),
            "reverse_optical_flow": readFlow(entry.reverse_optical_flow).copy(),
            "motion_boundaries": Image.open(entry.motion_boundaries),
            "index": index
        }

        sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.entries)

    @staticmethod
    def frame_number(filename):
        return os.path.splitext(filename)[0]

    @staticmethod
    def side_letter(side):
        return side[0].upper()


class MonkaaDataset(SceneFlowDataset):
    def iterate_entries(self, root_dir, use_both_sides):
        for dirpath, dirnames, filenames in os.walk(os.path.join(root_dir, "frames_finalpass")):
            if len(filenames) == 0:
                continue

            scene, side = dirpath.split(os.sep)[-2:]

            if not use_both_sides and side != "left":
                continue

            filenames.sort()
            filenames = [filename for filename in filenames if filename.endswith(".png")]

            for i in range(1, len(filenames)):
                yield SceneFlowEntry(
                    os.path.join(dirpath, filenames[i]),
                    os.path.join(dirpath, filenames[i - 1]),
                    self.forward_optical_flow_path(root_dir, scene, side, self.frame_number(filenames[i - 1])),
                    self.reverse_optical_flow_path(root_dir, scene, side, self.frame_number(filenames[i])),
                    self.motion_boundaries_path(root_dir, scene, side, self.frame_number(filenames[i]))
                )

    def forward_optical_flow_path(self, root, scene, side, frame_number):
        return os.path.join(root, "optical_flow", scene, "into_future", side,
                            f"OpticalFlowIntoFuture_{frame_number}_{self.side_letter(side)}.pfm")

    def reverse_optical_flow_path(self, root, scene, side, frame_number):
        return os.path.join(root, "optical_flow", scene, "into_past", side,
                            f"OpticalFlowIntoPast_{frame_number}_{self.side_letter(side)}.pfm")

    def motion_boundaries_path(self, root, scene, side, frame_number):
        return os.path.join(root, "motion_boundaries", scene, "into_past", side,
                            f"{frame_number}.pgm")


class FlyingThings3DDataset(SceneFlowDataset):
    def iterate_entries(self, root_dir, use_both_sides):
        for dirpath, dirnames, filenames in os.walk(os.path.join(root_dir, "frames_finalpass")):
            if len(filenames) == 0:
                continue

            part, subset, scene, side = dirpath.split(os.sep)[-4:]

            if not use_both_sides and side != "left":
                continue

            filenames.sort()
            filenames = [filename for filename in filenames if filename.endswith(".png")]

            for i in range(1, len(filenames)):
                yield SceneFlowEntry(
                    os.path.join(dirpath, filenames[i]),
                    os.path.join(dirpath, filenames[i - 1]),
                    self.forward_optical_flow_path(root_dir, part, subset, scene, side,
                                                   self.frame_number(filenames[i - 1])),
                    self.reverse_optical_flow_path(root_dir, part, subset, scene, side,
                                                   self.frame_number(filenames[i])),
                    self.motion_boundaries_path(root_dir, part, subset, scene, side, self.frame_number(filenames[i]))
                )

    def forward_optical_flow_path(self, root, part, subset, scene, side, frame_number):
        return os.path.join(root, "optical_flow", part, subset, scene, "into_future", side,
                            f"OpticalFlowIntoFuture_{frame_number}_{self.side_letter(side)}.pfm")

    def reverse_optical_flow_path(self, root, part, subset, scene, side, frame_number):
        return os.path.join(root, "optical_flow", part, subset, scene, "into_past", side,
                            f"OpticalFlowIntoPast_{frame_number}_{self.side_letter(side)}.pfm")

    def motion_boundaries_path(self, root, part, subset, scene, side, frame_number):
        return os.path.join(root, "motion_boundaries", part, subset, scene, "into_past", side,
                            f"{frame_number}.pgm")
