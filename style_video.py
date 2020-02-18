import os
import argparse

import numpy as np

from lib import ReCoNetModel
from ffmpeg_tools import VideoReader, VideoWriter


def create_folder_for_file(path):
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input video file")
    parser.add_argument("output", help="Path to output video file")
    parser.add_argument("model", help="Path to model file")
    parser.add_argument("--use-cpu", action='store_true', help="Use CPU instead of GPU")
    parser.add_argument("--gpu-device", type=int, default=None, help="GPU device index")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--fps", type=int, default=None, help="FPS of output video")
    parser.add_argument("--frn", action='store_true', help="Use Filter Response Normalization and TLU ")

    args = parser.parse_args()

    batch_size = args.batch_size

    model = ReCoNetModel(args.model, use_gpu=not args.use_cpu, gpu_device=args.gpu_device, frn=args.frn)

    reader = VideoReader(args.input, fps=args.fps)

    create_folder_for_file(args.output)
    writer = VideoWriter(args.output, reader.width, reader.height, reader.fps)

    with writer:
        batch = []

        for frame in reader:
            batch.append(frame)

            if len(batch) == batch_size:
                batch = np.array(batch)
                for styled_frame in model.run(batch):
                    writer.write(styled_frame)

                batch = []

        if len(batch) != 0:
            batch = np.array(batch)
            for styled_frame in model.run(batch):
                writer.write(styled_frame)
