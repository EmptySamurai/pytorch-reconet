from math import ceil
import shlex
import json
from subprocess import Popen, DEVNULL, PIPE, check_output

import numpy as np


def _check_wait(p):
    status = p.wait()
    if status != 0:
        raise Exception("{} returned non-zero status {}".format(p.args, status))


def _default_param(value, default_value):
    return default_value if value is None else value


def _ffprobe(file, cmd):
    cmd = "{cmd} -loglevel fatal -print_format json -show_format -show_streams {file}".format(cmd=cmd, file=file)
    output = check_output(shlex.split(cmd))
    return json.loads(output)


def fraction(s):
    if s is None:
        return None
    num, den = s.split("/")
    return int(num) / int(den)


class _VideoIterator:

    def __init__(self, reader):
        self._reader = reader
        self._closed = False

        cmd = []
        cmd.append("{cmd} -loglevel error -y -nostdin".format(cmd=reader.ffmpeg_cmd))
        cmd.append("-i {file}".format(file=reader.filepath))
        if self._reader.fps is not None:
            cmd.append("-filter fps=fps={fps}:round=up".format(fps=self._reader.fps))
        cmd.append("-f rawvideo -pix_fmt {pix_fmt} pipe:".format(pix_fmt=reader.format + '24'))
        cmd = " ".join(cmd)

        self._ffmpeg_output = Popen(shlex.split(cmd), stdout=PIPE, stdin=DEVNULL)

    def __next__(self):
        frame_size = self._reader.width * self._reader.height * 3
        in_bytes = self._ffmpeg_output.stdout.read(frame_size)

        assert len(in_bytes) == 0 or len(in_bytes) == frame_size

        if len(in_bytes) == 0:
            self._close()
            raise StopIteration()

        return np.frombuffer(in_bytes, np.uint8).reshape([self._reader.height, self._reader.width, 3])

    def __iter__(self):
        return self

    def __del__(self):
        self._close()

    def _close(self):
        if self._closed:
            return
        else:
            self._closed = True
            self._ffmpeg_output.kill()


class VideoReader:

    def __init__(self, filepath, fps=None, format='rgb', ffmpeg_cmd="ffmpeg", ffprobe_cmd="ffprobe"):
        probe = self.probe = _ffprobe(filepath, cmd=ffprobe_cmd)
        stream = next((stream for stream in probe["streams"] if stream['codec_type'] == "video"))

        self.width = int(stream["width"])
        self.height = int(stream["height"])
        # FPS from ffprobe can be sometimes be incorrect, so it's better to specify FPS manually
        self.fps = fps or fraction(stream.get("r_frame_rate"))
        self.duration = float(stream["duration"])
        # self.frames_count = int(ceil(self.duration * fps))

        self.filepath = filepath
        self.format = format
        self.ffmpeg_cmd = ffmpeg_cmd
        self.ffprobe_cmd = ffprobe_cmd

    def __iter__(self):
        return _VideoIterator(self)


class VideoWriter:

    def __init__(self,
                 filepath,
                 input_width,
                 input_height,
                 input_fps,
                 input_format="rgb",
                 output_width=None,
                 output_height=None,
                 output_format="yuv420p",
                 ffmpeg_cmd="ffmpeg"):
        self.filepath = filepath
        self.input_width = input_width
        self.input_height = input_height
        self.input_fps = input_fps
        self.input_format = input_format
        self.output_width = output_width
        self.output_height = output_height
        self.output_format = output_format
        self.ffmpeg_cmd = ffmpeg_cmd

    def __enter__(self):
        cmd = []
        cmd.append("{cmd} -y -loglevel error".format(cmd=self.ffmpeg_cmd))
        cmd.append("-f rawvideo -pix_fmt {pix_fmt} -video_size {width}x{height} -framerate {fps} -i pipe:".format(
            pix_fmt=self.input_format + '24',
            width=self.input_width,
            height=self.input_height,
            fps=self.input_fps
        ))
        cmd.append("-pix_fmt {pix_fmt}".format(pix_fmt=self.output_format))
        if self.output_width is not None and self.output_height is not None:
            cmd.append("-s {width}x{height}".format(width=self.output_width, height=self.output_height))

        cmd.append(self.filepath)
        cmd = " ".join(cmd)

        self._ffmpeg_output = Popen(shlex.split(cmd), stdin=PIPE)
        return self

    def write(self, frame):
        assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape == (self.input_height, self.input_width, 3)
        self._ffmpeg_output.stdin.write(frame.tobytes())

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._ffmpeg_output.stdin.close()
        _check_wait(self._ffmpeg_output)
