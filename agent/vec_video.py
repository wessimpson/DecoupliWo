from pathlib import Path
from typing import Callable

import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import Image

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper


def _nn_upscale(frame: np.ndarray, scale: int) -> np.ndarray:
    h, w = frame.shape[:2]
    im = Image.fromarray(frame)
    im = im.resize((w * scale, h * scale), Image.NEAREST)
    return np.asarray(im)


def _save_mp4(path: str, frames: list[np.ndarray], fps: int) -> None:
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(path)
    del clip


class VecSub0ClipRecorder(VecEnvWrapper):
    """Sub-env 0 → upscaled MP4 clips. ``record_video_trigger`` receives the same cumulative
    step count Stable-Baselines3 uses for ``total_timesteps``: each vec ``step`` adds ``num_envs``."""

    def __init__(
        self,
        venv: VecEnv,
        video_folder: str,
        fps: int,
        record_video_trigger: Callable[[int], bool],
        video_length: int,
        name_prefix: str,
        upscale: int,
    ):
        super().__init__(venv)
        if fps < 1 or video_length < 2 or upscale < 1:
            raise ValueError("fps >= 1, video_length >= 2, upscale >= 1 required")
        if venv.render_mode != "rgb_array":
            raise ValueError(f"render_mode must be 'rgb_array', got {venv.render_mode!r}")

        self._fps = fps
        self._trigger = record_video_trigger
        self._dir = Path(video_folder).resolve()
        self._prefix = name_prefix
        self._clip_len = video_length
        self._scale = upscale
        self._timesteps = 0
        self._active = False
        self._frames: list[np.ndarray] = []
        self._path = ""
        self._dir.mkdir(parents=True, exist_ok=True)

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        if self._trigger(self._timesteps):
            self._start_clip()
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        out = self.venv.step_wait()
        # Match OnPolicyAlgorithm: num_timesteps += env.num_envs per vec step.
        self._timesteps += self.num_envs
        if self._active:
            self._frames.append(_nn_upscale(self._rgb_env0(), self._scale))
            if len(self._frames) > self._clip_len:
                print(f"Saving video to {self._path}")
                self._finish_clip()
        elif self._trigger(self._timesteps):
            self._start_clip()
        return out

    def close(self) -> None:
        self._finish_clip()
        VecEnvWrapper.close(self)

    def _rgb_env0(self) -> np.ndarray:
        frame = self.venv.get_images()[0]
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"env 0 render must be ndarray, got {type(frame)}")
        return frame

    def _start_clip(self) -> None:
        self._finish_clip()
        self._path = str(
            self._dir
            / f"{self._prefix}-from_ts_{self._timesteps}_{self._clip_len}f.mp4"
        )
        self._active = True
        self._frames = [_nn_upscale(self._rgb_env0(), self._scale)]

    def _finish_clip(self) -> None:
        if not self._active:
            return
        self._active = False
        frames, path = self._frames, self._path
        self._frames = []
        self._path = ""
        if not frames:
            raise RuntimeError("clip finished with zero frames")
        _save_mp4(path, frames, self._fps)
