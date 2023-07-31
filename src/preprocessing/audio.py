import os
from typing import Tuple

import librosa
import numpy as np
from moviepy.editor import VideoFileClip
from pydub import AudioSegment


class Audio:
    def __init__(
        self,
        audio_file: str,
        sr: int = 22050,
        begin_offset: float = 10.0,
        duration: float = 30.0,
        frame_length: int = 2048,
        hop_length: int = 512,
        n_mfcc: int = 13,
    ):
        self.audio_file = audio_file
        self._sampling_rate = sr
        self.begin_offset = begin_offset
        self.duration = duration
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.genre = __class__.extract_genre(audio_file)
        self.audio = self.run_loading_pipeline()

    @staticmethod
    def extract_genre(file_name: str) -> str:
        return os.path.split(file_name)[-1].split(".")[0].split("_")[0]

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, new_sampling_rate: int):
        if self._sampling_rate is not None:
            self.audio = librosa.resample(
                self.audio, orig_sr=self._sampling_rate, target_sr=new_sampling_rate
            )
        self._sampling_rate = new_sampling_rate

    @staticmethod
    def extract_file_name(file: str) -> str:
        return os.path.splitext(file)[0]

    @staticmethod
    def generate_file_name_with_new_ext(file: str, extension: str) -> str:
        filename = __class__.extract_file_name(file)
        return "".join((filename, extension))

    @staticmethod
    def mp4_to_mp3(video_file: str) -> str:
        dst_path = __class__.generate_file_name_with_new_ext(video_file, ".mp3")
        video = VideoFileClip(video_file)
        video.audio.write_audiofile(dst_path)
        return dst_path

    @staticmethod
    def mp3_to_wav(audio_file: str) -> str:
        dst_path = __class__.generate_file_name_with_new_ext(audio_file, ".wav")
        audio = AudioSegment.from_mp3(audio_file)
        audio.export(dst_path, format="wav")
        return dst_path

    @staticmethod
    def load_wav(
        audio_file: str,
        sr: int,
        begin_offset: float = 5.0,
        duration: float = 30.0,
    ) -> Tuple[np.ndarray, int]:
        y, sr = librosa.load(audio_file, sr=sr, offset=begin_offset, duration=duration)
        y, _ = librosa.effects.trim(y)
        return y, sr

    def run_loading_pipeline(self) -> np.ndarray:
        file_ext = os.path.splitext(self.audio_file)[1]
        if file_ext == ".mp4":
            self.audio_file = __class__.mp4_to_mp3(self.audio_file)
            file_ext = ".mp3"
        if file_ext == ".mp3":
            self.audio_file = __class__.mp3_to_wav(self.audio_file)
            file_ext = ".wav"
        if file_ext == ".wav":
            try:
                audio, sr = __class__.load_wav(
                    self.audio_file,
                    self.sampling_rate,
                    self.begin_offset,
                    self.duration,
                )
                self._sampling_rate = sr
            except:
                return None
        else:
            raise TypeError("Unexpected input type! (`mp3`, `mp4` or `wav` expected)")
        return audio

    @classmethod
    def load_audio(
        cls,
        audio_file: str,
        sr: int = 22050,
        begin_offset: float = 0.0,
        duration: float = 30.0,
        frame_length: int = 2048,
        hop_length: int = 512,
        n_mfcc: int = 13,
    ):
        audio = cls(
            audio_file,
            sr,
            begin_offset,
            duration,
            frame_length,
            hop_length,
            n_mfcc,
        )
        return audio
