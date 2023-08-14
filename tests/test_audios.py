import sys
from pathlib import Path

sys.path.append("./src")
import librosa
import numpy as np
import preprocessing.utils as utils
import pytest
from prefect.testing.utilities import prefect_test_harness
from preprocessing.audio import Audio
from preprocessing.feature_extractor import (
    FeatureExtractor,
    run_feature_extraction_from_audio_file,
)
from preprocessing.preprocessing_workflows import run_feature_extraction_all

CUR_PATH = Path(__file__).parent.resolve()


@pytest.fixture
def mp3_audio():
    file = CUR_PATH / "test_data/dustbus.mp3"
    config_file = CUR_PATH / "test_data/audio_config.yaml"
    audio_config = utils.load_config_from_yaml(config_file)
    audio_config["audio_file"] = file
    audio = Audio(**audio_config)
    return audio


@pytest.fixture
def wav_audio():
    file = CUR_PATH / "test_data/dustbus.wav"
    config_file = CUR_PATH / "test_data/audio_config.yaml"
    audio_config = utils.load_config_from_yaml(config_file)
    audio_config["audio_file"] = file
    audio = Audio(**audio_config)
    return audio


def test_load_mp3(mp3_audio):
    assert mp3_audio.audio is not None and isinstance(mp3_audio.audio, np.ndarray)
    assert isinstance(mp3_audio.sampling_rate, int) and mp3_audio.sampling_rate > 0
    np.testing.assert_allclose(
        [librosa.get_duration(y=mp3_audio.audio, sr=mp3_audio.sampling_rate)],
        [mp3_audio.duration],
        rtol=1e-1,
    )


def test_load_wav(wav_audio):
    assert wav_audio.audio is not None and isinstance(wav_audio.audio, np.ndarray)
    assert isinstance(wav_audio.sampling_rate, int) and wav_audio.sampling_rate > 0
    np.testing.assert_allclose(
        [librosa.get_duration(y=wav_audio.audio, sr=wav_audio.sampling_rate)],
        [wav_audio.duration],
        rtol=1e-1,
    )


def test_resample(wav_audio):
    new_sampling_rate = 22050 if wav_audio.sampling_rate != 22050 else 44100
    wav_audio.sampling_rate = new_sampling_rate
    assert wav_audio.sampling_rate == new_sampling_rate


def test_audio_features(wav_audio):
    feature_extractor = FeatureExtractor()
    features_dict = feature_extractor(wav_audio)
    assert isinstance(features_dict, dict)


def test_feature_extraction_func():
    file = CUR_PATH / "test_data/dustbus.wav"
    config_file = CUR_PATH / "test_data/audio_config.yaml"
    feature_dict = run_feature_extraction_from_audio_file.fn(file, config_file)
    assert isinstance(feature_dict, list) and isinstance(feature_dict[0], dict)


def test_feature_extraction_flow():
    with prefect_test_harness():
        files = utils.get_files_list.fn("data/raw/random_data_cut")
        config_file = CUR_PATH / "test_data/audio_config.yaml"
        (X, y), _, column_names, encoder = run_feature_extraction_all(
            files, config_file
        )
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
        assert X.shape[0] == len(files)
        assert X.shape[0] == y.shape[0]
        assert len(encoder.classes_) == len(np.unique(y))
