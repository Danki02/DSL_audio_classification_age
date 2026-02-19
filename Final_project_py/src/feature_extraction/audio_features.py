"""
Feature extraction utilities for audio files.

This module contains the functions originally developed in the notebook:
- extract_audio_features: compute Mel/MFCC + delta statistics on non-silent audio.
- create_features_dataframe_in_batches: iterate over a metadata dataframe and
  build a tabular feature dataframe.

Notes:
- The code assumes `df` contains at least: Id, path, and other metadata columns.
- `audio_base_path` should point to the folder that contains the audio files,
  not the CSV.
"""
from __future__ import annotations

import gc
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import librosa


def extract_audio_features(audio_path: str) -> Optional[Dict[str, Any]]:
    """Extract audio features from a single file.

    The notebook logic:
    - load audio at 44.1kHz
    - trim silence based on `top_db`
    - compute log-mel and MFCC
    - compute deltas (speed) and delta-deltas (acceleration)
    - aggregate by mean/std per coefficient
    - compute ACF mean/std on the trimmed signal

    Returns a dict with numpy arrays and scalar features, or None on failure.
    """
    try:
        audio_44100, sr_44100 = librosa.load(audio_path, sr=44100)
        non_silent_intervals = librosa.effects.split(audio_44100, top_db=25)
        if len(non_silent_intervals) == 0:
            # all silence
            return None

        y_trimmed = np.concatenate([audio_44100[start:end] for start, end in non_silent_intervals])

        duration_trimmed = sum((end - start) for start, end in non_silent_intervals) / sr_44100

        # Mel Spectrogram (log scale)
        mel_spectrogram = librosa.feature.melspectrogram(y=y_trimmed, sr=sr_44100, n_mels=40)
        log_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        features_mel = np.concatenate([np.mean(log_spectrogram, axis=1), np.std(log_spectrogram, axis=1)])

        # Deltas on log-mel
        delta_features = librosa.feature.delta(log_spectrogram)
        features_speed = np.concatenate([np.mean(delta_features, axis=1), np.std(delta_features, axis=1)])
        delta_delta_features = librosa.feature.delta(log_spectrogram, order=2)
        features_acceleration = np.concatenate([np.mean(delta_delta_features, axis=1), np.std(delta_delta_features, axis=1)])

        # MFCC
        mfcc = librosa.feature.mfcc(y=audio_44100, sr=sr_44100, n_mfcc=13)
        features_mfcc = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])

        # MFCC deltas
        mfcc_speed = librosa.feature.delta(mfcc)
        features_mfcc_speed = np.concatenate([np.mean(mfcc_speed, axis=1), np.std(mfcc_speed, axis=1)])
        mfcc_acc = librosa.feature.delta(mfcc, order=2)
        features_mfcc_acceleration = np.concatenate([np.mean(mfcc_acc, axis=1), np.std(mfcc_acc, axis=1)])

        # Autocorrelation on trimmed signal
        acf = librosa.autocorrelate(y_trimmed)
        acf_mean = float(np.mean(acf))
        acf_std = float(np.std(acf))

        return {
            "mel_features": features_mel,
            "mfcc_features": features_mfcc,
            "speed_features": features_speed,
            "acceleration_features": features_acceleration,
            "mfcc_acc": features_mfcc_acceleration,
            "mfcc_speed": features_mfcc_speed,
            "duration_trimmed": float(duration_trimmed),
            "acf_mean": acf_mean,
            "acf_std": acf_std,
        }

    except Exception as e:
        print(f"Errore durante il processamento di {audio_path}: {e}")
        return None


def create_features_dataframe_in_batches(df: pd.DataFrame, audio_base_path: str, batch_size: int = 100) -> pd.DataFrame:
    """Compute audio features for a dataframe, batching to reduce memory spikes."""
    calculated_features = []

    def batch_generator(_df: pd.DataFrame, _batch_size: int):
        for i in range(0, len(_df), _batch_size):
            yield _df.iloc[i : i + _batch_size]

    for batch_df in batch_generator(df, batch_size):
        batch_features = []
        for _, row in batch_df.iterrows():
            # Notebook print
            if "Id" in row:
                print(f"Sono arrivato alla riga: {row['Id']}")

            # Resolve path according to notebook conventions
            if isinstance(row.get("path", None), str):
                if "audios_evaluation" in row["path"]:
                    audio_path = os.path.join(audio_base_path, row["path"][len("audios_evaluation/"):])
                elif "audios_development" in row["path"]:
                    audio_path = os.path.join(audio_base_path, row["path"][len("audios_development/"):])
                else:
                    # fallback: use basename
                    audio_path = os.path.join(audio_base_path, os.path.basename(row["path"]))
            else:
                continue

            features = extract_audio_features(audio_path)
            if not features:
                continue

            combined_features = row.to_dict()

            # Expand arrays into columns (mean_* and std_*)
            mel_features = features["mel_features"]
            half = len(mel_features) // 2
            for i in range(half):
                combined_features[f"mean_mel_{i+1}"] = float(mel_features[i])
                combined_features[f"std_mel_{i+1}"] = float(mel_features[half + i])

            mfcc_features = features["mfcc_features"]
            half = len(mfcc_features) // 2
            for i in range(half):
                combined_features[f"mean_mfcc_{i+1}"] = float(mfcc_features[i])
                combined_features[f"std_mfcc_{i+1}"] = float(mfcc_features[half + i])

            speed_features = features["speed_features"]
            half = len(speed_features) // 2
            for i in range(half):
                combined_features[f"mean_speed_{i+1}"] = float(speed_features[i])
                combined_features[f"std_speed_{i+1}"] = float(speed_features[half + i])

            acceleration_features = features["acceleration_features"]
            half = len(acceleration_features) // 2
            for i in range(half):
                combined_features[f"mean_acceleration_{i+1}"] = float(acceleration_features[i])
                combined_features[f"std_acceleration_{i+1}"] = float(acceleration_features[half + i])

            mfcc_speed_features = features["mfcc_speed"]
            half = len(mfcc_speed_features) // 2
            for i in range(half):
                combined_features[f"mean_mfcc_speed_{i+1}"] = float(mfcc_speed_features[i])
                combined_features[f"std_mfcc_speed_{i+1}"] = float(mfcc_speed_features[half + i])

            mfcc_acc_features = features["mfcc_acc"]
            half = len(mfcc_acc_features) // 2
            for i in range(half):
                combined_features[f"mean_mfcc_acceleration_{i+1}"] = float(mfcc_acc_features[i])
                combined_features[f"std_mfcc_acceleration_{i+1}"] = float(mfcc_acc_features[half + i])

            combined_features["acf_mean"] = features["acf_mean"]
            combined_features["acf_std"] = features["acf_std"]
            combined_features["duration"] = features["duration_trimmed"]

            batch_features.append(combined_features)

            del features
            gc.collect()

        calculated_features.extend(batch_features)

        del batch_features
        gc.collect()

    return pd.DataFrame(calculated_features)
