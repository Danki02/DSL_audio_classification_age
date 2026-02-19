"""
Preprocessing logic.

This wraps the notebook's `preprocessing(df, audio_path)` function and keeps it
separate from feature extraction.

Typical usage:
    from src.preprocessing.preprocess import preprocess_dataframe
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.feature_extraction.audio_features import create_features_dataframe_in_batches


def preprocess_dataframe(df: pd.DataFrame, audio_base_path: str, batch_size: int = 100) -> pd.DataFrame:
    """Clean metadata columns and compute audio features."""
    df = df.copy()

    # Fix known typo
    if "gender" in df.columns:
        df["gender"] = df["gender"].replace("famale", "female")
        le = LabelEncoder()
        df["gender"] = le.fit_transform(df["gender"].astype(str))

    # Tempo column comes like "[0.12]" -> 0.12
    if "tempo" in df.columns:
        df["tempo"] = df["tempo"].astype(str).str.strip("[]").astype(float)

    # Drop columns not used by the final model
    if "ethnicity" in df.columns:
        df = df.drop(columns=["ethnicity"])

    final_df = create_features_dataframe_in_batches(df, audio_base_path, batch_size=batch_size)

    # Remove extra columns if they exist
    final_df = final_df.drop(
        columns=["num_words", "num_characters", "num_pauses", "sampling_rate"],
        errors="ignore",
    )
    return final_df
