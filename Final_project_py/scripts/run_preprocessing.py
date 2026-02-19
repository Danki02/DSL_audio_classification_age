#!/usr/bin/env python
from __future__ import annotations

import argparse
import pandas as pd

from src.preprocessing.preprocess import preprocess_dataframe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to input metadata CSV (development/evaluation)")
    ap.add_argument("--audio_base", required=True, help="Folder containing audio files")
    ap.add_argument("--out", required=True, help="Output CSV path for features")
    ap.add_argument("--batch_size", type=int, default=100)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out_df = preprocess_dataframe(df, args.audio_base, batch_size=args.batch_size)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
