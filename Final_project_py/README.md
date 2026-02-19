# Audio Age Prediction (refactor from notebook)

This repo is a Git-friendly refactor of the original Jupyter notebook into modules:

- `src/preprocessing/` : dataframe cleanup + orchestration
- `src/feature_extraction/` : audio feature extraction (Mel/MFCC + deltas)
- `src/training/` : training / test + hyperparameter tuning
- `src/plots/` : plotting utilities
- `scripts/` : runnable CLI entrypoints
- `notebooks/` : original notebook kept for reference

## Quick start

```bash
pip install -r requirements.txt
```

### 1) Preprocess (build feature CSV)
```bash
python scripts/run_preprocessing.py --csv path/to/development.csv --audio_base path/to/audios_development --out data/train_features.csv
python scripts/run_preprocessing.py --csv path/to/evaluation.csv  --audio_base path/to/audios_evaluation  --out data/eval_features.csv
```

### 2) Train + predict
```bash
python scripts/train_and_predict.py --train_features_csv data/train_features.csv --eval_features_csv data/eval_features.csv --eval_ids_csv path/to/evaluation.csv --max_sample_cat 150 --out_submission submission.csv
```

### 3) Tune (optional)
```bash
python scripts/tune_model.py --train_features_csv data/train_features.csv --model RF --max_sample_cat 450
```

### 4) Plots (optional)
```bash
python scripts/make_plots.py --train_features_csv data/train_features.csv --start 140 --stop 181 --step 10
```


## Run everything from `main.py`

Example:

```bash
python main.py --train_csv data/train.csv --eval_csv data/eval.csv --audio_base data/audio \
  --workdir outputs --batch_size 100 --max_sample_cat 150 --make_plots --plots_out outputs/rmse.png
```

Optional tuning:

```bash
python main.py --train_csv data/train.csv --eval_csv data/eval.csv --audio_base data/audio --tune --tune_model RF
```
