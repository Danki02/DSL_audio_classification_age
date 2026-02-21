# Age Regression from Speech Signals

**Data Science Lab – Winter Call A.Y. 2024/2025**
Politecnico di Torino

This project addresses the task of **speaker age estimation from speech signals**, framed as a regression problem.
The objective is to predict the chronological age of a speaker using acoustic and metadata features extracted from spoken sentences.

The project follows the official assignment guidelines provided in the course specification .

---

## Project Overview

Given a dataset of speech recordings and pre-extracted acoustic features, the goal is to:

* Build a complete regression pipeline
* Extract additional audio features (Mel-Spectrograms, MFCCs, deltas, etc.)
* Train machine learning models
* Optimize hyperparameters
* Minimize **Root Mean Square Error (RMSE)**

---

## 📂 Repository Structure

```
DSL_Winter_Project_2025/
│
├── src/                    # Modularized pipeline
│   ├── preprocessing/
│   ├── feature_extraction/
│   ├── training/
│   └── plots/
│
├── main.py                 # End-to-end execution script
├── requirements.txt
├── DSL_report.pdf          # Final report
├── Final_project_jupyter.ipynb  # Original notebook version
└── README.md
```

⚠️ Audio files are NOT stored in this repository due to size constraints.

---

## 📊 Dataset

The dataset consists of:

* **2,933 development samples**
* **691 evaluation samples**
* Total: **3,624 audio recordings**

Each sample contains acoustic features such as:

* Pitch statistics
* Jitter / Shimmer
* Spectral centroid
* ZCR
* HNR
* Tempo
* Linguistic metadata
* Path to WAV file

### 🔗 Dataset Availability

The datasets are publicly available on Hugging Face:

* **Training set:**
  [https://huggingface.co/datasets/danki2meme/Audio_for_age_classification_Train](https://huggingface.co/datasets/danki2meme/Audio_for_age_classification_Train)

* **Evaluation set:**
  [https://huggingface.co/datasets/danki2meme/Audio_for_age_classification_Eval](https://huggingface.co/datasets/danki2meme/Audio_for_age_classification_Eval)

---

## ⚙️ Models Implemented

Two regression models were evaluated:

* **Random Forest Regressor (RFR)**
* **Support Vector Regressor (SVR)**

Hyperparameter tuning was performed using:

* Grid Search (Random Forest)
* Optuna (SVR)

Best configuration:

* SVR with optimized (C) and (\epsilon)
* Age-group balancing threshold applied

Final public score on DSLE:

```
RMSE = 9.114
```

---

## ▶️ How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run full pipeline:

```bash
python main.py \
  --train_csv development.csv \
  --eval_csv evaluation.csv \
  --audio_base path/to/audio \
  --workdir outputs \
  --max_sample_cat 150
```

---

## Internal Technical Documentation

This README provides a general overview.

For detailed instructions about:

* Modular pipeline structure
* Preprocessing steps
* Feature extraction details
* CLI usage examples

 See the internal README located in the project structure .

---

## Notes on LLM Usage

In accordance with course rules :

* LLMs were used **only for report drafting**
* No external datasets or pre-trained models were used for implementation
* The regression pipeline was fully implemented from scratch

---

## Authors

Rosario Interlandi
Ramadan Mehmetaj
Politecnico di Torino

