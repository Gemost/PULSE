# PULSE: Generative Phase Evolution for Non-Stationary Time Series Forecasting

<p align="center">
  <img src="assets/teaser.png" width="92%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Conference-ICML%202026-red" alt="Conference">
  <img src="https://img.shields.io/badge/Task-Time%20Series%20Forecasting-green" alt="Task">
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange" alt="Framework">
  <img src="https://img.shields.io/badge/Model-PULSE-blue" alt="Model">
</p>

This is the official implementation of **PULSE: Generative Phase Evolution for Non-Stationary Time Series Forecasting**.

PULSE is a physics-informed framework for non-stationary time series forecasting. It shifts forecasting from **passive historical fitting** to **generative phase evolution**, enabling a lightweight backbone to model evolving temporal structures under distribution shifts.

## Introduction

Time series forecasting under non-stationarity requires models to capture stable temporal structures while adapting to future distribution shifts. Existing methods often rely on static historical assumptions, such as restoring future statistics from historical windows or directly copying past periodic patterns. These assumptions can lead to **Phase Amnesia**, where models lose awareness of the evolving global temporal context.

PULSE addresses this issue with a simple design philosophy:

```text
Disentangle → Evolve → Simulate
```

PULSE contains three key components:

- **Phase-Anchored Disentanglement**  
  Separates deterministic phase structures from stochastic residual fluctuations.

- **Generative Phase Router**  
  Generates future phase trajectories instead of rigidly copying historical patterns.

- **Statistic-Aware Mixup**  
  Simulates residual distribution shifts while avoiding artificial scale collapse.

---

## Framework

<p align="center">
  <img src="assets/framework.png" width="90%">
</p>

Given a historical sequence, PULSE first constructs a deterministic phase anchor and extracts the residual component:

```text
Historical Input = Phase Anchor + Residual
```

Different from standard normalization methods that normalize the raw sequence directly, PULSE normalizes only the stochastic residual while preserving the deterministic phase anchor in the original coordinate system. The Phase Router then evolves the historical phase anchor into a future-oriented phase anchor. Finally, PULSE reconstructs the prediction by injecting the generated future anchor back into the denormalized residual.

---

## Main Results

<p align="center">
  <img src="assets/main_results.png" width="90%">
</p>

PULSE is evaluated on 12 real-world multivariate time series forecasting datasets.

```text
MSE:   10 / 12 first-place results
MAE:    8 / 12 first-place results
Total: 18 / 24 first-place entries
```

These results suggest that a phase-aware inductive bias can be more important than simply increasing architectural complexity.

---

## Plug-and-Play Capability

<p align="center">
  <img src="assets/plug_and_play.png" width="86%">
</p>

PULSE can also be used as a plug-and-play enhancement module for different forecasting backbones. The repository includes implementations of several representative models:

```text
Autoformer
CycleNet
DLinear
iTransformer
PatchTST
PULSE
TimesNet
TimeXer
```

---

## Visualization

<p align="center">
  <img src="assets/phase_anchor_vis.png" width="90%">
</p>

The visualization shows that PULSE does not simply copy historical patterns. Instead, it learns future-oriented phase evolution and generates adaptive future phase anchors.

---

## Requirements

Create the environment with Conda:

```bash
conda create -n PULSE python=3.8
conda activate PULSE
pip install -r requirements.txt
```

The main dependencies are:

```text
numpy
matplotlib
pandas
scikit-learn
torch
```

---

## Data Preparation

All datasets used in the experiments can be downloaded from:

```text
https://drive.usercontent.google.com/download?id=1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot&export=download&authuser=0
```

These datasets are provided by the official iTransformer repository.

After downloading the datasets, create a folder named `all_datasets/` in the root directory and place all dataset files directly into this folder.

```text
PULSE/
├── all_datasets/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   ├── ETTm2.csv
│   ├── electricity.csv
│   ├── traffic.csv
│   ├── weather.csv
│   ├── solar_AL.txt
│   ├── PEMS03.npz
│   ├── PEMS04.npz
│   ├── PEMS07.npz
│   └── PEMS08.npz
```

Please make sure that the dataset filenames are consistent with the `data_path_name` specified in each script under `scripts/`.

---

## Dataset Summary

We evaluate PULSE on both long-term and short-term forecasting benchmarks.

| Dataset | File Name | Variables | Prediction Lengths | Script |
| --- | --- | ---: | --- | --- |
| ETTh1 | `ETTh1.csv` | 7 | 96, 192, 336, 720 | `scripts/ETTh1.sh` |
| ETTh2 | `ETTh2.csv` | 7 | 96, 192, 336, 720 | `scripts/ETTh2.sh` |
| ETTm1 | `ETTm1.csv` | 7 | 96, 192, 336, 720 | `scripts/ETTm1.sh` |
| ETTm2 | `ETTm2.csv` | 7 | 96, 192, 336, 720 | `scripts/ETTm2.sh` |
| Electricity | `electricity.csv` | 321 | 96, 192, 336, 720 | `scripts/electricity.sh` |
| Traffic | `traffic.csv` | 862 | 96, 192, 336, 720 | `scripts/traffic.sh` |
| Weather | `weather.csv` | 21 | 96, 192, 336, 720 | `scripts/Weather.sh` |
| Solar | `solar_AL.txt` | 137 | 96, 192, 336, 720 | `scripts/Solar.sh` |
| PEMS03 | `PEMS03.npz` | 358 | 12, 24, 48, 96 | `scripts/PEMS03.sh` |
| PEMS04 | `PEMS04.npz` | 307 | 12, 24, 48, 96 | `scripts/PEMS04.sh` |
| PEMS07 | `PEMS07.npz` | 883 | 12, 24, 48, 96 | `scripts/PEMS07.sh` |
| PEMS08 | `PEMS08.npz` | 170 | 12, 24, 48, 96 | `scripts/PEMS08.sh` |

The default look-back length is fixed as:

```text
seq_len = 96
```

The evaluation metrics are:

```text
MSE and MAE
```

---

## Training

All training scripts are provided in the `scripts/` folder.

To reproduce the results on ETTh1, run:

```bash
bash ./scripts/ETTh1.sh
```

To run other datasets:

```bash
bash ./scripts/ETTh2.sh
bash ./scripts/ETTm1.sh
bash ./scripts/ETTm2.sh
bash ./scripts/electricity.sh
bash ./scripts/traffic.sh
bash ./scripts/Weather.sh
bash ./scripts/Solar.sh
bash ./scripts/PEMS03.sh
bash ./scripts/PEMS04.sh
bash ./scripts/PEMS07.sh
bash ./scripts/PEMS08.sh
```

To run all provided scripts sequentially:

```bash
for script in ./scripts/*.sh; do
    bash "$script"
done
```

---

## Custom Usage

You can customize experiments by modifying the corresponding shell script in `scripts/`.

A typical command is:

```bash
python -u run.py \
  --is_training 1 \
  --root_path ./all_datasets/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model PULSE \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 7 \
  --time_dim 1 \
  --dsa 1 \
  --dsb 1 \
  --ksize 5 \
  --d_model 16 \
  --inv_len 96 \
  --patch_size 24 \
  --rec_lambda 0 \
  --auxi_lambda 1 \
  --train_epochs 30 \
  --patience 5 \
  --gpu 0 \
  --freq h \
  --itr 1 \
  --batch_size 256 \
  --learning_rate 0.005
```

Important arguments:

| Argument | Description |
| --- | --- |
| `--model` | Model name, e.g., `PULSE` |
| `--data` | Dataset loader type, e.g., `ETTh1`, `ETTm1`, `custom`, `Solar`, `PEMS`, `Weather` |
| `--root_path` | Root directory of dataset files |
| `--data_path` | Dataset filename |
| `--features` | Forecasting setting; `M` denotes multivariate-to-multivariate forecasting |
| `--seq_len` | Historical look-back length |
| `--pred_len` | Prediction length |
| `--enc_in` | Number of input variables |
| `--time_dim` | Number of timestamp features used by PULSE |
| `--d_model` | Hidden dimension of the Phase Router |
| `--inv_len` | Codebook length for phase-anchor retrieval |
| `--patch_size` | Patch size used by the Phase Router |
| `--auxi_lambda` | Weight of the auxiliary frequency-domain loss |

---

## Outputs

The scripts create log files under:

```text
logs/
├── ETTh1/
├── ETTh2/
├── ETTm1/
└── ...
```

Model checkpoints are saved under:

```text
checkpoints/
```

Testing results are saved under:

```text
results/
```

The summary metrics are appended to:

```text
result.txt
```

---

## Repository Structure

```text
PULSE/
├── data_provider/
│   ├── data_factory.py          # Dataset selection and dataloader construction
│   ├── data_loader.py           # Dataset classes for ETT, custom, Solar, PEMS, and Weather
│   └── __init__.py
├── exp/
│   ├── exp_basic.py             # Basic experiment class
│   ├── exp_main.py              # Training, validation, testing, and prediction pipeline
│   └── __init__.py
├── layers/
│   ├── AutoCorrelation.py
│   ├── Autoformer_EncDec.py
│   ├── Embed.py
│   ├── RevIN.py
│   ├── SelfAttention_Family.py
│   ├── Transformer_EncDec.py
│   └── __init__.py
├── models/
│   ├── Autoformer.py
│   ├── CycleNet.py
│   ├── DLinear.py
│   ├── iTransformer.py
│   ├── PatchTST.py
│   ├── PULSE.py                 # Main implementation of PULSE
│   ├── TimesNet.py
│   └── TimeXer.py
├── scripts/
│   ├── ETTh1.sh
│   ├── ETTh2.sh
│   ├── ETTm1.sh
│   ├── ETTm2.sh
│   ├── electricity.sh
│   ├── traffic.sh
│   ├── Weather.sh
│   ├── Solar.sh
│   ├── PEMS03.sh
│   ├── PEMS04.sh
│   ├── PEMS07.sh
│   └── PEMS08.sh
├── utils/
│   ├── lead_estimate.py
│   ├── learnable_wavelet.py
│   ├── masking.py
│   ├── metrics.py
│   ├── polynomial.py
│   ├── shift.py
│   ├── timefeatures.py
│   ├── tools.py
│   └── __init__.py
├── run.py                       # Main entry for training and evaluation
├── requirements.txt
├── result.txt
├── README.md
└── __init__.py
```

---

## README Figures

The figures used in this README are expected to be placed under `assets/`.

```text
assets/
├── teaser.png
├── framework.png
├── main_results.png
├── plug_and_play.png
└── phase_anchor_vis.png
```

These files are only used for README visualization. They are not required for training or evaluation.

---

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{liu2026pulse,
  title     = {PULSE: Generative Phase Evolution for Non-Stationary Time Series Forecasting},
  author    = {Liu, Yangyou and Shao, Zezhi and Chen, Xinyu and Chen, Hu and Wang, Fei and Wu, Yuankai},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

---

## Acknowledgement

We sincerely thank the following repositories and benchmark suites for their valuable code bases and datasets:

```text
https://github.com/thuml/iTransformer
https://github.com/thuml/Time-Series-Library
https://github.com/yuqinie98/PatchTST
https://github.com/cure-lab/LTSF-Linear
https://github.com/zhouhaoyi/Informer2020
https://github.com/thuml/Autoformer
```

---

## Contact

For questions or discussions, please contact:

```text
Yuankai Wu: wuyk0@scu.edu.cn
```
