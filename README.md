# PULSE

## Getting Started

### 1. Environment Requirements

To get started, ensure you have Conda installed on your system and follow these steps to set up the environment:

```bash
conda create -n PULSE python=3.8
conda activate PULSE
pip install -r requirements.txt
```

### 2. Download Data
All the datasets needed for experiments can be obtained from:

[Data Download Link]: [Wait for URL]

Data Organization:

Create a directory named all_datasets in the root folder.

Place all downloaded datasets files directly into this directory.
```text
Directory Structure:
PULSE/
├── ...
├── all_datasets/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ... (place all datasets files here)
├── scripts/
│   ├── ETTh1.sh
│   └── ...
├── run.py
└── README.md
```
### 3. Training Example
You can easily reproduce the results from the paper by running the provided script commands. For instance, to reproduce the results for the ETTh1 dataset, execute the following command:

```bash
bash ./scripts/ETTh1.sh
```
```
