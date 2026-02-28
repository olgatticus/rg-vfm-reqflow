# 🧬 V-ReQFLow: Riemannian Variational Flow Matching for protein backbone generation 🧬

This repository contains the **protein-backbone generation** experiments of the paper **[Riemannian Variational Flow Matching for Material and Protein Design](https://arxiv.org/abs/2502.12981v2)**, accepted at ICLR 2026. 

**Authors**: Olga Zaghen, Floor Eijkelboom*, Alison Pouplin*, Cong Liu, Max Welling, Jan-Willem van de Meent, Erik J. Bekkers

\* equal contribution

Since each experiment was implemented independently, we provide separate repositories for clarity:
* Synthetic **checkerboard** experiments are in the **[main RG-VFM repository](https://github.com/olgatticus/rg-vfm)**,
* **MOF generation** experiments are in the **[V-MOFFlow repository](https://github.com/olgatticus/rg-vfm-mofflow)**.
* **Protein backbone generation** experiments are in **this repository**.

___
## Overview 

This protein experiment code is based on the original **[ReQFlow repository](https://github.com/AngxiaoYue/ReQFlow)**.
Our primary code modification is in `flow_module`: we replace the original loss with the **RG-VFM-form loss**. For full formulation and derivation details, please refer to the paper **[Riemannian Variational Flow Matching for Material and Protein Design](https://arxiv.org/abs/2502.12981v2)**.

___

## Download Checkpoints

Download checkpoints from Google Drive:

- https://drive.google.com/drive/folders/1vZKZcc9oyTs1Fcbb6hhG3YDzTQB2j0Hn?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto

Then place the downloaded checkpoint folders under `ReQFlow/` in this repository.

Expected structure:

```text
rg-vfm-reqflow/
└── ReQFlow/
    ├── rg-vfm-qflow-ckpt/
    └── rg-vfm-reqflow-ckpt/
```

___

## Run Inference

From repository root:

```bash
cd ReQFlow
PYTHONPATH=$(pwd):$PYTHONPATH python -W ignore experiments/inference_se3_flows.py -cn inference_unconditional
```
