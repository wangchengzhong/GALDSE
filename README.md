# GALD-SE: Guided Anisotropic Lightweight Diffusion for Efficient Speech Enhancement

Official PyTorch Implementation of ["GALD-SE: Guided Anisotropic Lightweight Diffusion for Efficient Speech Enhancement"](https://ieeexplore.ieee.org/document/10816305) (IEEE Signal Processing Letters 2025)



## Overview

GALD-SE is a diffusion-based speech enhancement method that leverages guided anisotropic noise to preserve clean speech clues during the diffusion process. Key features:

- Lightweight architecture with only 4.5M parameters
- Fast inference with just 6 sampling steps
- State-of-the-art performance on multiple benchmarks

## Installation

1. Create a new virtual environment with Python 3.9 (we have not tested other Python versions, but they may work):


2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

Train the model using:

```bash
python train.py --base_dir <your_base_dir> --batch_size <your_batch_size_per_gpu> # 32 in total is recommended
```
We have incorporated several additional loss terms into the diffusion network's training objective, as proposed in the paper below, resulting in further performance enhancements. When evaluated on the VBD dataset, this fine-tuned system achieves a PESQ score of 3.40 and SI-SNR values exceeding 20dB.

```bibtex
@article{richter2024investigating,
    title={Investigating Training Objectives for Generative Speech Enhancement},
    author={Julius Richter and Danilo de Oliveira and Timo Gerkmann},
    journal={arXiv preprint arXiv:2409.10753},
    year={2024}
}
```

If you choose not to use the additional loss terms, we recommend first training for approximately 700 epochs with default parameters, then performing fine-tuning with the following configuration for 10~30 epochs to reproduce the final results:

```bash
python train.py --base_dir <your_base_dir> --spec_exp_exponent 0.76 --spec_factor 0.15 --kappa 0.1 --power 0.39 --min_noise_level 0.006 --batch_size <your_batch_size> --finetuned --resume_from_checkpoint <your_checkpoint_path>
```

Note: During fine-tuning, we modify the noise generation formula from `noise = torch.randn_like(x) * mask` to `noise = torch.randn_like(x) * torch.sqrt(mask)`(i.e. making the noise more "challenging"). Note that this modification only affects the fine-tuning phase and does not require changes to the inference process.


Your base directory should contain:
- `train/` - Training data directory
  - `clean/` - Clean speech files
  - `noisy/` - Noisy speech files  
- `valid/` - Validation data directory
  - `clean/` - Clean speech files
  - `noisy/` - Noisy speech files
- `test/` - Test data directory
  - `clean/` - Clean speech files
  - `noisy/` - Noisy speech files

All audio files should be in wav format.


## Evaluation 

1. Generate enhanced speech:
```bash
python enhancement.py --noisy_dir <your_noisy_dir> --enhanced_dir <your_enhanced_dir> --ckpt <path_to_checkpoint> --batch_size <your_batch_size>
```

2. Calculate metrics:
```bash
python c_metrics.py --clean_dir <your_clean_dir> --enhanced_dir <your_enhanced_dir> --csvname <your_csv_name.csv>
```



## Citation

If you find this work useful, please kindly cite the following papers:
```bibtex
@article{10816305,
  author={Wang, Chengzhong and Gu, Jianjun and Yao, Dingding and Li, Junfeng and Yan, Yonghong},
  journal={IEEE Signal Processing Letters}, 
  title={GALD-SE: Guided Anisotropic Lightweight Diffusion for Efficient Speech Enhancement}, 
  year={2025},
  volume={32},
  number={},
  pages={426-430},
  doi={10.1109/LSP.2024.3522852}}

@article{richter2023speech,
  title={Speech Enhancement and Dereverberation with Diffusion-based Generative Models},
  author={Richter, Julius and Welker, Simon and Lemercier, Jean-Marie and Lay, Bunlong and Gerkmann, Timo},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={31},
  pages={2351-2364},
  year={2023},
  doi={10.1109/TASLP.2023.3285241}
}

@inproceedings{NEURIPS2023_2ac2eac5,
 author = {Yue, Zongsheng and Wang, Jianyi and Loy, Chen Change},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {13294--13307},
 publisher = {Curran Associates, Inc.},
 title = {ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/2ac2eac5098dba08208807b65c5851cc-Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}

```



## Acknowledgments

Special thanks to the authors of [sgmse+](https://github.com/sp-uhh/galdse) and [Resshift](https://github.com/zsyOAOA/ResShift) for their open-source contribution which served as the foundation for this work.