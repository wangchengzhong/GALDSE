# GALD-SE: Guided Anisotropic Lightweight Diffusion for Efficient Speech Enhancement

Official PyTorch Implementation of ["GALD-SE: Guided Anisotropic Lightweight Diffusion for Efficient Speech Enhancement"](https://ieeexplore.ieee.org/document/10816305) (IEEE Signal Processing Letters 2025)



## Overview

GALD-SE is a diffusion-based speech enhancement method that leverages guided anisotropic noise to preserve clean speech clues during the diffusion process. Key features:

- Lightweight architecture with only <5M parameters
- Fast inference with just 6 sampling steps
- State-of-the-art performance on multiple benchmarks

## Update

Our implementation differs slightly from the architecture described in the published paper, in channel dimensions and bottleneck structure (from attention block to conformer block). To better demonstrate the intrinsic effectiveness of our method, we have removed explicit guidance inputs from the diffusion model, without significant degradation. This modification was made after observing that the model would otherwise become overly dependent on the guidance signals rather than learning robust noise removal capabilities independently, which cannot effectively validate our method's core contribution and may limit performance in challenging scenarios.

## Installation

1. Create a new virtual environment with Python 3.9 (we have not tested other Python versions, but they may work):


2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

Train the model using:

```bash
python train.py --base_dir <your_base_dir> --batch_size <your_batch_size_per_gpu> # 42 in total is recommended
```
We have incorporated several additional loss terms into the diffusion network's training objective, as proposed in the paper below, resulting in further performance enhancements. 

```bibtex
@article{richter2024investigating,
    title={Investigating Training Objectives for Generative Speech Enhancement},
    author={Julius Richter and Danilo de Oliveira and Timo Gerkmann},
    journal={arXiv preprint arXiv:2409.10753},
    year={2024}
}
```


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