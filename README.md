# imc2022
Image Matching Challenge toward reconstruct landscape from 2d image
1. 1st 2022 [Solution](https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/329131); [notebook](https://www.kaggle.com/code/gufanmingmie/imc-2022-final-ensemble)
2. 2nd 2022 [Solution](https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/329317)
### ensemble
1. [super glue](https://paperswithcode.com/paper/superglue-learning-feature-matching-with)
2. [LoFTR](https://github.com/zju3dv/LoFTR)
3. [DKM](https://github.com/Parskatt/DKM)
4. [AdaLAM](https://arxiv.org/abs/2006.04250)
5. [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)

### setup
1. download ``superglue`` as zip file, and unzip into sub folder `superGlue`
2. download ``dkm`` as zip file , unzip into `dkm` folder
<details>
<summary>tree </summary>

```
.
├── dkm
│   ├── assets
│   ├── data
│   ├── demo
│   ├── dkm
│   │   ├── benchmarks
│   │   │   └── deprecated
│   │   ├── checkpointing
│   │   ├── datasets
│   │   ├── losses
│   │   ├── models
│   │   │   ├── deprecated
│   │   │   ├── model_zoo
│   │   │   └──
│   │   ├── train
│   │   └── utils
│   ├── dkm.egg-info
│   ├── docs
│   ├── experiments
│   │   ├── deprecated
│   │   │   └── dkmv2
│   │   └── dkm
│   ├── pretrained
│   └── scripts
├── loFTR
└── superGlue
    ├── assets
    │   ├── freiburg_sequence
    │   ├── phototourism_sample_images
    │   └── scannet_sample_images
    └── models
        └── weights
```
</details>
### Weight pretrained
0. [input](https://www.kaggle.com/code/gufanmingmie/imc-2022-final-ensemble/input)
1. [LoFTR](https://www.kaggle.com/code/gufanmingmie/imc-2022-final-ensemble/input?select=loftr_outdoor.ckpt) do'nt need predownload, see [matching_LoFRT.py](matching_LoFRT.py#L38)
2. `dkm` don't need predownload
3. `superglue` don't need predownload
