# TripleS: Mitigating multi-task learning conflicts for semantic change detection in high-resolution remote sensing imagery

This is the official implementation of **[CVEO](https://github.com/cveo)**'s recent research paper published on [ISPRS Journal of Photogrammetry and Remote Sensing](https://www.sciencedirect.com/science/article/pii/S0924271625003776).

## Abstract

> Periodical earth observation from multi-temporal high spatial resolution remote sensing imagery (RSI) offers valuable insights into the complex dynamics of land surface changes. Semantic change detection (SCD), cooperating with deep learning (DL) architectures, has evolved from binary change detection (BCD) into an effective technique capable of not only identifying change locations but also specifying land-cover and land-use (LCLU) categories. Recent advancements suggest that SCD can be modeled as a multi-task learning (MTL) framework, involving multiple branches for individual subtasks to process dual RSI inputs, and optimized through joint training. However, limitations remain in the inadequate interactions between bi-temporal branches and semantic-change branches, as well as the pervasive gradient conflicts among subtasks within MTL frameworks, which can lead to counterbalanced performances. To address the above limitations, we propose an MTL-oriented SCD model (MOSCD), which mutually enhances bi-temporal features, while ensuring that representations across the subtask branches are coherently correlated. Furthermore, the TripleS framework is designed to enhance the optimization of the MTL framework through counteracting the conflicting subtask objectives, which incorporates three novel schemes: Stepwise multi-task optimization, Selective parameter binding, and Scheduling for dynamically training MTL bindings. Extensive experiments conducted on three full-coverage land-cover SCD datasets, including one public dataset (HRSCD) and two self-constructed datasets (SC-SCD7 and CC-SCD5), demonstrate that the MOSCD enhanced with TripleS outperforms eleven existing SCD methods and three MTL methods by up to 21.17% on SeK metrics. The robust performances over diverse landscapes and transferability on other componentized benchmarks validate that the MOSCD trained with TripleS is a practicable tool for detecting subtle land-cover changes from high spatial resolution RSI data.

## Method

* MOSCD model optimized with the proposed MTL framework TripleS.

<div align="center">
<img src="./docs/MOSCD.png" width="600"/>
</div>

* Stepwise optimization scheme cooperating with Selective parameter binding.

<div align="center">
<img src="./docs/triS-SS.png" width="600"/>
</div>

* Different schedulings for training MTL bindings.

<div align="center">
<img src="./docs/scheduling.png" width="600"/>
</div>

## Preparation

#### Dependencies and Installation

```shell
conda create -n triples python=3.9
conda activate triples
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.6 pillow numpy tqdm matplotlib segmentation-models-pytorch opencv -c pytorch -c conda-forge
pip install segmentation-models-pytorch==0.3.4
```

#### Full-coverage SCD Datasets

Download the dataset and perform pre-processing according to the settings described in the paper:
* [HRSCD](https://ieee-dataport.org/open-access/hrscd-high-resolution-semantic-change-detection-dataset) dataset
* The proposed SC-SCD7 and CC-SCD5 datasets [Coming soon]

and organize the dataset according to the `.txt` files in the `/txt` directory.


## Model training and evaluation

* Jointly training:

```shell
python train_jointly.py --congfig_file ./configs/HRSCD/MOSCD_triS.json
```

* Training with TripleS-A:

```shell
python train_tripleS_A.py --congfig_file ./configs/SCSCD7/MOSCD_triS.json
```

* Training with TripleS-C:

```shell
python train_tripleS_C.py --congfig_file ./configs/HRSCD/MOSCD_triS.json
```

Soon you'll acquire trained weights in `trained_models/`.



* Inference and evaluation:

```shell
python infereval.py --congfig_file ./configs/CCSCD5/MOSCD_triS.json --ckpt_path ./trained_models/ccscd5_512/MOSCD_triS/MOSCD_triS_1/state/checkpoint.pth.tar
```

Results can be found in `infer/`.

## Results

<div align="center">
<img src="./docs/exp-scscd7.png" />
</div>

<div align="center">
<img src="./docs/exp-hrscd.png" width="600"/>
</div>

## Citation

Please consider citing the following paper if you used this project and dataset in your research.

```shell
@article{TAN2025374,
    title = {TripleS: Mitigating multi-task learning conflicts for semantic change detection in high-resolution remote sensing imagery},
    journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
    volume = {230},
    pages = {374-401},
    year = {2025},
    issn = {0924-2716},
    doi = {https://doi.org/10.1016/j.isprsjprs.2025.09.019},
    url = {https://www.sciencedirect.com/science/article/pii/S0924271625003776},
    author = {Xiaoliang Tan and Guanzhou Chen and Xiaodong Zhang and Tong Wang and Jiaqi Wang and Kui Wang and Tingxuan Miao},
    keywords = {Semantic change detection, Remote sensing, Multi-task learning, Deep learning, Land-cover and land-use}
}
```

### Acknowledgement

Code is released for non-commercial and research purposes **ONLY**. For commercial purposes, please contact the authors.

### Reference

Appreciate the work from the following repositories: [ClearSCD](https://github.com/tangkai-RS/ClearSCD).
