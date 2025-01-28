# TripleS for MTL-SCD

This is the official implementation for the paper on MTL-SCD.

**We present part of experimental codes and scripts, full codes and datasets would be made available after the review process.**

## Contents of Directory

* configs/: training config dir.
  * hrscd_512/
* dataset/: dataset & datatrarnsform
* loss/: loss function.
* model/: model structure.
* utils/:
* main.py: conduct training.

## Usage

### Create a conda virtual env and install necessary packages:

```shell
conda create -n MTL python=3.9
conda activate MTL
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.6 pillow scikit-learn scikit-image tqdm matplotlib segmentation-models-pytorch opencv -c pytorch -c conda-forge
```

### Quick Start on HRSCD dataset

To conduct MTL training on HSRCD dataset, run like:

```shell
python main.py --congfig_file ./configs/hrscd_512/scd4-full-v2.json
```

Soon you'll acquire trained weights in 'results/HSRCD/'.

## License

Code is released for non-commercial and research purposes **ONLY**. For commercial purposes, please contact the authors.
