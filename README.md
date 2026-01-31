# ICAttack
Adversarial perturbation for RGB-T tracking via intra-modal excavation and cross-modal collusion

This is the official PyTorch implementation of "[Adversarial perturbation for RGB-T tracking via intra-modal excavation and cross-modal collusion](https://www.sciencedirect.com/science/article/pii/S156625352600062X)"

## Framework
![The overall framework of the proposed ICAttack algorithm.](https://github.com/Xinyu-Xiang/ICAttack/assets/framework.jpg)

## Usage

### Install the environment

Create and activate a conda environment:

```
conda create -n icattack python=3.8
conda activate icattack
```

Install the required packages:

```
cd SDSTrack
bash install_sdstrack.sh
```

### Data Preparation

Put the training datasets in `./dataset/`. It should look like:

```
$<PROJECT_ROOT>
-- dataset
    -- RGBT234
        |-- afterrain
        |-- aftertree
        ...
    -- LasHeR
      --testingset
        |--10runone
        |--11leftboy
        ...
    --GTOT
        |--BlackCar
        |--BlackSwan1
        ...

```
### Prepare the tracker

Download the pre-trained models for each tracker (ViPT, TBSI, BAT, SDSTrack) following their official instructions and place them in the correct directories.

### Training
The bash scripts have been modified accordingly. You can run the relevant scripts for training. The hyperparameters have been fixed to the values reported in the paper.
You have to revise the `data_path` on the `args`.
#### Train on ViPT
```
sh train_vipt.sh
```
#### Train on BAT
```
sh train_bat.sh
```
#### Train on SDSTrack
```
sh train_sdstrack.sh
```
#### Train on TBSI
```
sh train_tbsi.sh
```
### Test
Test scripts are also prepared, as shown below:
```
sh test_vipt.sh
sh test_bat.sh
sh test_tbsi.sh
sh test_sdstrack.sh
```
Please configure the arguments (`args`) appropriately, including the target dataset name (`dataset_name`), the path to the test weights (`ckp_path`), and the folder where tracking results will be saved (`save_folder`). 

If you wish to visualize the qualitative tracking results after adding noise (e.g., restoring the noisy search region back to the original image and comparing its tracking outcome with that of a clean sample), please correctly set the draw flag(`draw`) and provide the path to the pre-downloaded tracking results obtained on clean samples (`original_result_dir`). These clean-sample tracking results can typically be downloaded from the official releases provided by the respective trackers.

For more argument options, please refer to `test_options.py`.


## Acknowledgments
We extend our sincere gratitude to the authors and developers of the following tracking libraries, which were instrumental in this project:

*   [VIPT](https://github.com/jiawen-zhu/ViPT)
*   [BAT](https://github.com/SparkTempest/BAT)
*   [TBSI](https://github.com/RyanHTR/TBSI)
*   [SDSTrack](https://github.com/hoqolo/SDSTrack)


Their work has significantly contributed to our research and implementation.

## If this work is helpful to you, please cite it as：
```
@article{xiang2026adversarial,
  title={Adversarial perturbation for RGB-T tracking via intra-modal excavation and cross-modal collusion},
  author={Xiang, Xinyu and Wu, Xuying and Li, Shengxiang and Yan, Qinglong and Zou, Tong and Zhang, Hao and Ma, Jiayi},
  journal={Information Fusion},
  pages={104183},
  year={2026},
  publisher={Elsevier}
}
```
