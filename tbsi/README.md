# TBSI for RGB-T Tracking

Implementation of the paper [Bridging Search Region Interaction With Template for RGB-T Tracking](https://openaccess.thecvf.com/content/CVPR2023/papers/Hui_Bridging_Search_Region_Interaction_With_Template_for_RGB-T_Tracking_CVPR_2023_paper.pdf), CVPR 2023.

## Environment Installation

```
conda create -n tbsi python=3.8
conda activate tbsi
bash install.sh
```

## Project Paths Setup

Run the following command to set paths for this project

```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

After running this command, you can also modify paths by editing these two files

```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation

Put the tracking datasets in `./data`. It should look like:

```
${PROJECT_ROOT}
  -- data
      -- lasher
          |-- trainingset
          |-- testingset
          |-- trainingsetList.txt
          |-- testingsetList.txt
          ...
```

## Training

Download [ImageNet or SOT](https://pan.baidu.com/s/1U42J6b3g1htma0OvmXRQCw?pwd=at5b) pretrained weights and put them under `$PROJECT_ROOT$/pretrained_models`.

```
python tracking/train.py --script tbsi_track --config vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --save_dir ./output/vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --mode multiple --nproc_per_node 4
```

Replace `--config` with the desired model config under `experiments/tbsi_track`.

## Evaluation

Put the checkpoint into `$PROJECT_ROOT$/output/config_name/...` or modify the checkpoint path in testing code.

```
python tracking/test.py tbsi_track vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --dataset_name lasher_test --threads 6 --num_gpus 1

python tracking/analysis_results.py --tracker_name tbsi_track --tracker_param vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --dataset_name lasher_test

python tracking/test.py tbsi_track vitb_256_tbsi_32x1_1e4_lasher_15ep_sot --dataset_name lasher_test --threads 6 --num_gpus 1

python tracking/analysis_results.py --tracker_name tbsi_track --tracker_param vitb_256_tbsi_32x1_1e4_lasher_15ep_sot --dataset_name lasher_test
```

### Results on LasHeR testing set

| Model | Backbone | Pretraining | Precision | NormPrec | Success | FPS |                                                        Checkpoint |                                                        Raw Result |
| ----- | :------: | :---------: | :-------: | :------: | :-----: | :--: | ----------------------------------------------------------------: | ----------------------------------------------------------------: |
| TBSI  | ViT-Base |  ImageNet  |   64.3   |   60.8   |  51.0  | 36.2 | [download](https://pan.baidu.com/s/18MYRT4jkunIPklD02daFXA?pwd=y2rz) | [download](https://pan.baidu.com/s/1CP07T0VmtxPr6KcWqszY1w?pwd=6v3b) |
| TBSI  | ViT-Base |     SOT     |   70.2   |   66.5   |  56.5  | 36.2 | [download](https://pan.baidu.com/s/18MYRT4jkunIPklD02daFXA?pwd=y2rz) | [download](https://pan.baidu.com/s/1CP07T0VmtxPr6KcWqszY1w?pwd=6v3b) |

## Acknowledgments

Our project is developed upon [OSTrack](https://github.com/botaoye/OSTrack). Thanks for their contributions which help us to quickly implement our ideas.

## Citation

If our work is useful for your research, please consider cite:

```
@inproceedings{hui2023bridging,
  title={Bridging Search Region Interaction With Template for RGB-T Tracking},
  author={Hui, Tianrui and Xun, Zizheng and Peng, Fengguang and Huang, Junshi and Wei, Xiaoming and Wei, Xiaolin and Dai, Jiao and Han, Jizhong and Liu, Si},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13630--13639},
  year={2023}
}
```
## usage tips 
### written by wxy @5310_timer 2024.11.4
1. I downloaded the checkoints in the files whose path is "/data/wxy/TBSI/output/vitb_256_tbsi_32x1_1e4_lasher_15ep_sot/checkpoints/train/tbsi_track/vitb_256_tbsi_32x1_1e4_lasher_15ep_sot/TBSITrack_ep0015.pth.tar" instead of training.
2. I have changed codes in ../lib/test/evaluation/tracker.py line 75 and line 80 and rename the checkpoint tar file. 

# TBSI USAGE 
written by wxy 2024.11.6
## Environment
```
conda activate wxy
cd /data/wxy/TBSI
```
If you want to set the environment by yourself instead of using the conda env wxy, here are some tips:
1. To match the version of gpu in this server, you have had to change the version of torch from 1.9.0 given by install.sh to 1.8.0+cu111. So did torchvision. 
2. Set the python path.

## Path Setup
Set the data path in
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```
To do the testing, you only need to change the path in `lib/test/evaluation/local.py`.The lasher path setting is in line 36 and I have changed the LasHeR path to  '/data/xxy/dataset/LasHeR/'.

## Training
Though I didn't do the training job, I have checked and revised to make sure that the training file can run. But it may need 4 GPUs to run. Evert time I run the training, it will report an error about GPU memory.
```
python tracking/train.py --script tbsi_track --config vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --save_dir ./output/vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --mode multiple --nproc_per_node 4
```
Instead, I downloaded the checkpoint straightly and put it in '/data/wxy/TBSI/output/vitb_256_tbsi_32x1_1e4_lasher_15ep_sot/checkpoints/train/tbsi_track'. In fact, for the reason my computer doesn't have enough room, I only downloaded the checkpoints of SOT. The checkpoints of ImageNet can be 
downloaded from  [download](https://pan.baidu.com/s/18MYRT4jkunIPklD02daFXA?pwd=y2rz). If you want to add the checkpoints of ImageNet, you need to rename the .tar file to `TBSITrack_ep0015.pth.tar`

## Evaluation
```
CUDA_VISIBLE_DEVICE=3 python tracking/test.py tbsi_track vitb_256_tbsi_32x1_1e4_lasher_15ep_sot --dataset_name lasher_test --threads 6 --num_gpus 1

CUDA_VISIBLE_DEVICE=3 python tracking/analysis_results.py --tracker_name tbsi_track --tracker_param vitb_256_tbsi_32x1_1e4_lasher_15ep_sot --dataset_name lasher_test
```
Since only the SOT checkpoints are downloaded, run the above command.
test跑可以跑，但是会报 CUDA error: out of memory。
如果成功运行test.py，结果会保存在`/data/wxy/TBSI/output/test/tracking_results/tbsi_track/vitb_256_tbsi_32x1_1e4_lasher_15ep_sot`，查看该文件夹即可判断是否成功运行。确保test.py已经输出结果，analysis_results.py才能运行。


