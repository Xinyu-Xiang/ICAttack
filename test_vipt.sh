#!/bin/bash
export PYTHONPATH="./vipt:$PYTHONPATH"
python ours_test.py --dataset_name RGBT234 --ckp_path '/data/wxy/MFIAttack/ICAttack/checkpoints/vipt/6.5_6.0mfiattack_60_stealth.pth' --dn_vi 6.5 --dn_ir 6.0


