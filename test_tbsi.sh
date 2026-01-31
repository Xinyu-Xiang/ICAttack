export PYTHONPATH="./tbsi:$PYTHONPATH" \n
python ours_test.py --script_name tbsi --yaml_name vitb_256_tbsi_32x1_1e4_lasher_15ep_sot --ckp_path './checkpoints/tbsi/4.0_3.0mfiattack_60_stealth.pth' --dataset_name GTOT --dn_vi 4.0 --dn_ir 3.0
