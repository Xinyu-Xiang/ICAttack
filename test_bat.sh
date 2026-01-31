export PYTHONPATH="./bat:$PYTHONPATH" \n
python ours_test.py --script_name bat --dataset_name RGBT234 --yaml_name rgbt --ckp_path './checkpoints/bat/3.5_3.0mfiattack_50_stealth.pth' --dn_vi 3.5 --dn_ir 3.0
