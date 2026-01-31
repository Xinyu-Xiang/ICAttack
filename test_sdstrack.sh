EXPERIMENT="cvpr2024_rgbt"
export PYTHONPATH="./SDSTrack:$PYTHONPATH" \n

python ours_test.py --script_name sdstrack --dataset_name LasHeR --yaml_name $EXPERIMENT  --ckp_path './checkpoints/sdstrack/5.0_4.0mfiattack_60_stealth.pth' --dn_vi 5.0 --dn_ir 4.0


