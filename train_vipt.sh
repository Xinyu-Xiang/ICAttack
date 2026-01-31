

export PYTHONPATH="./vipt:$PYTHONPATH"
python attack_video.py --tracker_name vipt --yaml_name deep_rgbt --dn_vi 6.5 --dn_ir 6.0 --n_epochs 61 --data_path ./dataset

