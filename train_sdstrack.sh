EXPERIMENT="cvpr2024_rgbt"
export PYTHONPATH="./SDSTrack:$PYTHONPATH" \n
python attack_video.py --tracker_name sdstrack --yaml_name $EXPERIMENT --dn_vi 5.0 --dn_ir 4.0 --n_epochs 61 --data_path ./dataset





