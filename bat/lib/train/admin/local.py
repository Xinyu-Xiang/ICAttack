class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = './bat'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = './bat/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = './bat/pretrained_networks'
        self.got10k_val_dir = './got10k/val'
        self.lasot_lmdb_dir = './lasot_lmdb'
        self.got10k_lmdb_dir = './got10k_lmdb'
        self.trackingnet_lmdb_dir = './trackingnet_lmdb'
        self.coco_lmdb_dir = './coco_lmdb'
        self.coco_dir = './coco'
        self.lasot_dir = './lasot'
        self.got10k_dir = './got10k/train'
        self.trackingnet_dir = './trackingnet'
        self.depthtrack_dir = './depthtrack/train'
        self.lasher_dir = './lasher/trainingset'
        self.visevent_dir = './visevent/train'
