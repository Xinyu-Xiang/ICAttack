class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = './vipt'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = './vipt/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = './vipt/pretrained_networks'
        self.got10k_val_dir = './RGBT234/got10k/val'
        self.lasot_lmdb_dir = './RGBT234/lasot_lmdb'
        self.got10k_lmdb_dir = './RGBT234/got10k_lmdb'
        self.trackingnet_lmdb_dir = './RGBT234/trackingnet_lmdb'
        self.coco_lmdb_dir = './RGBT234/coco_lmdb'
        self.coco_dir = './RGBT234/coco'
        self.lasot_dir = './RGBT234/lasot'
        self.got10k_dir = './RGBT234/got10k/train'
        self.trackingnet_dir = './RGBT234/trackingnet'
        self.depthtrack_dir = './RGBT234/depthtrack/train'
        self.lasher_dir = './RGBT234/lasher/trainingset'
        self.visevent_dir = './RGBT234/visevent/train'
