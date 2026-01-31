from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = './data/got10k_lmdb'
    settings.got10k_path = './data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = './data/itb'
    settings.lasot_extension_subset_path_path = './data/lasot_extension_subset'
    settings.lasot_lmdb_path = './data/lasot_lmdb'
    settings.lasot_path = './data/lasot'
    settings.network_path = './output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = './data/nfs'
    settings.otb_path = './data/otb'
    settings.prj_dir = './SDSTrack'
    settings.result_plot_path = './output/test/result_plots'
    settings.results_path = './output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = './output'
    settings.segmentation_path = './output/test/segmentation_results'
    settings.tc128_path = './data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = './data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = './data/trackingnet'
    settings.uav_path = './data/uav'
    settings.vot18_path = './data/vot2018'
    settings.vot22_path = './data/vot2022'
    settings.vot_path = './data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

