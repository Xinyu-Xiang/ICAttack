from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = './got10k_lmdb'
    settings.got10k_path = './got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = './itb'
    settings.lasot_extension_subset_path_path = './lasot_extension_subset'
    settings.lasot_lmdb_path = './lasot_lmdb'
    settings.lasot_path = './lasot'
    settings.network_path = '/data/wxy/IoUattack/bat/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = './nfs'
    settings.otb_path = './otb'
    settings.prj_dir = './bat'
    settings.result_plot_path = './bat/output/test/result_plots'
    settings.results_path = './bat/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = './bat/output'
    settings.segmentation_path = './bat/output/test/segmentation_results'
    settings.tc128_path = './TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = './tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = './trackingnet'
    settings.uav_path = './uav'
    settings.vot18_path = './vot2018'
    settings.vot22_path = './vot2022'
    settings.vot_path = './VOT2019'
    settings.youtubevos_dir = ''

    return settings

