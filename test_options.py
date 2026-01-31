import argparse

class TestOptions():
    """This class includes test options.
    """
    def initialize(self):
        parser = argparse.ArgumentParser(description='Run attacker on RGBT dataset.')

        parser.add_argument('--script_name', type=str, default='vipt', help='Name of tracking method(ostrack, prompt, ftuning).')
        parser.add_argument('--yaml_name', type=str, default='deep_rgbt', help='Name of tracking method.')  #
        parser.add_argument('--threads', default=2, type=int, help='Number of threads')
        parser.add_argument('--num_gpus', default=0, type=int, help='Number of gpus')
        parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
        parser.add_argument('--video', default='', type=str, help='specific video name')

        # self designed
        parser.add_argument('--dataset_name', type=str, default='RGBT234', help='Name of dataset (GTOT,RGBT234,LasHeR,VTUAV, NOT).')  
        parser.add_argument('--ckp_path', type=str, default='./checkpoints/bat/3.5_3.0mfiattack_50_stealth.pth')
        parser.add_argument('--draw', type=bool, default=False, help='whether graw the region on restored img') 
        parser.add_argument('--orginal_result_dir', type=str, default='./orginal_result/', help='orginal(clean) txt dir of rgbt tracker') 
        #  the results txt path:orginal_result_dir + tracker_name + dataset_name
        parser.add_argument('--save_folder', type=str, default='./output', help='the folder of save')  
        parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
        parser.add_argument('--dn_vi', type=float, default=6.5, help='vi image denormalized rate')
        parser.add_argument('--dn_ir', type=float, default=6.0, help='ir image denormalized rate')
        return parser.parse_args()
    
    