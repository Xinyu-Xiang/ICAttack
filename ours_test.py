import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import sys
from os.path import join, isdir, abspath, dirname
import numpy as np
import argparse
from torchvision.utils import save_image
import math
from torch.utils.data import DataLoader
import multiprocessing
import torch
import random
from tqdm import tqdm
import cfg as cfg

from lib.train.data.processing_utils import sample_target

import time

from test_options import TestOptions
from utils import *
from data_utils import *
from module import *

args = TestOptions().initialize()

def genConfig(seq_path, set_type):
    if set_type == 'RGBT234':
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] == '.jpg'])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if os.path.splitext(p)[1] == '.jpg'])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')

        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

    elif set_type == 'GTOT':
        ############################################  have to refine #############################################
        RGB_img_list = sorted([seq_path + '/v/' + p for p in os.listdir(seq_path + '/v') if os.path.splitext(p)[1] in ['.png', '.bmp']])  
        T_img_list = sorted([seq_path + '/i/' + p for p in os.listdir(seq_path + '/i') if os.path.splitext(p)[1] in ['.png', '.bmp']])

        RGB_gt = np.loadtxt(seq_path + '/groundTruth_v.txt', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '/groundTruth_i.txt', delimiter=' ')

        x_min = np.min(RGB_gt[:,[0,2]],axis=1)[:,None]
        y_min = np.min(RGB_gt[:,[1,3]],axis=1)[:,None]
        x_max = np.max(RGB_gt[:,[0,2]],axis=1)[:,None]
        y_max = np.max(RGB_gt[:,[1,3]],axis=1)[:,None]
        RGB_gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

        x_min = np.min(T_gt[:,[0,2]],axis=1)[:,None]
        y_min = np.min(T_gt[:,[1,3]],axis=1)[:,None]
        x_max = np.max(T_gt[:,[0,2]],axis=1)[:,None]
        y_max = np.max(T_gt[:,[1,3]],axis=1)[:,None]
        T_gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)
    
    elif set_type == 'LasHeR':
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

    elif 'VTUAV' in set_type:
        RGB_img_list = sorted([seq_path + '/rgb/' + p for p in os.listdir(seq_path + '/rgb') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/ir/' + p for p in os.listdir(seq_path + '/ir') if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/rgb.txt', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '/ir.txt', delimiter=' ')
    elif set_type == 'NOT':
        RGB_img_list = sorted([seq_path + '/channel/' + p for p in os.listdir(seq_path + '/channel') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/channel2/' + p for p in os.listdir(seq_path + '/channel2') if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/groundtruth_rect.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/groundtruth_rect.txt', delimiter=',')

    return RGB_img_list, T_img_list, RGB_gt, T_gt


def box2crop_left_one(input, search_factor):
    '''
    将预测结果从GT映射到search上
    input: [n, 4]
    output: [n, 4]
    ''' 
    output = torch.zeros_like(input)
    # index = 0
    crop_z = math.ceil(math.sqrt(input[2] * input[3]) * search_factor)
    resize_factor =  256.0 / crop_z
    tmp = input[:]  * resize_factor
    output = tmp
    
    output[0] = 128.0  # center
    output[1] = 128.0  # center
    return output


def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W-margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H-margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2-x1)
    h = max(margin, y2-y1)
    return [x1, y1, w, h]


def set_seed(seed):
    random.seed(seed)  # set Python seed
    np.random.seed(seed)  # set NumPy seed
    torch.manual_seed(seed)  # set PyTorch seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # set CUDA seed
        torch.cuda.manual_seed_all(seed)  # multi-GPU，set all GPU seed
    torch.backends.cudnn.deterministic = True   
    torch.backends.cudnn.benchmark = False  



def run_sequence_attack_fullimage(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1,  debug=0, script_name='prompt'):
    # print(seq_name, seq_home, dataset_name, yaml_name, num_gpu, epoch, debug, script_name)
   
    save_dir = args.save_folder 
    
    # RGBT234 has lost sth
    if dataset_name =='RGBT234' and seq_name == 'orangeman1':
        import shutil
        tmp_name = os.path.join(seq_home, seq_name) + '/' + 'visible.txt'
        now_name = save_dir + '/' + 'save_txt'  + '/' + dataset_name  + '/' + script_name + '/'+ 'orangeman1.txt' 
        shutil.copy(tmp_name, now_name)
        return 
    
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    params = get_parameters(script_name, yaml_name)
    print('loading ckpt:', params.checkpoint) 
    mmtrack = get_tracker(script_name, params)
    tracker = ADV_RGBT(tracker=mmtrack)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise = ckpt['noise'].to(device)
    
    seq_path = seq_home + '/' + seq_name
    print('——————————Process sequence: '+seq_name +'——————————————')
    RGB_img_list, T_img_list, RGB_gt, T_gt = genConfig(seq_path, dataset_name)

    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)
    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
    result[0] = np.copy(RGB_gt[0])
    toc = 0
    
    pbar = tqdm(enumerate(zip(RGB_img_list, T_img_list)), position=0, leave=True)
    
    for frame_idx, (rgb_path, T_path) in pbar:
        pbar.set_postfix_str('frame %d: ' % (frame_idx))

        tic = cv2.getTickCount()

        image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))  # (H,W,6) RGB T concat
        img_vi=cv2.imread(rgb_path)
        img_ir=cv2.imread(T_path)
        vi_uint8 = cv2.cvtColor(img_vi, cv2.COLOR_BGR2RGB)
        ir_uint8 = img_ir

        if frame_idx == 0:
            # initialization
            tracker.initialize(image, RGB_gt[0].tolist())  # xywh

        elif frame_idx > 0:
            if RGB_gt[frame_idx][0] == 0 or RGB_gt[frame_idx][1] == 0 or RGB_gt[frame_idx][2] == 0 or RGB_gt[frame_idx][3] == 0:
                result[frame_idx] = result[frame_idx-1]
                continue
            else:
                with torch.no_grad():
                    gt = result[frame_idx-1]
                    
                    search_vi_arr, _, _ = sample_target(vi_uint8, gt, cfg.search_factor, output_sz=cfg.search_size)  
                    search_ir_arr, _, _ = sample_target(ir_uint8, gt, cfg.search_factor, output_sz=cfg.search_size)
                    search_tensor = img2tensor(search_vi_arr).cuda()
                    search_x_tensor = img2tensor(search_ir_arr).cuda()

                    flag = 'fusion_attack'
                    if flag == 'fusion_attack':
                        # noramlize
                        search_tensor = search_tensor/255.0
                        search_x_tensor = search_x_tensor/255.0
                        noise = noise/255.0

                        # generate noise
                        # [0,1] -> [-1, 1]    
                        vi_adv_noise, ir_adv_noise = attack_model(search_tensor ,search_x_tensor, noise)  # add clean get adv [-1, 1]

                        # Restore
                        search_tensor = search_tensor*255.0
                        search_x_tensor = search_x_tensor*255.0
                        noise = noise*255.0
                        

                        vi_adv_noise = denormalize_noise(vi_adv_noise, args.dn_vi) 
                        ir_adv_noise = denormalize_noise_ir(ir_adv_noise, args.dn_ir)
                        vi_adv_search = vi_adv_noise + search_tensor
                        ir_adv_search = ir_adv_noise + search_x_tensor
                        vi_adv_search = vi_adv_search.clamp(0, 255)  # clip range [0, 255]
                        ir_adv_search = ir_adv_search.clamp(0, 255)  # clip range [0, 255]

                        
                        region, _ = tracker.track_adv(image, search_tensor, search_x_tensor, vi_adv_search, ir_adv_search, gt)  # xywh
                        result[frame_idx] = np.array(region)


                save_txt_path = save_dir +'/' + 'save_txt' + '/' + dataset_name +  '/' + script_name + '/'
                if not os.path.exists(save_txt_path):
                    os.makedirs(save_txt_path)
                np.savetxt(save_txt_path + seq_name + '.txt', result, delimiter=' ')

                if args.draw:
                    # save_img
                    x_adv_vi = cv2.cvtColor(tensor_to_cv2(vi_adv_search.squeeze(dim=0)), cv2.COLOR_RGB2BGR)
                    x_crop_vi = cv2.cvtColor(tensor_to_cv2(search_tensor.squeeze(dim=0)), cv2.COLOR_RGB2BGR)
                    x_adv_ir = tensor_to_cv2(ir_adv_search.squeeze(dim=0))
                    x_crop_ir = tensor_to_cv2(search_x_tensor.squeeze(dim=0))
                    x_adv_ir = cv2.cvtColor(tensor_to_cv2(ir_adv_search.squeeze(dim=0)), cv2.COLOR_RGB2BGR)
                    x_crop_ir = cv2.cvtColor(tensor_to_cv2(search_x_tensor.squeeze(dim=0)), cv2.COLOR_RGB2BGR)
                    noise = cv2.absdiff(x_adv_vi, x_crop_vi)
                    delta = x_adv_vi - x_crop_vi
                    delta_ir = x_adv_ir - x_crop_ir

                    # restore:turn the search region with perturbation back to origion img(only for visible)
                    restored_vi = restore_noisy_crop_to_original(original_img=img_vi,
                                                                noisy_crop=x_adv_vi,
                                                                box_extract=gt,
                                                                search_area_factor=cfg.search_factor,
                                                                output_sz=cfg.search_size)
                    restored_ir = restore_noisy_crop_to_original(original_img=img_ir,
                                                                noisy_crop=x_adv_ir,
                                                                box_extract=gt,
                                                                search_area_factor=cfg.search_factor,
                                                                output_sz=cfg.search_size)

                    seq_path_clean = args.orginal_result_dir + script_name + '/' + dataset_name 
                    region_clean = np.loadtxt(seq_path_clean + '/' + seq_name + '.txt', delimiter=' ')
                    # tbsi_origin-las'\t' other' '
                    search_see_bbox = draw_boxes_on_cv2_image_compare(restored_vi, RGB_gt[frame_idx], region, region_clean[frame_idx])
                    search_see_bbox_ir = draw_boxes_on_cv2_image_compare(restored_ir, RGB_gt[frame_idx], region, region_clean[frame_idx])

                    
                    if not os.path.exists(save_dir +'/' + dataset_name + '/' + script_name + '/' + seq_name +'/'):
                        os.makedirs(save_dir +'/' + dataset_name  + '/' + script_name + '/' + seq_name +'/')
                    cv2.imwrite(save_dir +'/' + dataset_name + '/' + script_name + '/' + seq_name +'/' + seq_name + str(frame_idx) + '.png', search_see_bbox)
                    cv2.imwrite(save_dir +'/' + dataset_name + '/' + script_name + '/' + seq_name +'/' + seq_name + str(frame_idx) + '_ir.png', search_see_bbox_ir)
            

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    print('{} , fps:{}'.format(seq_name, frame_idx / toc))



class ADV_RGBT(object): 
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        
        init_info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB):
        '''TRACK'''  # 其实这里输入的不是RGB，而是concat(RGB,T)
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        return pred_bbox, pred_score

    def track_adv(self, image, search_tensor, search_x_tensor, vi_adv, ir_adv, gt):
        '''TRACK'''  
        outputs = self.tracker.track_adv(image, search_tensor, search_x_tensor, vi_adv, ir_adv, gt)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        return pred_bbox, pred_score


if __name__ == '__main__':
    args = TestOptions().initialize()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name

    # path initialization
    seq_list = None
    if dataset_name == 'GTOT':
        seq_home = '/data/dataset/GTOT'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
        seq_list.sort()
    elif dataset_name == 'RGBT234':
        seq_home = "/data/dataset/RGBT234"
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
        seq_list.sort()
    elif dataset_name == 'LasHeR':
        seq_home = "/data/dataset/LasHeR/TestingSet/testingset/"
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
        seq_list.sort()
    elif dataset_name == 'VTUAVST':
        seq_home = '/mnt/6196b16a-836e-45a4-b6f2-641dca0991d0/VTUAV/test/short-term'
        with open(join(join(seq_home, 'VTUAV-ST.txt')), 'r') as f:
            seq_list = f.read().splitlines()
    elif dataset_name == 'VTUAVLT':
        seq_home = '/mnt/6196b16a-836e-45a4-b6f2-641dca0991d0/VTUAV/test/long-term'
        with open(join(seq_home, 'VTUAV-LT.txt'), 'r') as f:
            seq_list = f.read().splitlines()
    elif dataset_name == 'NOT':
        seq_home = '/data/dataset/NOT/test/NOT156_test/'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
    else:
        raise ValueError("Error dataset!")

    start = time.time()

    attack_model = MFIAttack().cuda()

    ckpt = torch.load(args.ckp_path)
    attack_model.load_state_dict(ckpt['model_state_dict'])
    attack_model.eval()
    noise = ckpt['noise']

    set_seed(args.seed)

    if args.video != '':
        # test_one
        run_sequence_attack_fullimage('boywalkinginsnow2', seq_home, dataset_name, yaml_name, args.num_gpus, args.debug, args.script_name)
    else:
        # test_all
        seq_list = [args.video] if args.video != '' else seq_list
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus,  args.debug, args.script_name) for s in seq_list]
        # parallel
        for seqlist in sequence_list:
            run_sequence_attack_fullimage(*seqlist)
    


    
      
    
              