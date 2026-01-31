import numpy as np
import math
from tqdm import tqdm
import os
import cv2
import argparse
import warnings
warnings.filterwarnings("ignore")
import time
import random

import torch
from torch.utils.data import DataLoader

from lib.train.data.processing_utils import sample_target
from utils import *
from data_utils import *
from module import *


def set_seed(seed):
    random.seed(seed)  # set Python seed
    np.random.seed(seed)  # set NumPy seed
    torch.manual_seed(seed)  # set PyTorch seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # set CUDA seed
        torch.cuda.manual_seed_all(seed)  # multi-GPU，set all GPU seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    seed = args.seed + worker_id
    np.random.seed(seed)
    random.seed(seed)


def chooseModule(str):
    if str == 'fusion_attack':
        # fusion_attack
        fusion_attack = MFIAttack().cuda()
        return fusion_attack


class ADV_RGBT(object): 
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        
        init_info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB):
        '''TRACK'''  
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        return pred_bbox, pred_score

    def track_adv(self, image, search_tensor, search_x_tensor, vi_adv, ir_adv, gt):
        '''TRACK for adv'''  
        outputs = self.tracker.track_adv_ours(image, search_tensor, search_x_tensor, vi_adv, ir_adv, gt)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        response_clean = outputs['response_clean']
        response = outputs['response']
        clean_bbox = outputs['clean_bbox']
        return pred_bbox, pred_score, response_clean, response, clean_bbox


def chooseTracker(str):
    """
    input str
    return a tracker
    """
    params = get_parameters(str, args.yaml_name)
    print('loading ckpt:', params.checkpoint) 
    mmtrack = get_tracker(str, params)
    mmtrack.network.eval()
    tracker = ADV_RGBT(tracker=mmtrack)
    return tracker, mmtrack
    

def train(args):
    # hyper
    attack_loss_name, attack_param = [], []
    for key, value in args.attack_type.items():
        print(f"attack_loss_name:{key}, hyper_param:{value}")
        attack_loss_name.append(key)
        attack_param.append(value)

    # dataset
    dataset = RGB234_dataset_full_image(max_num=args.frame_num, data_path=args.data_path)  # 5 is fixed if change, need to change the dataset
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, worker_init_fn=worker_init_fn, generator=g)
    dataset_size = len(dataloader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print('The number of training images = %d' % dataset_size)

    # choose victim tracker
    tracker_name = args.tracker_name
    tracker, mmtrack = chooseTracker(tracker_name)  

    # ours perturbation
    flag = 'fusion_attack' 
    if flag == 'fusion_attack':
        model = chooseModule('fusion_attack')  
    
    # learning 
    start_lr = 0.0002
    iter_max = 30000
    power = 0.9
    iter_batch = 0
    
    
    # noise
    mean = 127.5  
    std = 25.0    
    noise = torch.nn.Parameter(torch.normal(mean=mean, std=std, size=(1, 3, 256, 256), device=device).clamp(0, 255))
    

    # optimizer
    if flag == 'fusion_attack':
        optimizer = torch.optim.Adam(list(model.parameters()) + [noise], lr=start_lr, betas=(0.5, 0.999), amsgrad=True)
        for param in mmtrack.network.parameters():
            # make sure the tracker parameters are not updated
            param.requires_grad = False


    # start training
    torch.cuda.empty_cache()
    start_epoch = 0
    # n_epochs = math.floor(iter_max / dataset_size)
    n_epochs = args.n_epochs
    print("n_epochs:{}".format(n_epochs))
    
    L1_loss = torch.nn.L1Loss().cuda()


    # start training
    for epoch in range(start_epoch, n_epochs):
        # torch.autograd.set_detect_anomaly(True)
        dataset.set_epoch(epoch)


        for i, (RGB_img_list, RGB_gt, T_img_list, T_gt) in enumerate(dataloader):
            iter_batch = iter_batch + 1
            RGB_gt = RGB_gt.squeeze(dim=0)
            T_gt =T_gt.squeeze(dim=0)
            if len(RGB_img_list) == RGB_gt.shape[0]:
                result = np.zeros_like(RGB_gt)
                result_clean = np.zeros_like(RGB_gt)
            else:
                result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
            result[0] = np.copy(RGB_gt[0])
            result_clean[0] = np.copy(RGB_gt[0])
            pbar = tqdm(enumerate(zip(RGB_img_list, T_img_list)), position=0, leave=True)
            
            response_fake_gt_batch = torch.zeros((args.frame_num-1, 1, 16, 16)).cuda()  # fake gt response

            vi_adv_search_batch = []  # visible adv search
            ir_adv_search_batch = []  # infrared adv search
            search_tensor_batch = []  # visible search
            search_x_tensor_batch = []  # infrared search
            response_adv_batch = []  # adv response
            response_clean_batch = []# clean response
            response_center_batch = []  # center response clear 
            vi_draw_batch = []
            # response_fake_gt_batch = []  # fake gt response

            for frame_idx, (rgb_path, T_path) in pbar:
                rgb_path, T_path = rgb_path[0], T_path[0]  
                image = get_x_frame(rgb_path, T_path, dtype='rgbrgb')  # (H,W,6) RGB T concat
                img_vi=cv2.imread(rgb_path)
                img_ir=cv2.imread(T_path)
                vi_uint8 = cv2.cvtColor(img_vi, cv2.COLOR_BGR2RGB)
                vi_draw = torch.from_numpy(vi_uint8).permute(2, 0, 1).float()
                vi_draw_batch.append(vi_draw)
                ir_uint8 = img_ir

                if frame_idx == 0:
                    # initialization
                    tracker.initialize(image, RGB_gt[0].tolist())  # xywh

                elif frame_idx > 0:
                    if RGB_gt[frame_idx][0] == 0 or RGB_gt[frame_idx][1] == 0 or RGB_gt[frame_idx][2] == 0 or RGB_gt[frame_idx][3] == 0:
                        result[frame_idx] = RGB_gt[frame_idx]
                        break
                    else:
                        gt = result[frame_idx-1]  
                        search_vi_arr, _, _ = sample_target(vi_uint8, gt, cfg.search_factor, output_sz=cfg.search_size)  
                        search_ir_arr, _, _ = sample_target(ir_uint8, gt, cfg.search_factor, output_sz=cfg.search_size)
                        search_tensor = img2tensor(search_vi_arr).cuda()
                        search_x_tensor = img2tensor(search_ir_arr).cuda()
                    
                    noise_norm = noise/255.0

                    flag = 'fusion_attack'
                    if flag == 'fusion_attack':
                        # noramlize
                        search_tensor_norm = search_tensor/255.0
                        search_x_tensor_norm = search_x_tensor/255.0
                        
                        # generate noise
                        # [0,1] -> [-1, 1]    
                        vi_adv_noise, ir_adv_noise = model(search_tensor_norm, search_x_tensor_norm, noise_norm)  # add clean get adv [-1, 1]
                        noise_norm = np.concatenate((vi_adv_noise.cpu().detach().numpy(), ir_adv_noise.cpu().detach().numpy()), axis=0)
                        #[bs*n,3,256,256]
                        # Restore
                        search_tensor_restore = search_tensor_norm*255.0
                        search_tensor_batch.append(search_tensor_restore[0, :, :, :])
                        search_x_tensor_restore = search_x_tensor_norm*255.0
                        search_x_tensor_batch.append(search_x_tensor_restore[0, :, :, :])

                        
                        vi_adv_noise = denormalize_noise(vi_adv_noise, args.dn_vi)  # turn to [0, 255] range image
                        ir_adv_noise = denormalize_noise_ir(ir_adv_noise, args.dn_ir)
                        vi_adv_search = vi_adv_noise + search_tensor_restore
                        ir_adv_search = ir_adv_noise + search_x_tensor_restore
                        vi_adv_search = vi_adv_search.clamp(0, 255)  # clip range [0, 255]
                        ir_adv_search = ir_adv_search.clamp(0, 255)  # clip range [0, 255]
                        vi_adv_search_batch.append(vi_adv_search[0, :, :, :])
                        ir_adv_search_batch.append(ir_adv_search[0, :, :, :])
                        
                        region, score, response_clean, response, region_clean = tracker.track_adv(image, search_tensor, search_x_tensor, vi_adv_search, ir_adv_search, gt)  # xywh
                        result[frame_idx] = np.array(region)
                        result_clean[frame_idx] = np.array(region_clean)
                        
                        response_adv_batch.append(response[0,:,:,:])
                        response_clean_batch.append(response_clean[0,:,:,:])

                        response_center_batch.append(zero_circle_region(response, r = 2)[0,:,:,:])
                        

            
            if RGB_gt[frame_idx][0] == 0 or RGB_gt[frame_idx][1] == 0 or RGB_gt[frame_idx][2] == 0 or RGB_gt[frame_idx][3] == 0:
                continue
            vi_adv_search_batch = torch.stack(vi_adv_search_batch, dim=0)
            ir_adv_search_batch = torch.stack(ir_adv_search_batch, dim=0)
            search_tensor_batch = torch.stack(search_tensor_batch, dim=0)
            search_x_tensor_batch = torch.stack(search_x_tensor_batch, dim=0)
            response_adv_batch = torch.stack(response_adv_batch, dim=0)
            response_clean_batch = torch.stack(response_clean_batch, dim=0)
            response_center_batch = torch.stack(response_center_batch, dim=0)
            vi_draw_batch = torch.stack(vi_draw_batch, dim=0)

            # loss fuction
            # ssim loss
            loss_recon_vi = L1_loss(vi_adv_search_batch, search_tensor_batch)
            loss_recon_ir = 1-calculate_ssim_color(ir_adv_search_batch, search_x_tensor_batch)

            
            # response loss(clear center response)
            loss_response_center = L1_loss(response_adv_batch, response_center_batch)

            loss = attack_param[0] * loss_recon_vi + attack_param[1] * loss_recon_ir + attack_param[2] * loss_response_center


            msg = f"Total loss:{loss.detach().cpu().numpy():8.5f}," \
                f"VI recon loss:{attack_param[0] * loss_recon_vi.detach().cpu().numpy():8.5f}," \
                f"IR recon loss:{attack_param[1] * loss_recon_ir.detach().cpu().numpy():8.5f}," \
                f"Attack loss:{attack_param[2] * loss_response_center.detach().cpu().numpy():8.5f}," \
                f"learining rate:{optimizer.param_groups[0]['lr']}," \
                f"iter:{iter_batch}," \
                f"epoch:{epoch}"     
            print(msg)      

            loss.backward()
            optimizer.step()

            del vi_adv_search_batch, ir_adv_search_batch, search_tensor_batch, search_x_tensor_batch, response_adv_batch, response_center_batch, \
            response_clean_batch, vi_draw_batch

            optimizer.zero_grad()
            optimizer.param_groups[0]['lr'] = start_lr * (1 - iter_batch / iter_max) ** power
            torch.cuda.empty_cache()

        if epoch == n_epochs-1 or epoch % 10 == 0:
            ckp_save_path = f"./checkpoints/{args.tracker_name}/{attack_param[0]}{attack_loss_name[0]}_{attack_param[1]}{attack_loss_name[1]}_{attack_param[2]}{attack_loss_name[2]}/"
            if not os.path.exists(ckp_save_path):
                os.makedirs(ckp_save_path)
            torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),\
                'noise': noise.detach().cpu(), 'epoch': epoch}, ckp_save_path + f"mfiattack_{epoch}.pth")
        optimizer.zero_grad()
        



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run attacker on RGBT dataset.')
    # self designed
    parser.add_argument('--data_path', type=str, default='/data/dataset/RGBT234', help='Path of dataset RGBT234.')  
    parser.add_argument('--tracker_name', default='bat', help='different victim tracker')
    parser.add_argument('--attack_type', type=dict, default={'vi_reson': 1.0, 'ir_reson': 1.0, 'response_center':10000.0},  help='different attack type,  10000 and 10')
    parser.add_argument('--batch_size', default=1, help='batch_size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--frame_num', type=int, default=10, help='Train frame length')
    parser.add_argument('--yaml_name', type=str, default='deep_rgbt', help='Name of tracking method.')
    parser.add_argument('--dn_vi', type=float, default=6.5, help='vi image denormalized rate')
    parser.add_argument('--dn_ir', type=float, default=6.0, help='ir image denormalized rate')
    parser.add_argument('--n_epochs', type=int, default=61, help='the training number of epochs')

    args = parser.parse_args()

    set_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    train(args)  