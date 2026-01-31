
import torch
from torch.utils.data.dataset import Dataset
import os
import glob
import cv2
import numpy as np

from utils import *

dataset_dir = './dataset/RGBT234_reproduce_interval2'

def img2tensor(img_arr):
    '''float64 ndarray (H,W,3) ---> float32 torch tensor (1,3,H,W)'''
    img_arr = img_arr.astype(np.float32)
    img_arr = img_arr.transpose(2, 0, 1) # channel first
    img_arr = img_arr[np.newaxis, :, :, :]
    init_tensor = torch.from_numpy(img_arr)  # (1,3,H,W)
    return init_tensor
def normalize(im_tensor):
    '''(0,255) ---> (-1,1)'''
    im_tensor = im_tensor / 255.0
    im_tensor = im_tensor - 0.5
    im_tensor = im_tensor / 0.5
    return im_tensor
def tensor2img(tensor):
    '''(0,255) tensor ---> (0,255) img'''
    '''(1,3,H,W) ---> (H,W,3)'''
    tensor = tensor.squeeze(0).permute(1,2,0)
    img = tensor.cpu().numpy().clip(0,255).astype(np.uint8)
    return img

class RGB234_dataset(Dataset):
    def __init__(self, max_num=15):
        # dataset_dir is common_path import train_set_path_
        folders = sorted(os.listdir(dataset_dir))
        folders.remove('init_gt_visible.txt')
        folders.remove('init_gt_infrared.txt')
        self.folders_list = [os.path.join(dataset_dir,folder) for folder in folders]
        self.max_num = max_num
    def __getitem__(self, index):

        cur_folder = self.folders_list[index]
        img_paths = sorted(glob.glob(os.path.join(cur_folder,'visible','*.jpg')))  # 中间加入modal
        '''get init frame tensor'''
        init_frame_path = img_paths[0]
        init_frame_arr = cv2.imread(init_frame_path)
        # BGR2RGB
        init_frame_arr = cv2.cvtColor(init_frame_arr, cv2.COLOR_BGR2RGB)
        init_tensor = img2tensor(init_frame_arr)
        '''get search regions' tensor'''
        search_region_paths = img_paths[1:self.max_num+1]
        num_search = len(search_region_paths)
        search_tensor = torch.zeros((num_search,3,256,256),dtype=torch.float32)
        for i in range(num_search):
            search_arr = cv2.imread(search_region_paths[i])
            #BGR2RGB
            search_arr = cv2.cvtColor(search_arr, cv2.COLOR_BGR2RGB)
            search_tensor[i,:,:,:] = img2tensor(search_arr)
        '''Note: we don't normalize these tensors here, 
        but leave normalization to training process'''
        gt_file_visible = os.path.join(dataset_dir,'init_gt_visible.txt')
        gt_arr_visible = np.loadtxt(gt_file_visible,dtype=np.float64,delimiter=',')
        gt_init = gt_arr_visible[index]

        x_cur_folder = self.folders_list[index]
        x_img_paths = sorted(glob.glob(os.path.join(x_cur_folder,'infrared','*.jpg')))
        '''get init frame tensor'''
        x_init_frame_path = x_img_paths[0]
        x_init_frame_arr = cv2.imread(x_init_frame_path)
        # BGR2RGB
        x_init_frame_arr = cv2.cvtColor(x_init_frame_arr, cv2.COLOR_BGR2RGB)
        init_x_tensor = img2tensor(x_init_frame_arr)
        '''get search regions' tensor'''
        x_search_region_paths = x_img_paths[1:self.max_num+1] # to avoid being out of GPU memory
        x_num_search = len(x_search_region_paths)
        search_x_tensor = torch.zeros((x_num_search,3,256,256),dtype=torch.float32)
        for i in range(x_num_search):
            x_search_arr = cv2.imread(x_search_region_paths[i])
            #BGR2RGB
            x_search_arr = cv2.cvtColor(x_search_arr, cv2.COLOR_BGR2RGB)
            search_x_tensor[i,:,:,:] = img2tensor(x_search_arr)
        '''Note: we don't normalize these tensors here, 
        but leave normalization to training process'''
        

        gt = np.loadtxt(os.path.join(cur_folder,'visible', 'visible.txt'), dtype=np.float64,delimiter=' ')[0:self.max_num]
        gt_x = np.loadtxt(os.path.join(x_cur_folder,'infrared', 'infrared.txt'), dtype=np.float64,delimiter=' ')[0:self.max_num]
        return init_tensor, search_tensor, init_x_tensor, search_x_tensor, gt_init, gt, gt_x
    
    def __len__(self):
        return len(self.folders_list)


class RGB234_dataset_wuli_point(Dataset):
    def __init__(self, max_num=15):

        folders = sorted(os.listdir(dataset_dir))
        folders.remove('init_gt_visible.txt')
        folders.remove('init_gt_infrared.txt')
        self.folders_list = [os.path.join(dataset_dir,folder) for folder in folders]
        self.max_num = max_num
    def __getitem__(self, index):

        cur_folder = self.folders_list[index]
        img_paths = sorted(glob.glob(os.path.join(cur_folder,'visible','*.jpg')))
        '''get init frame tensor'''
        init_frame_path = img_paths[0]
        init_frame_arr = cv2.imread(init_frame_path)
        # BGR2RGB
        init_frame_arr = cv2.cvtColor(init_frame_arr, cv2.COLOR_BGR2RGB)
        init_tensor = img2tensor(init_frame_arr)
        '''get search regions' tensor'''
        search_region_paths = img_paths[1:self.max_num+1] # to avoid being out of GPU memory
        num_search = len(search_region_paths)
        search_tensor = torch.zeros((num_search,3,256,256),dtype=torch.float32)
        for i in range(num_search):
            search_arr = cv2.imread(search_region_paths[i])
            #BGR2RGB
            search_arr = cv2.cvtColor(search_arr, cv2.COLOR_BGR2RGB)
            search_tensor[i,:,:,:] = img2tensor(search_arr)
        '''Note: we don't normalize these tensors here, 
        but leave normalization to training process'''


        gt_file_visible = os.path.join(dataset_dir,'init_gt_visible.txt')
        gt_arr_visible = np.loadtxt(gt_file_visible,dtype=np.float64,delimiter=',')
        gt_init = gt_arr_visible[index]

        x_cur_folder = self.folders_list[index]
        x_img_paths = sorted(glob.glob(os.path.join(x_cur_folder,'infrared','*.jpg')))  # 中间加入modal
        '''get init frame tensor'''
        x_init_frame_path = x_img_paths[0]
        x_init_frame_arr = cv2.imread(x_init_frame_path)
        # BGR2RGB
        x_init_frame_arr = cv2.cvtColor(x_init_frame_arr, cv2.COLOR_BGR2RGB)
        init_x_tensor = img2tensor(x_init_frame_arr)
        '''get search regions' tensor'''
        x_search_region_paths = x_img_paths[1:self.max_num+1] # to avoid being out of GPU memory
        x_num_search = len(x_search_region_paths)
        search_x_tensor = torch.zeros((x_num_search,3,256,256),dtype=torch.float32)
        for i in range(x_num_search):
            x_search_arr = cv2.imread(x_search_region_paths[i])
            #BGR2RGB
            x_search_arr = cv2.cvtColor(x_search_arr, cv2.COLOR_BGR2RGB)
            search_x_tensor[i,:,:,:] = img2tensor(x_search_arr)
        '''Note: we don't normalize these tensors here, 
        but leave normalization to training process'''
        

        gt = np.loadtxt(os.path.join(cur_folder,'visible', 'visible.txt'), dtype=np.float64,delimiter=' ')[0:self.max_num]
        gt_x = np.loadtxt(os.path.join(x_cur_folder,'infrared', 'infrared.txt'), dtype=np.float64,delimiter=' ')[0:self.max_num]
        point = np.loadtxt(os.path.join(x_cur_folder,'visible', 'visible_4point.txt'), dtype=np.float64,delimiter=' ')[0:self.max_num]
        point_x = np.loadtxt(os.path.join(x_cur_folder,'infrared', 'infrared_4point.txt'), dtype=np.float64,delimiter=' ')[0:self.max_num]

        return init_tensor, search_tensor, init_x_tensor, search_x_tensor, gt_init, gt, gt_x, point, point_x
    

class RGB234_dataset_random(Dataset):

    def __init__(self, max_num=15):
        # dataset_dir就是common_path import train_set_path_，在这里设置数据地址
        folders = sorted(os.listdir(dataset_dir))
        folders.remove('init_gt_visible.txt')
        folders.remove('init_gt_infrared.txt')
        self.folders_list = [os.path.join(dataset_dir,folder) for folder in folders]
        self.max_num = max_num
    def __getitem__(self, index):
        # 先录入可见光search等信息
        cur_folder = self.folders_list[index]
        img_paths = sorted(glob.glob(os.path.join(cur_folder,'visible','*.jpg')))  # 中间加入modal
        '''get init frame tensor'''
        init_frame_path = img_paths[0]
        init_frame_arr = cv2.imread(init_frame_path)
        # BGR2RGB
        init_frame_arr = cv2.cvtColor(init_frame_arr, cv2.COLOR_BGR2RGB)
        init_tensor = img2tensor(init_frame_arr)
        '''get random number'''
        import random
        random_number = random.randint(1, len(img_paths) - 5)  # 包括首和尾两个数的
        '''get search regions' tensor'''
        search_region_paths = img_paths[random_number:random_number+5] # 直接获得所有的路径
        num_search = len(search_region_paths)
        search_tensor = torch.zeros((num_search,3,256,256),dtype=torch.float32)
        for i in range(num_search):
            search_arr = cv2.imread(search_region_paths[i])
            #BGR2RGB
            search_arr = cv2.cvtColor(search_arr, cv2.COLOR_BGR2RGB)
            search_tensor[i,:,:,:] = img2tensor(search_arr)
        '''Note: we don't normalize these tensors here, 
        but leave normalization to training process'''

        gt_file_visible = os.path.join(dataset_dir,'init_gt_visible.txt')
        gt_arr_visible = np.loadtxt(gt_file_visible,dtype=np.float64,delimiter=',')
        gt_init = gt_arr_visible[index]

        x_cur_folder = self.folders_list[index]
        x_img_paths = sorted(glob.glob(os.path.join(x_cur_folder,'infrared','*.jpg')))  # 中间加入modal
        '''get init frame tensor'''
        x_init_frame_path = x_img_paths[0]
        x_init_frame_arr = cv2.imread(x_init_frame_path)
        # BGR2RGB
        x_init_frame_arr = cv2.cvtColor(x_init_frame_arr, cv2.COLOR_BGR2RGB)
        init_x_tensor = img2tensor(x_init_frame_arr)
        '''get search regions' tensor'''
        x_search_region_paths = x_img_paths[random_number:random_number+5] # to avoid being out of GPU memory
        x_num_search = len(x_search_region_paths)
        search_x_tensor = torch.zeros((x_num_search,3,256,256),dtype=torch.float32)
        for i in range(x_num_search):
            x_search_arr = cv2.imread(x_search_region_paths[i])
            #BGR2RGB
            x_search_arr = cv2.cvtColor(x_search_arr, cv2.COLOR_BGR2RGB)
            search_x_tensor[i,:,:,:] = img2tensor(x_search_arr)
        '''Note: we don't normalize these tensors here, 
        but leave normalization to training process'''
        

        gt = np.loadtxt(os.path.join(cur_folder,'visible', 'visible.txt'), dtype=np.float64,delimiter=' ')[random_number-1:random_number-1+5]  # 不包括第一帧所以减一
        gt_x = np.loadtxt(os.path.join(x_cur_folder,'infrared', 'infrared.txt'), dtype=np.float64,delimiter=' ')[random_number-1:random_number-1+5]
        return init_tensor, search_tensor, init_x_tensor, search_x_tensor, gt_init, gt, gt_x
    
    def __len__(self):
        return len(self.folders_list)
    
    def __len__(self):
        return len(self.folders_list)


class RGB234_dataset_random_response(Dataset):
    def __init__(self, max_num=15, response_path="./dataset/RGBT234_response_vipt", seed=42):
        # dataset_dir就是common_path import train_set_path_，在这里设置数据地址
        folders = sorted(os.listdir(dataset_dir))
        folders.remove('init_gt_visible.txt')
        folders.remove('init_gt_infrared.txt')
        self.folders_list = [os.path.join(dataset_dir,folder) for folder in folders]
        self.max_num = max_num
        # response
        self.response_path = response_path  # 不同的算法有不一致的response图，外置路径记得修改
        # random for repeat easily
        rng = np.random.RandomState(seed)
        self.random_numbers = []
        for folder in self.folders_list:
            img_paths = sorted(glob.glob(os.path.join(folder,'visible','*.jpg')))
            self.random_numbers.append(rng.randint(1, len(img_paths) - 5))

    def __getitem__(self, index):
        # 先录入可见光search等信息
        cur_folder = self.folders_list[index]
        img_paths = sorted(glob.glob(os.path.join(cur_folder,'visible','*.jpg')))  # 中间加入modal
        '''get init frame tensor'''
        init_frame_path = img_paths[0]
        init_frame_arr = cv2.imread(init_frame_path)
        # BGR2RGB
        init_frame_arr = cv2.cvtColor(init_frame_arr, cv2.COLOR_BGR2RGB)
        init_tensor = img2tensor(init_frame_arr)
        '''get random number'''
        random_number = self.random_numbers[index]
        '''get search regions' tensor'''
        search_region_paths = img_paths[random_number:random_number+5]
        num_search = len(search_region_paths)
        search_tensor = torch.zeros((num_search,3,256,256),dtype=torch.float32)
        for i in range(num_search):
            search_arr = cv2.imread(search_region_paths[i])
            #BGR2RGB
            search_arr = cv2.cvtColor(search_arr, cv2.COLOR_BGR2RGB)
            search_tensor[i,:,:,:] = img2tensor(search_arr)
        '''Note: we don't normalize these tensors here, 
        but leave normalization to training process'''
        gt_file_visible = os.path.join(dataset_dir,'init_gt_visible.txt')
        gt_arr_visible = np.loadtxt(gt_file_visible,dtype=np.float64,delimiter=',')
        gt_init = gt_arr_visible[index]

        x_cur_folder = self.folders_list[index]
        x_img_paths = sorted(glob.glob(os.path.join(x_cur_folder,'infrared','*.jpg')))  # 中间加入modal
        '''get init frame tensor'''
        x_init_frame_path = x_img_paths[0]
        x_init_frame_arr = cv2.imread(x_init_frame_path)
        # BGR2RGB
        x_init_frame_arr = cv2.cvtColor(x_init_frame_arr, cv2.COLOR_BGR2RGB)
        init_x_tensor = img2tensor(x_init_frame_arr)
        '''get search regions' tensor'''
        x_search_region_paths = x_img_paths[random_number:random_number+5] # to avoid being out of GPU memory
        x_num_search = len(x_search_region_paths)
        search_x_tensor = torch.zeros((x_num_search,3,256,256),dtype=torch.float32)
        for i in range(x_num_search):
            x_search_arr = cv2.imread(x_search_region_paths[i])
            #BGR2RGB
            x_search_arr = cv2.cvtColor(x_search_arr, cv2.COLOR_BGR2RGB)
            search_x_tensor[i,:,:,:] = img2tensor(x_search_arr)
        '''Note: we don't normalize these tensors here, 
        but leave normalization to training process'''

        '''get response path'''
        cur_folder_response = replace_path_component(cur_folder, os.path.basename(self.response_path), -2)
        response_fusion_path = sorted(glob.glob(os.path.join(cur_folder_response,'fusion','*.jpg')))
        response_visible_path = sorted(glob.glob(os.path.join(cur_folder_response,'visible','*.jpg')))
        response_infrared_path = sorted(glob.glob(os.path.join(cur_folder_response,'infrared','*.jpg')))
        '''get response tensor'''
        response_fusion_region_paths = response_fusion_path[random_number-1:random_number+4]
        response_visible_region_paths = response_visible_path[random_number-1:random_number+4]
        response_infrared_region_paths = response_infrared_path[random_number-1:random_number+4]
        num_response_fusion = len(response_fusion_region_paths)
        response_fusion_tensor = torch.zeros((num_response_fusion,3,16,16),dtype=torch.float32)
        response_visible_tensor = torch.zeros((num_response_fusion,3,16,16),dtype=torch.float32)
        response_infrared_tensor = torch.zeros((num_response_fusion,3,16,16),dtype=torch.float32)
        for i in range(num_response_fusion):
            response_fusion_arr = cv2.imread(response_fusion_region_paths[i])
            response_visible_arr = cv2.imread(response_visible_region_paths[i])
            response_infrared_arr = cv2.imread(response_infrared_region_paths[i])
            #BGR2RGB
            response_fusion_arr = cv2.cvtColor(response_fusion_arr, cv2.COLOR_BGR2RGB)
            response_visible_arr = cv2.cvtColor(response_visible_arr, cv2.COLOR_BGR2RGB)
            response_infrared_arr = cv2.cvtColor(response_infrared_arr, cv2.COLOR_BGR2RGB)
            response_fusion_tensor[i,:,:,:] = img2tensor(response_fusion_arr)
            response_visible_tensor[i,:,:,:] = img2tensor(response_visible_arr)
            response_infrared_tensor[i,:,:,:] = img2tensor(response_infrared_arr)

        gt = np.loadtxt(os.path.join(cur_folder,'visible', 'visible.txt'), dtype=np.float64,delimiter=' ')[random_number-1:random_number-1+5]
        gt_x = np.loadtxt(os.path.join(x_cur_folder,'infrared', 'infrared.txt'), dtype=np.float64,delimiter=' ')[random_number-1:random_number-1+5]
        return init_tensor, search_tensor, init_x_tensor, search_x_tensor, gt_init, gt, gt_x, response_fusion_tensor, response_visible_tensor, response_infrared_tensor
    
    def __len__(self):
        return len(self.folders_list)
    
    def __len__(self):
        return len(self.folders_list)


# class RGB234_dataset_full_image(Dataset):
#     def __init__(self, max_num=20, data_path="/data/dataset/RGBT234", seed=42):
#         # max_num is number of frames
#         # random for repeat easily
#         self.max_num = max_num
#         self.data_path = data_path
#         self.rng = np.random.RandomState(seed)
#         exclude = ['orangeman1']
#         self.all_data_list = sorted(os.listdir(data_path))
#         self.data_list = [f for f in self.all_data_list if f not in exclude]
#         self.random_numbers = []
#         for folder in self.data_list:
#             img_paths = sorted(glob.glob(os.path.join(self.data_path, folder, 'visible','*.jpg')))
#             self.random_numbers.append(self.rng.randint(1, len(img_paths) - self.max_num))

#     def __getitem__(self, index):
#         '''get random number'''
#         random_number = self.random_numbers[index]
#         '''get visible image folder'''
#         cur_folder = self.data_list[index]
#         img_paths = sorted(glob.glob(os.path.join(self.data_path, cur_folder,'visible','*.jpg')))
#         '''get visible init frame path and gt'''
#         init_frame_path = img_paths[random_number]
#         gt_file = os.path.join(self.data_path, cur_folder,'visible.txt')
#         gt_arr = np.loadtxt(gt_file, dtype=np.float64,delimiter=',')
#         gt_init = gt_arr[random_number]
#         '''get all visible gt'''
#         gt = gt_arr[random_number:random_number+self.max_num]
#         '''get all frames paths'''
#         frame_path = img_paths[random_number:random_number+self.max_num]

#         '''get infrared image folder'''
#         img_paths_x = sorted(glob.glob(os.path.join(self.data_path, cur_folder,'infrared','*.jpg')))
#         '''get visible init frame path and gt'''
#         init_frame_path_x = img_paths_x[random_number]
#         gt_file_x = os.path.join(self.data_path, cur_folder,'infrared.txt')
#         gt_arr_x = np.loadtxt(gt_file_x, dtype=np.float64,delimiter=',')
#         gt_init_x = gt_arr_x[random_number]
#         '''get all visible gt'''
#         gt_x = gt_arr_x[random_number:random_number+self.max_num]
#         '''get all frames paths'''
#         frame_path_x = img_paths_x[random_number:random_number+self.max_num]
        
#         return frame_path, gt, frame_path_x, gt_x 
     
#     def __len__(self):
#         return len(self.data_list)


class RGB234_dataset_full_image(Dataset):
    def __init__(self, max_num=20, data_path="/data/dataset/RGBT234", seed=42):
        self.max_num = max_num
        self.data_path = data_path
        self.seed = seed
        exclude = ['orangeman1']
        self.all_data_list = sorted(os.listdir(data_path))
        self.data_list = [f for f in self.all_data_list if f not in exclude]

    def __getitem__(self, index):
        # 这里假设你有全局变量 epoch，如果没有，可以传入 epoch 参数
        epoch = getattr(self, 'epoch', 0)
        # 用 index+epoch+seed 保证每次采样都可复现且不同
        rng = np.random.RandomState(self.seed + index + epoch * len(self))
        cur_folder = self.data_list[index]
        img_paths = sorted(glob.glob(os.path.join(self.data_path, cur_folder, 'visible','*.jpg')))
        valid_range = len(img_paths) - self.max_num
        random_number = rng.randint(1, valid_range)

        # visible
        init_frame_path = img_paths[random_number]
        gt_file = os.path.join(self.data_path, cur_folder,'visible.txt')
        gt_arr = np.loadtxt(gt_file, dtype=np.float64,delimiter=',')
        gt_init = gt_arr[random_number]
        gt = gt_arr[random_number:random_number+self.max_num]
        frame_path = img_paths[random_number:random_number+self.max_num]

        # infrared
        img_paths_x = sorted(glob.glob(os.path.join(self.data_path, cur_folder,'infrared','*.jpg')))
        init_frame_path_x = img_paths_x[random_number]
        gt_file_x = os.path.join(self.data_path, cur_folder,'infrared.txt')
        gt_arr_x = np.loadtxt(gt_file_x, dtype=np.float64,delimiter=',')
        gt_init_x = gt_arr_x[random_number]
        gt_x = gt_arr_x[random_number:random_number+self.max_num]
        frame_path_x = img_paths_x[random_number:random_number+self.max_num]
        
        return frame_path, gt, frame_path_x, gt_x 
     
    def __len__(self):
        return len(self.data_list)

    # 新增方法：每个epoch开始时设置当前epoch
    def set_epoch(self, epoch):
        self.epoch = epoch

class VTUAV_dataset_full_image(Dataset):
    def __init__(self, max_num=20, data_path="/data/dataset/VTUAV/train/train_ST/", seed=42):
        self.max_num = max_num
        self.data_path = data_path
        self.seed = seed
        self.data_list = sorted(os.listdir(data_path))

    def __getitem__(self, index):
        # 这里假设你有全局变量 epoch，如果没有，可以传入 epoch 参数
        epoch = getattr(self, 'epoch', 0)
        # 用 index+epoch+seed 保证每次采样都可复现且不同
        rng = np.random.RandomState(self.seed + index + epoch * len(self))
        cur_folder = self.data_list[index]
        img_paths = sorted(glob.glob(os.path.join(self.data_path, cur_folder, 'rgb','*.jpg')))
        img_paths_x = sorted(glob.glob(os.path.join(self.data_path, cur_folder, 'ir', '*.jpg')))

        # gt
        gt_file = os.path.join(self.data_path, cur_folder,'rgb.txt')
        gt_arr = np.loadtxt(gt_file, dtype=np.float64,delimiter=' ')
        gt_file_x = os.path.join(self.data_path, cur_folder,'ir.txt')
        gt_arr_x = np.loadtxt(gt_file_x, dtype=np.float64,delimiter=' ')

        valid_range = len(gt_arr) - self.max_num
        gt_start_idx = rng.randint(1, valid_range)

        # 10 frame per
        stride = 10
        frame_indices = [gt_idx * stride for gt_idx in range(gt_start_idx, gt_start_idx + self.max_num)]

        frame_path   = [img_paths[i] for i in frame_indices]
        frame_path_x = [img_paths_x[i] for i in frame_indices]
        gt           = gt_arr[gt_start_idx : gt_start_idx + self.max_num]
        gt_x         = gt_arr_x[gt_start_idx : gt_start_idx + self.max_num]
        
        return frame_path, gt, frame_path_x, gt_x 
     
    def __len__(self):
        return len(self.data_list)

    # 新增方法：每个epoch开始时设置当前epoch
    def set_epoch(self, epoch):
        self.epoch = epoch