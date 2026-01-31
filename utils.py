
import math
import torch
import cv2
import numpy as np
import os
from torchvision.transforms import Resize
import torch.nn.functional as F

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def save_checkpoint(model, optimizer, epoch, path):
    """
    save ckpt
    Args:
        model (torch.nn.Module): model save
        optimizer (torch.optim.Optimizer): opt
        epoch (int):  epoch right now
        path (str): path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, os.path.join(path, f"checkpoint_epoch_{epoch}.pth"))
    print(f"Checkpoint saved at epoch {epoch} to {path}/checkpoint_epoch_{epoch}.pth")

def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Restored from checkpoint - ckp_path:{filepath},Epoch: {epoch+1}")
    return model, optimizer, epoch

def denormalize(im_tensor):
    '''(-1,1) ---> (0,255)'''
    im_tensor = (im_tensor + 1) * 127.5
    # return im_tensor.clamp(0, 255)
    return im_tensor


def denormalize_noise(im_tensor, dn_vi):
    '''(-1,1) ---> (0,255)'''
    im_tensor_ = im_tensor * dn_vi  # 15.0以前是10， 20
    # return im_tensor.clamp(0, 255)
    return im_tensor_

def denormalize_noise_ir(im_tensor, dn_ir):
    '''(-1,1) ---> (0,255)'''
    im_tensor_ = im_tensor * dn_ir    # 以前是2， 10
    # return im_tensor.clamp(0, 255)
    return im_tensor_


def box2crop_left(input, search_factor):
    '''
    input: [b, n, 4] 
    output: [b, n, 4]
    ''' 
    output = torch.zeros_like(input)
    crop_z = torch.ceil(torch.sqrt(input[:, :, 2] * input[:, :, 3]) * search_factor)
    crop_z = crop_z.unsqueeze(2).repeat(1, 1, 4)
    resize_factor =  256.0 / crop_z
    output = input  * resize_factor
    output[:, :, 0] = 128.0  # center
    output[:, :, 1] = 128.0  # center
    return output


def box2crop_center(input, search_factor):
    '''
    input: [n, 4]
    output: [n, 4]
    ''' 
    output = torch.zeros_like(input)
    for index in range(input.size(0)):
        crop_z = math.ceil(math.sqrt(input[index, 2] * input[index, 3]) * search_factor)
        resize_factor =  256.0 / crop_z
        tmp = input[index, :]  * resize_factor
        output[index, :] = tmp
        output[index, 0] = 128.0 - tmp[2]/2.0  # 左上角点x1
        output[index, 1] = 128.0 - tmp[3]/2.0  # y1
        # output[index, 0] = 128.0  # center
        # output[index, 1] = 128.0  # center
    return output


def tensor_to_cv2(tensor_):

    tensor = tensor_.cpu()
    tensor = tensor.squeeze(dim=0)
    image = tensor.detach().numpy()
    image = image.transpose((1, 2, 0))

    image = image.clip(0, 255).astype('uint8')
    return image

def draw_boxes_on_cv2_image(image, boxes, color='red', thickness=2):

    image = image.copy()  # 防止修改原图

    

    x_min, y_min, w, h = map(int, boxes)
    

    
    cv2.rectangle(image, (x_min, y_min), (x_min+w, y_min+h), (255,0,0),2)
    return image

def draw_boxes_on_cv2_image_compare(image, boxes, boxes1, boxes2 ,color='red', thickness=1):
    """在OpenCV图像上绘制边界框"""
    image = image.copy()  
    # if not isinstance(colors, (list, tuple)):
    #     colors = [colors]*len(boxes)  # 如果只给定一种颜色，为每个框重复该颜色
    #debug

    x_min, y_min, w, h = map(int, boxes)
    x_min1, y_min1, w1, h1 = map(int, boxes1)
    x_min2, y_min2, w2, h2 = map(int, boxes2)

    
    # OpenCV使用BGR格式，因此需要调整颜色顺序
    # color = tuple(reversed(tuple(map(int, color.split(','))))) if isinstance(color, str) else color
    image=image[:, :, :3]
    image = np.array(image, dtype=np.uint8, order='C')
    cv2.rectangle(image, (x_min, y_min), (x_min+w, y_min+h), (0,255,0), 2) #green gt
    cv2.rectangle(image, (x_min2, y_min2), (x_min2+w2, y_min2+h2), (255,0,0), 2) #blue region_clean
    cv2.rectangle(image, (x_min1, y_min1), (x_min1+w1, y_min1+h1), (0,0,255), 2) #red region
    return image    

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

def replace_path_component(path, replacement, id):


    parts = path.split(os.sep)

    if len(parts) > abs(id):
        parts[id] = replacement

    new_path = os.sep.join(parts)
    return new_path

def simulate_test_patch(patch, lab_batch):
    '''

    lab_batch: torch.Size([4, 5, 4])
    patch: torch.Size([1, 3, 128, 128])
    '''
    patch_resize = torch.zeros(lab_batch.size(0), lab_batch.size(1), )  # [4,5,3,h,w]
    patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)  
    scale_ratio = 0.25
    lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)  # torch.Size([8, 14, 5])
    lab_batch_scaled[:, :, 0] = lab_batch[:, :, 0]
    lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1]
    lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2]
    lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3]
    target_size = 1.2 * torch.sqrt(((lab_batch_scaled[:, :, 2].mul(scale_ratio)) ** 2) + (
                        (lab_batch_scaled[:, :, 3].mul(scale_ratio)) ** 2))  # [4, 5]
    
    for i in range(lab_batch.size(0)):
        for j in range(lab_batch.size(1)):
            torch_resize = Resize([int(target_size[i,j].item()), int(target_size[i,j].item())]) # 定义Resize类对象
            # a = torch_resize(patch).squeeze(dim=0)
            patch_resize[i,j,:,:,:] = torch_resize(patch).squeeze(dim=0)
    return patch_resize

def iter_choose(response_fusion, response_vi, response_ir):
    '''
    输入：
    response_fusion:[4, 5, 3, 16, 16]  [0-255]
    response_vi:[4, 5, 3, 16, 16]
    response_ir:[4, 5, 3, 16, 16]

    '''
    mse = torch.nn.MSELoss()
    b_size = response_fusion.size(0)*response_fusion.size(1)
    response_fusion = response_fusion.view(-1, 3, 16, 16)
    response_vi = response_vi.view(-1, 3, 16, 16)
    response_ir = response_ir.view(-1, 3, 16, 16)  
    dis_vi = mse(response_fusion, response_vi)/b_size
    dis_ir = mse(response_fusion, response_ir)/b_size
    alpha_before = 1.0/(dis_vi)
    beta_before = 1.0/(dis_ir)
    values = torch.tensor([[alpha_before,beta_before]]).cuda()
    final = F.softmax(values, dim=1)
    return final[:,0], final[:,1]

def compute_gradient(image):

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    if not image.dtype.is_floating_point:
        image = image.float()

    num_channels = image.shape[1]

    sobel_x = sobel_x.expand(num_channels, -1, -1, -1)
    sobel_y = sobel_y.expand(num_channels, -1, -1, -1)

    gradient_x = F.conv2d(image, sobel_x, padding=1, groups=num_channels)

    gradient_y = F.conv2d(image, sobel_y, padding=1, groups=num_channels)

    gradient = torch.cat((gradient_x, gradient_y), dim=1)
    return gradient



def draw_boxes_on_tensor(input, input_gt, pred_gt, attack_box, save_path):
    '''
    input: [b, 5, 3, 256, 256] vi
    input_gt, pred_gt, attack_box: [b, 5, 4]
    '''
    from torchvision.utils import save_image
    input = input.view(-1, *input[2:])
    input_gt, pred_gt, attack_box  = input_gt.view(-1, *input_gt[2:]), pred_gt.view(-1, *pred_gt[2:]), attack_box.view(-1, *attack_box[2:])
    input = input.detach().cpu().numpy()
    draw_tensor = torch.zeros_like(input)
    for idx in range(input.shape[0]):
        x1, y1, w, h = attack_box[idx,:]
        x1_clean, y1_clean, w_clean, h_clean = pred_gt[idx,:]
        x1_gt, y1_gt, w_gt, h_gt = input_gt[idx,:]
        image_adv = tensor_to_cv2(input[idx,:,:,:])
        image_BGR_adv = cv2.cvtColor(image_adv, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image_BGR_adv, (int(x1- 0.5*w), int(y1 - 0.5*h)), (int(x1 + 0.5*w), int(y1 + 0.5*h)), color=(0, 0, 255), thickness=2)
        cv2.rectangle(image_BGR_adv, (int(x1_gt- 0.5*w_gt), int(y1_gt - 0.5*h_gt)), (int(x1_gt + 0.5*w_gt), int(y1_gt + 0.5*h_gt)), color=(0, 255, 0), thickness=2)
        cv2.rectangle(image_BGR_adv, (int(x1_clean- 0.5*w_clean), int(y1_clean - 0.5*h_clean)), (int(x1_clean + 0.5*w_clean), int(y1_clean + 0.5*h_clean)), color=(255, 0, 0), thickness=2)        
        image_tensor = torch.from_numpy(image_BGR_adv.transpose(2, 0, 1)).float() / 255.0
        draw_tensor[idx,:,:,:] = image_tensor.cuda()
    save_image(draw_tensor, save_path, normalize=True)


def restore_to_original_image(x_adv, original_img, model_sz, original_sz, crop_coords, pad_info):
    """
    sub_window to origin 
        x_adv_vi,x_adv_ir:torch.Size([1, 3, 255, 255])
        original_img: 原始图像 (HWC)
        crop_coords: 裁剪坐标 [ymin, ymax, xmin, xmax]
        pad_info: 填充信息 [top_pad, bottom_pad, left_pad, right_pad]
    """
   
    x_adv = x_adv.squeeze(0).permute(1, 2, 0).cpu()          
    x_adv = np.array(x_adv,dtype=np.uint8,order="C")
                   
    patch_hwc = x_adv
    # print("c_c:",crop_coords,pad_info) #[23, 347, 96, 420]

    if not np.array_equal(model_sz, original_sz):
        patch_hwc = cv2.resize(patch_hwc, (original_sz, original_sz)) #(325,325,3)
    # print(patch_hwc.shape)
    
    # 2. 提取原始填充信息
    top_pad, bottom_pad, left_pad, right_pad = pad_info #[15,0,0,0]
    ymin, ymax, xmin, xmax = crop_coords
    
    # 3. 计算有效区域（去除填充部分）
    h, w = patch_hwc.shape[:2]
    valid_h = h - top_pad - bottom_pad
    valid_w = w - left_pad - right_pad
    
    context_xmin = xmin - left_pad
    context_xmax = xmax - left_pad
    context_ymin = ymin - top_pad
    context_ymax = ymax - top_pad

    # cv2.imwrite(os.path.join('./img_test/debug/'+'patch_hwc'+'.jpg'),patch_hwc)
    # 4. 提取有效内容（去除填充边界）
    valid_patch = patch_hwc[top_pad:top_pad+valid_h, 
                           left_pad:left_pad+valid_w, :] #(0 265 0 288)
    
    
    # cv2.imwrite(os.path.join('./img_test/debug/'+'valid_patch'+'.jpg'),valid_patch) #(265, 288, 3)
    # print(valid_patch.shape)
   
    # print(context_ymin,context_ymax+1, context_xmin,context_xmax+1)
    context_ymin = max(0, int(context_ymin))          
    context_ymax = min(original_img.shape[0]-1, int(context_ymax)) 
    context_xmin = max(0, int(context_xmin))          
    context_xmax = min(original_img.shape[1]-1, int(context_xmax))

    original_img[context_ymin:context_ymax+1, context_xmin:context_xmax+1, :] = valid_patch
    # cv2.imwrite(os.path.join('./img_test/debug/'+'originimage'+'.jpg'),original_img)
    
    return original_img


def draw_boxes_on_tensor_ours(input, input_gt, pred_gt, attack_box, save_path):
    '''
    input: [b, 5, 3, 256, 256] vi
    input_gt, pred_gt, attack_box: [b, 5, 4]
    '''
    from torchvision.utils import save_image
    # input = input.view(-1, *input.shape[2:])
    input_gt, pred_gt, attack_box  = input_gt.view(-1, 4), pred_gt.view(-1, 4), attack_box.view(-1, 4)
    # input = input.detach().cpu().numpy()
    draw_tensor = torch.zeros_like(input)
    for idx in range(input.shape[0]):
        x1, y1, w, h = attack_box[idx,:]
        x1_clean, y1_clean, w_clean, h_clean = pred_gt[idx,:]
        x1_gt, y1_gt, w_gt, h_gt = input_gt[idx,:]
        image_adv = tensor_to_cv2(input[idx,:,:,:])
        image_BGR_adv = cv2.cvtColor(image_adv, cv2.COLOR_RGB2BGR)
        image_RGB_adv = cv2.cvtColor(image_BGR_adv, cv2.COLOR_BGR2RGB)
        cv2.rectangle(image_RGB_adv, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
        cv2.rectangle(image_RGB_adv, (int(x1_gt), int(y1_gt)), (int(x1_gt + w_gt), int(y1_gt + h_gt)), color=(0, 255, 0), thickness=2)
        cv2.rectangle(image_RGB_adv, (int(x1_clean), int(y1_clean)), (int(x1_clean + w_clean), int(y1_clean + h_clean)), color=(255, 0, 0), thickness=2)        
        # cv2.rectangle(image_RGB_adv, (int(x1- 0.5*w), int(y1 - 0.5*h)), (int(x1 + 0.5*w), int(y1 + 0.5*h)), color=(0, 0, 255), thickness=2)
        # cv2.rectangle(image_RGB_adv, (int(x1_gt- 0.5*w_gt), int(y1_gt - 0.5*h_gt)), (int(x1_gt + 0.5*w_gt), int(y1_gt + 0.5*h_gt)), color=(0, 255, 0), thickness=2)
        # cv2.rectangle(image_RGB_adv, (int(x1_clean- 0.5*w_clean), int(y1_clean - 0.5*h_clean)), (int(x1_clean + 0.5*w_clean), int(y1_clean + 0.5*h_clean)), color=(255, 0, 0), thickness=2)        
        image_tensor = torch.from_numpy(image_RGB_adv.transpose(2, 0, 1)).float() / 255.0
        # cv2.rectangle(image_BGR_adv, (int(x1- 0.5*w), int(y1 - 0.5*h)), (int(x1 + 0.5*w), int(y1 + 0.5*h)), color=(0, 0, 255), thickness=2)
        # cv2.rectangle(image_BGR_adv, (int(x1_gt- 0.5*w_gt), int(y1_gt - 0.5*h_gt)), (int(x1_gt + 0.5*w_gt), int(y1_gt + 0.5*h_gt)), color=(0, 255, 0), thickness=2)
        # cv2.rectangle(image_BGR_adv, (int(x1_clean- 0.5*w_clean), int(y1_clean - 0.5*h_clean)), (int(x1_clean + 0.5*w_clean), int(y1_clean + 0.5*h_clean)), color=(255, 0, 0), thickness=2)        
        # image_tensor = torch.from_numpy(image_BGR_adv.transpose(2, 0, 1)).float() / 255.0
        draw_tensor[idx,:,:,:] = image_tensor.cuda()
    save_image(draw_tensor, save_path, normalize=True)



import torch
from pytorch_msssim import ssim

def calculate_ssim_color(img1_tensor, img2_tensor):
    assert img1_tensor.shape == img2_tensor.shape, "input tensor dimension is not consistent"
    device = img2_tensor.device
    img1_tensor = img1_tensor.cuda()
    from pytorch_msssim import ssim
    ssim_value = ssim(img1_tensor, img2_tensor, data_range=1.0, size_average=True)
    return ssim_value


def get_x_frame(color_path, depth_path, dtype='rgbcolormap', depth_clip=False):
    ''' read RGB and depth images  get_rgbd_frame

        max_depth = 10 meter, in the most frames in CDTB and DepthTrack , the depth of target is smaller than 10 m
        When on CDTB and DepthTrack testing, we use this depth clip
    '''
    if color_path:
        rgb = cv2.imread(color_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    else:
        rgb = None

    if depth_path:
        dp = cv2.imread(depth_path, -1)  

        if depth_clip:
            max_depth = min(np.median(dp) * 3, 10000)
            dp[dp > max_depth] = max_depth
    else:
        dp = None

    if dtype == 'color':
        img = rgb

    elif dtype == 'raw_x':
        img = dp

    elif dtype == 'colormap':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dp = np.asarray(dp, dtype=np.uint8)
        img = cv2.applyColorMap(dp, cv2.COLORMAP_JET)

    elif dtype == '3x':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dp = np.asarray(dp, dtype=np.uint8)
        img = cv2.merge((dp, dp, dp))

    elif dtype == 'normalized_x':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img = np.asarray(dp, dtype=np.uint8)

    elif dtype == 'rgbcolormap':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dp = np.asarray(dp, dtype=np.uint8)
        colormap = cv2.applyColorMap(dp, cv2.COLORMAP_JET)  # (h,w) -> (h,w,3)
        img = cv2.merge((rgb, colormap))  # (h,w,6)

    elif dtype == 'rgb3x':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dp = np.asarray(dp, dtype=np.uint8)
        dp = cv2.merge((dp, dp, dp))
        img = cv2.merge((rgb, dp))

    elif dtype == 'rgbrgb':
        dp = cv2.cvtColor(dp, cv2.COLOR_BGR2RGB)
        img = cv2.merge((rgb, dp))

    else:
        print('No such dtype !!! ')
        img = None

    return img

def bbox_iou_xywh(box1, box2, eps=1e-6):
    """
    box1, box2: [N, 4] (x, y, w, h)
    返回每对 box 的 IoU，shape [N]
    """
    # 转为 x1y1x2y2
    box1_x1 = box1[:, 0]
    box1_y1 = box1[:, 1]
    box1_x2 = box1[:, 0] + box1[:, 2]
    box1_y2 = box1[:, 1] + box1[:, 3]

    box2_x1 = box2[:, 0]
    box2_y1 = box2[:, 1]
    box2_x2 = box2[:, 0] + box2[:, 2]
    box2_y2 = box2[:, 1] + box2[:, 3]

    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (box1_x2 - box1_x1).clamp(min=0) * (box1_y2 - box1_y1).clamp(min=0)
    area2 = (box2_x2 - box2_x1).clamp(min=0) * (box2_y2 - box2_y1).clamp(min=0)
    union_area = area1 + area2 - inter_area + eps

    iou = inter_area / union_area
    return iou


def cal_attack(adv, gt):
    device = adv.device
    loss_list = []
    for i in range(adv.shape[0]):
        cur_iou = bbox_iou_xywh(adv[i].unsqueeze(dim=0), gt[i].unsqueeze(dim=0))
        temp_center = torch.zeros_like(cur_iou).to(device)
        if i > 0:
            for j in range(i):
                temp_center = temp_center - 0.9**(i-j) * torch.mean(torch.abs(adv[j, :2] - gt[j, :2])) 
        loss_list.append(cur_iou + 0.1 * temp_center)
    return torch.stack(loss_list).sum()


def zero_circle_region(response, r):
    # response: [1, 1, H, W]
    H, W = response.shape[2], response.shape[3]
    # find the lightest
    max_idx = torch.argmax(response)
    cy = (max_idx // W) % H
    cx = max_idx % W

    y = torch.arange(H, device=response.device).view(-1, 1)
    x = torch.arange(W, device=response.device).view(1, -1)
    mask = (x - cx)**2 + (y - cy)**2 <= r**2  # 圆内为True

    # to 0
    response = response.clone()  
    response[0, 0][mask] = 0
    return response

# if __name__ == '__main__':
#     a = replace_path_component("./dataset/RGBT234/bike", "./dataset/RGBT234_response_vipt", -2)
#     print(a)

import importlib
def get_tracker(script_name, params):
    '''for other tracker'''
    tracker_module = importlib.import_module('lib.test.tracker.{}'.format(script_name))
    tracker_class = tracker_module.get_tracker_class()
    tracker = tracker_class(params)
    return tracker

def create_tracker(script_name, params, dataset_name):
    '''for sutrack'''
    tracker_module = importlib.import_module('lib.test.tracker.{}'.format(script_name))
    tracker_class = tracker_module.get_tracker_class()
    tracker = tracker_class(params, dataset_name)
    return tracker

def get_parameters(script_name, yaml_name):
    """Get parameters."""
    param_module = importlib.import_module('lib.test.parameter.{}'.format(script_name))
    params = param_module.parameters(yaml_name)
    return params

def restore_noisy_crop_to_original(original_img, noisy_crop, box_extract, search_area_factor, output_sz):
    """
    将加噪后的裁剪区域还原回原始图像
    
    参数:
        original_img: 原始图像 (H,W,C)
        noisy_crop: 加噪后的裁剪区域 (output_sz,output_sz,C)
        box_extract: 原始裁剪框 [x,y,w,h]
        search_area_factor: 原始裁剪时的搜索区域因子
        output_sz: 裁剪输出尺寸
        
    返回:
        restored_img: 还原后的完整图像
        mask: 显示修改区域的二值掩码
    """
    # x, y, w, h = gt.tolist()
    # crop_sz = math.ceil(math.sqrt(w * h) * cfg.search_factor)
    # box_extract = [x + 0.5*w - crop_sz*0.5, 
    #            y + 0.5*h - crop_sz*0.5, 
    #            crop_sz, crop_sz]
    # 转换为numpy数组
    if isinstance(noisy_crop, torch.Tensor):
        noisy_crop = noisy_crop.numpy().astype(np.uint8)
    if isinstance(original_img, torch.Tensor):
        original_img = original_img.numpy().astype(np.uint8)
    
    # 计算原始裁剪参数
    x, y, w, h = box_extract
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
    resize_factor = output_sz / crop_sz
    
    # 计算裁剪区域坐标（考虑边界填充）
    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    x2 = x1 + crop_sz
    y2 = y1 + crop_sz
    
    # 计算填充量
    x1_pad = max(0, -x1)
    x2_pad = max(x2 - original_img.shape[1], 0)
    y1_pad = max(0, -y1)
    y2_pad = max(y2 - original_img.shape[0], 0)
    
    # 将加噪图像缩回原始裁剪尺寸
    if resize_factor != 1.0:
        restored_crop = cv2.resize(noisy_crop, (crop_sz, crop_sz), interpolation=cv2.INTER_LINEAR)
    else:
        restored_crop = noisy_crop.copy()
    
    # 创建修改区域掩码
    mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
    
    # 计算有效区域
    y_start = max(y1, 0)
    y_end = min(y2, original_img.shape[0])
    x_start = max(x1, 0)
    x_end = min(x2, original_img.shape[1])
    
    # 还原到原始图像
    restored_img = original_img.copy()
    if y_end > y_start and x_end > x_start:
        crop_h = y_end - y_start
        crop_w = x_end - x_start
        
        # 从restored_crop中提取有效区域
        crop_region = restored_crop[y1_pad:y1_pad+crop_h, x1_pad:x1_pad+crop_w]
        
        # 替换原始图像区域
        restored_img[y_start:y_end, x_start:x_end] = crop_region
        
        # 生成掩码
        mask[y_start:y_end, x_start:x_end] = 255
    
    return restored_img