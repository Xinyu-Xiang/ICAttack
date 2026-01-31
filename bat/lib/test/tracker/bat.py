import math
from lib.models.bat import build_batrack
from lib.test.tracker.basetracker import BaseTracker
import torch
# from lib.test.tracker.vis.utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import PreprocessorMM
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


from utils import *
import torch.nn.functional as F
# from config import cfg
from torch.autograd import Variable



class BATTrack(BaseTracker):
    def __init__(self, params):
        super(BATTrack, self).__init__(params)
        network = build_batrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)  
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = PreprocessorMM()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        if getattr(params, 'debug', None) is None:
            setattr(params, 'debug', 0)
        self.use_visdom = True #params.debug   
        #self._init_visdom(None, 1)
        self.debug = params.debug
        self.frame_id = 0
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes

        self.count = 0

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        with torch.no_grad():
            self.z_tensor = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            x_tensor = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_tensor, search=x_tensor, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        
        # for debug
        if self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            cv2.putText(image_BGR, 'max_score:' + str(round(max_score, 3)), (40, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
            cv2.imshow('debug_vis', image_BGR)
            cv2.waitKey(1)


        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": max_score}
        else:
            return {"target_bbox": self.state,
                    "best_score": max_score}
        
    #-----------track_adv_ours-----------------------------
    def track_adv_ours(self, image, search_tensor, search_x_tensor, vi_adv, ir_adv, gt, info: dict = None ):
        '''
        input bs=1
        image concat
        vi,ir [1,3,256,256]
        vi_adv,ir_adv [1,3,256,256]
        init_tensor [1,3,256,256]
        gt [1,4]
        '''
        bs = 1
        H, W, _ = image.shape

        self.draw_pred_all = torch.zeros(bs, 5, 4).cuda().to(torch.float64)  # for draw box on tenor
        self.draw_pred_all_clean = torch.zeros(bs, 5, 4).cuda().to(torch.float64)
        lab_batch_draw = torch.zeros(bs, 5, 4).cuda().to(torch.float64)

        img_cat_search = torch.cat((vi_adv, ir_adv), dim=1)  # [n, 6, 256, 256] [0,255]
        img_cat_search_clean = torch.cat((search_tensor, search_x_tensor), dim=1)  # [n, 6, 256, 256] [0,255]
        
        # norm
        img_cat_search_norm = ((img_cat_search / 255.0) - self.preprocessor.mean) / self.preprocessor.std  # (n,6,H,W) search 
        img_cat_search_clean_norm = ((img_cat_search_clean / 255.0) - self.preprocessor.mean) / self.preprocessor.std  # (n,6,H,W) search 


        #init reuslt format
        self.pred_all = torch.zeros(img_cat_search_norm.shape[0], 4).cuda().to(torch.float64)
        self.pred_all_clean = torch.zeros(img_cat_search_norm.shape[0], 4).cuda().to(torch.float64)
        self.score_all = torch.zeros(img_cat_search_norm.shape[0]).cuda().to(torch.float64)
        
        # with torch.no_grad():
            # merge the template and the search
            # run the transformer
        out_dict = self.network.forward(template=self.z_tensor, search=img_cat_search_norm, ce_template_mask=self.box_mask_z) 
        out_dict_clean = self.network.forward(template=self.z_tensor, search=img_cat_search_clean_norm, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        crop_z = math.ceil(math.sqrt(gt[2] * gt[3]) * 4.0)
        resize_factor = self.params.search_size / crop_z
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # add hann windows clean
        pred_score_map_clean = out_dict_clean['score_map']
        response_clean = self.output_window * pred_score_map_clean
        pred_boxes_clean, best_score_clean = self.network.box_head.cal_bbox(response_clean, out_dict_clean['size_map'], out_dict_clean['offset_map'], return_score=True)
        max_score_clean = best_score_clean[0][0].item()
        pred_boxes_clean = pred_boxes_clean.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        crop_z = math.ceil(math.sqrt(gt[2] * gt[3]) * 4.0)
        resize_factor = self.params.search_size / crop_z
        pred_box_clean = (pred_boxes_clean.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state_clean = clip_box(self.map_box_back(pred_box_clean, resize_factor), H, W, margin=10)

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": max_score}
        else:
            return {"target_bbox": self.state,
                    "best_score": max_score,
                    "response_clean": response_clean,
                    "response": response,
                    "clean_bbox":self.state_clean}
    
    def track_adv(self, image, search_tensor, search_x_tensor, vi_adv, ir_adv, gt, info: dict = None):
        '''
        input bs=1
        image concat
        vi,ir [1,3,256,256]
        vi_adv,ir_adv [1,3,256,256]
        init_tensor [1,3,256,256]
        gt [1,4]
        '''

        bs = 1
        H, W, _ = image.shape

        self.draw_pred_all = torch.zeros(bs, 5, 4).cuda().to(torch.float64)  # for draw box on tenor
        self.draw_pred_all_clean = torch.zeros(bs, 5, 4).cuda().to(torch.float64)
        lab_batch_draw = torch.zeros(bs, 5, 4).cuda().to(torch.float64)

        img_cat_search = torch.cat((vi_adv, ir_adv), dim=1)  # [n, 6, 256, 256] [0,255]
        img_cat_search_clean = torch.cat((search_tensor, search_x_tensor), dim=1)  # [n, 6, 256, 256] [0,255]
        
        # norm
        img_cat_search_norm = ((img_cat_search / 255.0) - self.preprocessor.mean) / self.preprocessor.std  # (n,6,H,W) search 
        img_cat_search_clean_norm = ((img_cat_search_clean / 255.0) - self.preprocessor.mean) / self.preprocessor.std  # (n,6,H,W) search 


        #init reuslt format
        self.pred_all = torch.zeros(img_cat_search_norm.shape[0], 4).cuda().to(torch.float64)
        self.pred_all_clean = torch.zeros(img_cat_search_norm.shape[0], 4).cuda().to(torch.float64)
        self.score_all = torch.zeros(img_cat_search_norm.shape[0]).cuda().to(torch.float64)
        
        with torch.no_grad():
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(template=self.z_tensor, search=img_cat_search_norm, ce_template_mask=self.box_mask_z) 

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        crop_z = math.ceil(math.sqrt(gt[2] * gt[3]) * 4.0)
        resize_factor = self.params.search_size / crop_z
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)



        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": max_score}
        else:
            return {"target_bbox": self.state,
                    "best_score": max_score}
        

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return BATTrack
