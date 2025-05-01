import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from loss_utils import GRASP_MAX_WIDTH, min_mask
from models.tracker import ROT_THRESHOLD, TRANS_THRESHOLD, return_gt_grasp
from pytorch3d.transforms import matrix_to_quaternion


def get_loss(end_points, corr_pred_fine, corr_pred_coarse, training_mask, trans, rot, pose_label, cur_frame, rot_type, nfr):
    total_loss = 0

    loss1, end_points = compute_corrpondence_loss(end_points, corr_pred_coarse, corr_pred_fine, training_mask) #  轨迹关联损失 
    loss2, end_points = compute_rot_trans_loss(end_points, trans, rot, pose_label, training_mask, cur_frame, rot_type, nfr) # 运动补偿损失
    total_loss = total_loss + 10*loss1 + 10*loss2 
    end_points['loss/overall_loss'] = total_loss.detach()
    return total_loss, end_points

# 为每对“历史轨迹-当前抓取点”生成二值关联标签（是否为同一轨迹），作为监督信号。
def compute_part_gt_corrpondence(end_points, training_mask, trans=0.3, rot=0.1):
    # 提取预测的抓取点、索引和相机位姿
    grasp_preds = end_points['batch_grasp_preds']
    seed_inds = end_points['fp2_inds']
    camera_poses = end_points['camera_pose']

    # 获取获取维度信息
    B, Ns, _ = grasp_preds.size()
    B //= 2 

    # 重塑数据以分离两个视角的信息
    grasp_preds = grasp_preds.view(B, 2, Ns, 17)
    camera_poses = camera_poses.view(B, 2, 4, 4)
    seed_inds = seed_inds.view(B, 2, seed_inds.shape[1]).long()
    
    grasp_trans_1, grasp_trans_2 = grasp_preds[:, 0, :, 13:16], grasp_preds[:, 1, :, 13:16]
    grasp_rot_1, grasp_rot_2 = grasp_preds[:, 0, :, 4:13], grasp_preds[:, 1, :, 4:13]
    camera_poses_1, camera_poses_2 = camera_poses[:, 0], camera_poses[:, 1]

    # 准备坐标变换
    camera_poses_1 = camera_poses_1.unsqueeze(1).repeat(1, Ns, 1, 1)
    camera_poses_2 = torch.inverse(camera_poses_2).unsqueeze(1).repeat(1, Ns, 1, 1)
    
    # 创建源抓取变换矩阵
    source_grasp_mat = torch.zeros_like(camera_poses_1)
    source_grasp_mat[:, :, 3, 3] = 1
    source_grasp_trans = grasp_trans_1
    source_grasp_rot = grasp_rot_1.view(B, Ns, 3, 3)

    # 填充源抓取变换矩阵
    source_grasp_mat[:, :, :3, 3] = source_grasp_trans
    source_grasp_mat[:, :, :3, :3] = source_grasp_rot

    # 计算视角变换后的抓取点,相机视角1到相机视角2,再从相机视角2到
    grasp_gt = torch.matmul(camera_poses_2, camera_poses_1)
    grasp_gt = torch.matmul(grasp_gt, source_grasp_mat)

    # translation corrpondence，计算平移对应关系
    grasp_trans_1 = grasp_gt[:, :, :3, 3].unsqueeze(2)  # (B, Ns, 1, 3)
    grasp_trans_2 = grasp_trans_2.unsqueeze(1)  # (B, 1, Ns, 3)
    trans_corrpondence = torch.norm(grasp_trans_1-grasp_trans_2, dim=3) # (B, Ns, Ns)
    end_points['batch_grasp_trans_corrpondence'] = trans_corrpondence.detach()

    # rotation corrpondence，计算旋转对应关系
    grasp_rot_1 = grasp_gt[:, :, :3, :3].unsqueeze(2) # (B, Ns, 1, 3, 3)
    grasp_rot_2 = grasp_rot_2.reshape(B, Ns, 3, 3).unsqueeze(1) # (B, 1, Ns, 3, 3)
    rot_corrpondence = torch.matmul(grasp_rot_1, grasp_rot_2.transpose(3,4)) # (B, Ns, Ns, 3, 3)
    rot_corrpondence = torch.diagonal(rot_corrpondence, dim1=3, dim2=4).sum(dim=3) # (B, Ns, Ns)
    rot_corrpondence = torch.clamp((rot_corrpondence-1)/2, -1, 1) # (B, Ns, Ns)
    rot_corrpondence = torch.acos(rot_corrpondence) # (B, Ns, Ns)
    end_points['batch_grasp_rot_corrpondence'] = rot_corrpondence.detach()

    # compute valid grasp corrpondence，计算综合度量并应用掩码演码
    trans_corrpondence = trans_corrpondence / GRASP_MAX_WIDTH
    rot_corrpondence = 0.4 * (rot_corrpondence / np.pi)
    # 应用阈值判断有效对应关系
    trans_valid_mask = (trans_corrpondence) <= trans
    rot_valid_mask = rot_corrpondence <= rot
    valid_mask = trans_valid_mask & rot_valid_mask

    # compute overall grasp corrpondence，计算综合度量并应用训练掩码
    average_corrpondence = trans_corrpondence + rot_corrpondence
    grasp_mask = valid_mask
    for b in range(B):
        grasp_mask[b, ~training_mask[b]] = 0 # 对每个批次，将非训练点的掩码值设为0

    return grasp_mask.cuda(), average_corrpondence

# 衡量模型是否能正确关联当前帧的抓取点和历史轨迹
# 让模型学会判断当前帧的每个抓取点属于哪条历史轨迹，即“抓取点身份分配，时序一致性”
def compute_corrpondence_loss(end_points, corr_pred_coarse, corr_pred_fine, training_mask, gamma=0.1, thre=0.1, tau=0.1, symmetric=False):
    # 此处关于函数2次调用相同输入不同输出的疑问提了 issue1 
    corr_label_fine, aver_corr_fine = compute_part_gt_corrpondence(end_points, training_mask)
    corr_label_coarse, aver_corr_coarse = compute_part_gt_corrpondence(end_points, training_mask)

    corr_mask_fine = (torch.sum(corr_label_fine, dim=2) > 0).cuda()
    corr_loss_fine = compute_multi_label_loss(corr_pred_fine, corr_label_fine, training_mask) # 预测和真实对应关系损失

    corr_mask_coarse = (torch.sum(corr_label_coarse, dim=2) > 0).cuda()
    corr_loss_coarse = compute_multi_label_loss(corr_pred_coarse, corr_label_coarse, training_mask) # 预测和真实对应关系损失


    end_points['loss/grasp_corrpondence_fine_loss'] = corr_loss_fine.detach()
    end_points['loss/grasp_corrpondence_coarse_loss'] = corr_loss_coarse.detach()

    # 计算精度指标
    grasp_corrpondence_fine_top1_prec_list = []
    grasp_corrpondence_fine_top5_prec_list = []
    grasp_corrpondence_coarse_top1_prec_list = []
    
    for b in range(corr_label_fine.shape[0]): # 遍历每个批次，逐批次计算精度指标
        # fine
        corr_pred_mask = corr_mask_fine[b] # 获取当前批次的有效抓取点掩码
        pos_cnt_fine = torch.sum(corr_label_fine[b].float(), dim=1) # 计算每个点真是对应点数
        corr_pred_mask = corr_pred_mask & (pos_cnt_fine.int() > 0) # 更新掩码，确认只选择有对应点的抓取点

        corr_label_mask = corr_label_fine[b][corr_pred_mask]
        corr_pred_values = corr_pred_fine[b][corr_pred_mask]
        
        # 计算top1精度
        corr_pred_fine_top1_values, corr_pred_fine_top1_indices = corr_pred_values.topk(k=1, dim=1)
        corr_pred_fine_top1_label = corr_label_mask.gather(1, corr_pred_fine_top1_indices)
        grasp_corrpondence_fine_top1_prec = corr_pred_fine_top1_label.float().mean()
        grasp_corrpondence_fine_top1_prec_list.append(grasp_corrpondence_fine_top1_prec) 

        # 计算top5精度
        corr_pred_fine_top5_values, corr_pred_fine_top5_indices = corr_pred_values.topk(k=5, dim=1)
        corr_pred_fine_top5_label = corr_label_mask.gather(1, corr_pred_fine_top5_indices)
        grasp_corrpondence_fine_top5_prec = (corr_pred_fine_top5_label).any(dim=1).float().mean()
        grasp_corrpondence_fine_top5_prec_list.append(grasp_corrpondence_fine_top5_prec)

        # coarse 粗糙层级精度计算
        corr_pred_mask = corr_mask_coarse[b]
        pos_cnt_coarse = torch.sum(corr_label_coarse[b].float(), dim=1)
        corr_pred_mask = corr_pred_mask & (pos_cnt_coarse.int() > 0)

        corr_label_mask = corr_label_coarse[b][corr_pred_mask]
        corr_pred_values = corr_pred_coarse[b][corr_pred_mask]

        _, corr_pred_coarse_top1_indices = corr_pred_values.topk(k=1, dim=1)
        corr_pred_coarse_top1_label = corr_label_mask.gather(1, corr_pred_coarse_top1_indices)
        grasp_corrpondence_coarse_top1_prec = corr_pred_coarse_top1_label.float().mean()

        grasp_corrpondence_coarse_top1_prec_list.append(grasp_corrpondence_coarse_top1_prec)

    # 记录精度指标并返回总损失
    end_points['prec/grasp_corrpondence_coarse_rank1'] = sum(grasp_corrpondence_coarse_top1_prec_list) / corr_label_coarse.shape[0]
    end_points['prec/grasp_corrpondence_rank1'] = sum(grasp_corrpondence_fine_top1_prec_list) / corr_label_fine.shape[0]
    end_points['prec/grasp_corrpondence_rank5'] = sum(grasp_corrpondence_fine_top5_prec_list) / corr_label_fine.shape[0]

    return corr_loss_fine + corr_loss_coarse, end_points

# 计算预测的抓取点轨迹（平移和旋转）与真实轨迹之间的损失，衡量模型对抓取点运动（平移和旋转）的补偿预测是否准确。
def compute_rot_trans_loss(end_points, trans, rot, pose_label, training_mask, cur_frame, rot_type, nfr):
    # 确定时间窗口范围
    length = min(cur_frame, nfr)
    if cur_frame < nfr+2:
        left = 1+int(cur_frame/(nfr+1))
    else:
        left = cur_frame - nfr + 1

    # 获取目标抓取点轨迹和查询抓取点轨迹
    target_pose = pose_label[0]
    query_pose = pose_label[left:cur_frame+1]
    target_grasp = end_points['batch_grasp_preds'].view(2,2,1024,17)[:,0,...] # 重塑为[2,2,1024,17]，表示2个批次，2个视角，1024个抓取，每个抓取17维特征配制
    
    # 计算每个查询抓取点对应的gt抓取点
    grasp_gt_list = []

    for i in range(len(query_pose)):
        grasp_gt_list.append(return_gt_grasp(target_grasp, target_pose, query_pose[i]))
    grasp_gt = torch.stack(grasp_gt_list).permute(1,2,0,3).contiguous()

    # 提取平移和旋转
    trans_gt = grasp_gt[..., :3]
    rot_gt = grasp_gt[..., 3:12]
    trans = trans[..., -length:, :]
    rot = rot[:, :, -length:, ...]

    # 计算平移损失
    trans_loss = compute_l1_loss(trans, trans_gt, training_mask)

    # 计算旋转损失
    if rot_type == '6d':
        rot_loss = compute_l1_loss(rot, rot_gt.reshape(rot_gt.shape[0],rot_gt.shape[1],rot_gt.shape[2], 3,3), training_mask)
    else:
        rot_loss = compute_rot_loss(rot, rot_gt.reshape(rot_gt.shape[0],rot_gt.shape[1],rot_gt.shape[2], 3,3), training_mask)
    
    # 记录损失
    end_points['loss/rot_loss'] = rot_loss.detach()
    end_points['loss/trans_loss'] = trans_loss.detach()
    loss = trans_loss + rot_loss

    return loss, end_points


def compute_l1_loss(corr_pred, corr_gt, mask):
    loss = nn.SmoothL1Loss(reduction='none')
    output = loss(corr_pred, corr_gt)
    output = output[mask].mean()

    return output


def compute_rot_loss(pred, gt, mask):
    
    rot_z_pi = torch.tensor([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]).to(gt.device)
    gt_equal = matrix_to_quaternion(gt @ rot_z_pi)
    gt = matrix_to_quaternion(gt)

    loss = torch.min((1.0 - (pred * gt).sum(-1).abs()),
                     (1.0 - (pred * gt_equal).sum(-1).abs()))
    loss = loss[mask].mean()
    
    return loss


def compute_multi_label_loss(corr_pred, corr_gt, mask):
    criterion = nn.BCELoss(reduction='none')
    activation = nn.Sigmoid()
    loss = criterion(activation(corr_pred), corr_gt.float())

    pos_cnt = torch.sum(corr_gt.float(), dim=2) # (B, Ns)
    target_mask = (mask & (pos_cnt.int() > 0))

    loss = torch.sum(loss, dim=2) / (corr_pred.shape[2])
    loss = (loss * target_mask).mean()

    return loss
