import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset_synapse import Synapse_dataset
from utils.dataset_CheX import CheX_dataset, ResizeGenerator_CheX
from utils.utils import test_single_volume
from torchvision import transforms
from lib.networks import EMCADNet

import torchmetrics
from torchmetrics import Dice, Accuracy
from torchmetrics.classification import BinaryJaccardIndex

parser = argparse.ArgumentParser()

parser.add_argument('--volume_path', type=str,
                    default='./data/CheXpert_EMCAD/test', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='CheX', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_CheX', help='list dir')

# network related parameters
parser.add_argument('--encoder', type=str,
                    default='pvt_v2_b2', help='Name of encoder: pvt_v2_b2, pvt_v2_b0, resnet18, resnet34 ...')
parser.add_argument('--expansion_factor', type=int,
                    default=2, help='expansion factor in MSCB block')
parser.add_argument('--kernel_sizes', type=int, nargs='+',
                    default=[1, 3, 5], help='multi-scale kernel sizes in MSDC block')
parser.add_argument('--lgag_ks', type=int,
                    default=3, help='Kernel size in LGAG')
parser.add_argument('--activation_mscb', type=str,
                    default='relu6', help='activation used in MSCB: relu6 or relu')
parser.add_argument('--no_dw_parallel', action='store_true', 
                    default=False, help='use this flag to disable depth-wise parallel convolutions')
parser.add_argument('--concatenation', action='store_true', 
                    default=False, help='use this flag to concatenate feature maps in MSDC block')
parser.add_argument('--no_pretrain', action='store_true', 
                    default=False, help='use this flag to turn off loading pretrained enocder weights')
parser.add_argument('--pretrained_dir', type=str,
                    default='./pretrained_pth/pvt/', help='path to pretrained encoder dir')
parser.add_argument('--supervision', type=str,
                    default='mutation', help='loss supervision: mutation, deep_supervision or last_layer')

parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.0001, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", default=True, help='whether to save results during inference')

parser.add_argument('--test_save_dir', type=str, default='predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=2222, help='random seed')
args = parser.parse_args()

class SampleMeanBinaryJaccard(torchmetrics.Metric):
    """先对 batch 内每个样本独立计算 Binary Jaccard Index，
       然后对这些样本分数求平均。
    """
    higher_is_better: bool = True  # IoU 越大越好

    def __init__(self, **kwargs):
        super().__init__(dist_sync_on_step=False)
        # 内部仍然用官方 BinaryJaccardIndex
        self._jac = BinaryJaccardIndex(**kwargs)

        # 累加样本级 IoU 之和与样本计数
        self.add_state("sum_iou", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_items", default=torch.tensor(0),   dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Args:
            preds  : [B, ...]  二值预测
            target : [B, ...]  二值标签
        """
        B = preds.shape[0]

        # 将每个样本展平到一维后单独喂入 BinaryJaccardIndex
        preds_flat  = preds.reshape(B, -1)
        target_flat = target.reshape(B, -1)

        for i in range(B):
            score = self._jac(preds_flat[i], target_flat[i])  # 单样本 IoU
            self._jac.reset()  # 清掉内部状态，下一样本重新计算
            self.sum_iou += score
            self.n_items += 1

    def compute(self):
        # 返回所有样本 IoU 的平均值
        return self.sum_iou / self.n_items

# def inference_QaTa_test(args, model, test_save_path=None):
#     db_test = args.Dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir, nclass=args.num_classes)
#     testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
#     logging.info("{} test iterations per epoch".format(len(testloader)))
#     model.eval()
#     metric_list = 0.0
#     for i_batch, sampled_batch in tqdm(enumerate(testloader)):
#         h, w = sampled_batch["image"].size()[2:]
#         image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

#         logging.info('idx %d case %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))
#     metric_list = metric_list / len(db_test)

#     performance = np.mean(metric_list, axis=0)[0]
#     mean_hd95 = np.mean(metric_list, axis=0)[1]
#     mean_jacard = np.mean(metric_list, axis=0)[2]
#     mean_asd = np.mean(metric_list, axis=0)[3]
#     logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jacard : %f mean_asd : %f' % (performance, mean_hd95, mean_jacard, mean_asd))
#     return "Testing Finished!"

def inference_CheX_test(args, model, test_save_path=None):
    
    macro_dice_meter = Dice(average='samples').cuda()
    macro_miou_meter = SampleMeanBinaryJaccard().cuda()
    micro_dice_meter = Dice().cuda()
    micro_miou_meter = BinaryJaccardIndex().cuda()
    micro_acc_meter = Accuracy(task='binary').cuda()
    
    db_test = CheX_dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir, nclass=args.num_classes, transform=transforms.Compose([ResizeGenerator_CheX(output_size=[args.img_size, args.img_size])]))
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # print(type(sampled_batch["image"]), type(sampled_batch["label"]))
        # print(sampled_batch["image"].size(), sampled_batch["label"].size())
        image = sampled_batch["image"].unsqueeze(0).float().cuda()
        label = sampled_batch["label"].cuda()
        case_name = sampled_batch['case_name'][0]
        # print(image.size(), label.size())
        with torch.no_grad():
            P = model(image)
            pred = torch.softmax(P[-1], dim=1)
            out = torch.argmax(torch.softmax(P[-1], dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            # print(pred_metric.size())
        pred_metric = pred[:, 1]
        macro_dice_meter.update(pred_metric, label.long())
        macro_miou_meter.update(pred_metric, label.long())
        micro_dice_meter.update(pred_metric, label.long())
        micro_miou_meter.update(pred_metric, label.long())
        micro_acc_meter.update(pred_metric, label.long())
        # break
        
    test_macro_dice = macro_dice_meter.compute().item()
    test_macro_miou = macro_miou_meter.compute().item()
    test_micro_dice = micro_dice_meter.compute().item()
    test_micro_miou = micro_miou_meter.compute().item()
    test_micro_acc = micro_acc_meter.compute().item()
    logging.info('Testing performance in best val model: mDice : %f, mIoU : %f, gDice : %f, gIoU : %f, Acc. : %f' % (test_macro_dice, test_macro_miou, test_micro_dice, test_micro_miou, test_micro_acc))
    return "Testing Finished!"

if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'CheX': {
            'Dataset': CheX_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    print(args.no_pretrain)

    if args.concatenation:
        aggregation = 'concat'
    else: 
        aggregation = 'add'
    
    if args.no_dw_parallel:
        dw_mode = 'series'
    else: 
        dw_mode = 'parallel'

    run = 1

    args.exp = args.encoder + '_EMCAD_kernel_sizes_' + str(args.kernel_sizes) + '_dw_' + dw_mode + '_' + aggregation + '_lgag_ks_' + str(args.lgag_ks) + '_ef' + str(args.expansion_factor) + '_act_mscb_' + args.activation_mscb + '_loss_' + args.supervision + '_output_final_layer_Run'+str(run)+'_' + dataset_name + str(args.img_size)
    snapshot_path = "model_pth/{}/{}".format(args.exp, args.encoder + '_EMCAD_kernel_sizes_' + str(args.kernel_sizes) + '_dw_' + dw_mode + '_' + aggregation + '_lgag_ks_' + str(args.lgag_ks) + '_ef' + str(args.expansion_factor) + '_act_mscb_' + args.activation_mscb + '_loss_' + args.supervision + '_output_final_layer_Run'+str(run))
    snapshot_path = snapshot_path.replace('[', '').replace(']', '').replace(', ', '_')
    
    snapshot_path = snapshot_path + '_pretrain' if not args.no_pretrain else snapshot_path
    # snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 50000 else snapshot_path
    # snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 300 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.0001 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    
    model = EMCADNet(num_classes=args.num_classes, kernel_sizes=args.kernel_sizes, expansion_factor=args.expansion_factor, dw_parallel=not args.no_dw_parallel, add=not args.concatenation, lgag_ks=args.lgag_ks, activation=args.activation_mscb, encoder=args.encoder, pretrain= not args.no_pretrain, pretrained_dir=args.pretrained_dir)
    model.cuda()

    #snapshot_path = 'model_pth/'+args.encoder+'_EMCAD_wi_normal_dw_parallel_add_Conv2D_cec_cdc1x1_dwc_cs_ef2_k_sizes_1_3_5_ag3g_relu6_up3_relu_to1_3ch_relu_loss2p4_w1_out1_nlrd_mutation_True_cds_False_cds_decoder_FalseRun'+str(run)+'_Synapse224/'+args.encoder+'_EMCAD_wi_normal_dw_parallel_add_Conv2D_cec_cdc1x1_dwc_cs_ef2_k_sizes_1_3_5_ag3g_relu6_up3_relu_to1_3ch_relu_loss2p4_w1_out1_nlrd_mutation_True_cds_False_cds_decoder_FalseRun'+str(run)+'_50k_epo300_bs6_lr0.0001_224_s2222'
    snapshot = os.path.join(snapshot_path, 'best.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best', 'epoch_'+str(args.max_epochs-1))
    model.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = 'test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(snapshot_path, "predictions")
        test_save_path = os.path.join(args.test_save_dir, snapshot.split('/')[-1].split('.')[0])
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference_CheX_test(args, model, test_save_path)


