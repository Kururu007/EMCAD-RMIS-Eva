import argparse
import logging
import os
import random
import sys
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

from utils.dataset_synapse import Synapse_dataset, RandomGenerator
from utils.dataset_QaTa import QaTa_dataset, RandomGenerator_QaTa
from utils.dataset_CheX import CheX_dataset, RandomGenerator_CheX
from utils.utils import powerset, one_hot_encoder, DiceLoss, val_single_volume

import torchmetrics
from torchmetrics import Dice, Accuracy
from torchmetrics.classification import BinaryJaccardIndex

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
            
def inference(args, model, best_performance):
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, nclass=args.num_classes)
    
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = val_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(db_test)
    performance = np.mean(metric_list, axis=0)
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, best_performance))
    return performance

def inference_QaTa_val(args, model, best_performance):
    
    macro_dice_meter = Dice(average='samples').cuda()
    macro_miou_meter = SampleMeanBinaryJaccard().cuda()
    micro_dice_meter = Dice().cuda()
    micro_miou_meter = BinaryJaccardIndex().cuda()
    micro_acc_meter = Accuracy(task='binary').cuda()
    
    db_test = QaTa_dataset(base_dir=args.volume_path, split="val", list_dir=args.list_dir, nclass=args.num_classes, transform=None)
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
    performance = test_macro_dice
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, best_performance))
    logging.info('Testing performance in val model: mDice : %f, mIoU : %f, gDice : %f, gIoU : %f, Acc. : %f' % (test_macro_dice, test_macro_miou, test_micro_dice, test_micro_miou, test_micro_acc))
    return performance

def inference_CheX_val(args, model, best_performance):
    
    macro_dice_meter = Dice(average='samples').cuda()
    macro_miou_meter = SampleMeanBinaryJaccard().cuda()
    micro_dice_meter = Dice().cuda()
    micro_miou_meter = BinaryJaccardIndex().cuda()
    micro_acc_meter = Accuracy(task='binary').cuda()
    
    db_test = CheX_dataset(base_dir=args.volume_path, split="val", list_dir=args.list_dir, nclass=args.num_classes, transform=None)
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
    performance = test_macro_dice
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, best_performance))
    logging.info('Testing performance in val model: mDice : %f, mIoU : %f, gDice : %f, gIoU : %f, Acc. : %f' % (test_macro_dice, test_macro_miou, test_micro_dice, test_micro_miou, test_micro_acc))
    return performance

def trainer_synapse(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", nclass=args.num_classes,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1 and args.n_gpu > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            print(image_batch.shape, label_batch.shape)
            print(torch.unique(label_batch))
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
            
            P = model(image_batch, mode='train')

            if  not isinstance(P, list):
                P = [P]
            if epoch_num == 0 and i_batch == 0:
                n_outs = len(P)
                out_idxs = list(np.arange(n_outs)) #[0, 1, 2, 3]#, 4, 5, 6, 7]
                if args.supervision == 'mutation':
                    ss = [x for x in powerset(out_idxs)]
                elif args.supervision == 'deep_supervision':
                    ss = [[x] for x in out_idxs]
                else:
                    ss = [[-1]]
                print(ss)
            
            loss = 0.0
            w_ce, w_dice = 0.3, 0.7
          
            for s in ss:
                iout = 0.0
                if(s==[]):
                    continue
                for idx in range(len(s)):
                    iout += P[s[idx]]
                loss_ce = ce_loss(iout, label_batch[:].long())
                loss_dice = dice_loss(iout, label_batch, softmax=True)
                loss += (w_ce * loss_ce + w_dice * loss_dice)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # we did not use this
            lr_ = base_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            

            if iter_num % 50 == 0:
                logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss.item(), lr_))
                
        logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss.item(), lr_))
        
        save_mode_path = os.path.join(snapshot_path, 'last.pth')
        torch.save(model.state_dict(), save_mode_path)
        
        performance = inference(args, model, best_performance)
        
        save_interval = 50

        if(best_performance <= performance):
            best_performance = performance
            save_mode_path = os.path.join(snapshot_path, 'best.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

def trainer_qata(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    db_train = QaTa_dataset(
        base_dir=args.root_path, 
        list_dir=args.list_dir, 
        split="train", 
        nclass=args.num_classes,
        transform=transforms.Compose([RandomGenerator_QaTa(output_size=[args.img_size, args.img_size])])
    )
    
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1 and args.n_gpu > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
            # print(image_batch.size())
            P = model(image_batch, mode='train')

            if  not isinstance(P, list):
                P = [P]
            if epoch_num == 0 and i_batch == 0:
                n_outs = len(P)
                out_idxs = list(np.arange(n_outs)) #[0, 1, 2, 3]#, 4, 5, 6, 7]
                if args.supervision == 'mutation':
                    ss = [x for x in powerset(out_idxs)]
                elif args.supervision == 'deep_supervision':
                    ss = [[x] for x in out_idxs]
                else:
                    ss = [[-1]]
                # print(ss)
            
            loss = 0.0
            w_ce, w_dice = 0.3, 0.7
          
            for s in ss:
                iout = 0.0
                if(s==[]):
                    continue
                for idx in range(len(s)):
                    iout += P[s[idx]]
                loss_ce = ce_loss(iout, label_batch[:].long())
                loss_dice = dice_loss(iout, label_batch, softmax=True)
                loss += (w_ce * loss_ce + w_dice * loss_dice)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # we did not use this
            lr_ = base_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            
            # break
            if iter_num % 10 == 0:
                logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss.item(), lr_))
                
        logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss.item(), lr_))
        
        save_mode_path = os.path.join(snapshot_path, 'last.pth')
        torch.save(model.state_dict(), save_mode_path)
        
        performance = inference_QaTa_val(args, model, best_performance)
        
        save_interval = 50

        if(best_performance <= performance):
            best_performance = performance
            save_mode_path = os.path.join(snapshot_path, 'best.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break
        # break

    writer.close()
    return "Training Finished!"

def trainer_chex(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    db_train = CheX_dataset(
        base_dir=args.root_path, 
        list_dir=args.list_dir, 
        split="train", 
        nclass=args.num_classes,
        transform=transforms.Compose([RandomGenerator_CheX(output_size=[args.img_size, args.img_size])])
    )
    
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1 and args.n_gpu > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
            # print(image_batch.size())
            P = model(image_batch, mode='train')

            if  not isinstance(P, list):
                P = [P]
            if epoch_num == 0 and i_batch == 0:
                n_outs = len(P)
                out_idxs = list(np.arange(n_outs)) #[0, 1, 2, 3]#, 4, 5, 6, 7]
                if args.supervision == 'mutation':
                    ss = [x for x in powerset(out_idxs)]
                elif args.supervision == 'deep_supervision':
                    ss = [[x] for x in out_idxs]
                else:
                    ss = [[-1]]
                # print(ss)
            
            loss = 0.0
            w_ce, w_dice = 0.3, 0.7
          
            for s in ss:
                iout = 0.0
                if(s==[]):
                    continue
                for idx in range(len(s)):
                    iout += P[s[idx]]
                loss_ce = ce_loss(iout, label_batch[:].long())
                loss_dice = dice_loss(iout, label_batch, softmax=True)
                loss += (w_ce * loss_ce + w_dice * loss_dice)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # we did not use this
            lr_ = base_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            
            # break
            if iter_num % 10 == 0:
                logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss.item(), lr_))
                
        logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss.item(), lr_))
        
        save_mode_path = os.path.join(snapshot_path, 'last.pth')
        torch.save(model.state_dict(), save_mode_path)
        
        performance = inference_QaTa_val(args, model, best_performance)
        
        save_interval = 50

        if(best_performance <= performance):
            best_performance = performance
            save_mode_path = os.path.join(snapshot_path, 'best.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break
        # break

    writer.close()
    return "Training Finished!"