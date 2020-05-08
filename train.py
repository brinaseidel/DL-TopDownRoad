import os
import random

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [3, 3]
matplotlib.rcParams['figure.dpi'] = 200

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

#!conda install -c conda-forge opencv -y
import cv2 
from yolov3 import *


from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box

from monodepth2.monodepth import MonoDepthEstimator

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iou", type=float, default=0.3, help="iou training threshold")
    parser.add_argument("--lr", type=float, default=10e-6, help="learning rate")
    parser.add_argument("--depth", type=bool, default=False, help="use depth model")
    parser.add_argument("--precomputed", type=bool, default=False, help="depth image precompute")
    opt = parser.parse_args()
    print("iou: ", opt.iou)
    print("lr: ", opt.lr)
    print("depth: ", opt.depth)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(device)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0);


    image_folder = '../data'
    annotation_csv = '../data/annotation.csv'


    # The scenes from 106 - 133 are labeled
    labeled_scene_index = np.arange(106, 134)
    # Split 75/25 into training and validation
    random.shuffle(labeled_scene_index)
    labeled_scene_index_train = labeled_scene_index[0:21]
    labeled_scene_index_val = labeled_scene_index[21:28]
    print("Train scenes: {} \nVal scenes: {}".format(len(labeled_scene_index_train), len(labeled_scene_index_val)))


    if opt.depth && !opt.precomputed:
        # transform = MonoDepthEstimator("/scratch/dy1078/monodepth2/models/mono_model/models/weights_13/")
        transform = MonoDepthEstimator("./models/weights_13/")
    else:
        transform = torchvision.transforms.ToTensor()


    # The labeled dataset can only be retrieved by sample.
    # And all the returned data are tuple of tensors, since bounding boxes may have different size
    labeled_trainset = LabeledDataset(image_folder=image_folder,
                                      annotation_file=annotation_csv,
                                      scene_index=labeled_scene_index_train,
                                      transform=transform,
                                      extra_info=False,
                                      precomputed=opt.precomputed
                                     )
    trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=2, shuffle=True, num_workers=16, collate_fn=collate_fn)
    labeled_valset = LabeledDataset(image_folder=image_folder,
                                      annotation_file=annotation_csv,
                                      scene_index=labeled_scene_index_val,
                                      transform=transform,
                                      extra_info=False,
                                      precomputed=opt.precomputed
                                     )
    valloader = torch.utils.data.DataLoader(labeled_valset, batch_size=2, shuffle=True, num_workers=16, collate_fn=collate_fn)




    # Create model config

    config = "yolov3-spp.cfg"
    hyp = {'giou': 3.54,  # giou loss gain
           'cls': 2.4,  # cls loss gain
           'cls_pw': 1.0,  # cls BCELoss positive_weight
           'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
           'obj_pw': 1.0,  # obj BCELoss positive_weight
           'iou_t': opt.iou,  # iou training threshold
           'lr0': opt.lr,  # initial learning rate (SGD=5E-3, Adam=5E-4)
           'lrf': 0.00005,  # final learning rate (with cos scheduler)
           'momentum': 0.937,  # SGD momentum
           'weight_decay': 0.000484,  # optimizer weight decay
           'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
           'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
           'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
           'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
           'degrees': 1.98 * 0,  # image rotation (+/- deg)
           'translate': 0.05 * 0,  # image translation (+/- fraction)
           'scale': 0.05 * 0,  # image scale (+/- gain)
           'shear': 0.641 * 0}  # image shear (+/- deg)
    print(hyp)



    # Initialize model
    if opt.depth:
        model = Darknet(config, verbose=False, depth=True).to(device)
    else:
        model = Darknet(config, verbose=False).to(device)
    #ONNX_EXPORT = False


    # Intialize optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            if v.is_leaf:
                pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    optimizer = optim.Adam(pg0, lr=hyp['lr0'])
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2



    # Create training function
    def train(train_loader, model, optimizer, criterion, epoch, sixinput):
        
        model.train()

        for batch_idx, (sample, target, road_image) in enumerate(train_loader):

            # Rework target into expected format:
            # A tensor of size [B, 6]
            # where B is the total # of bounding boxes for all observvations in the batch 
            # and 6 is [id, class, x, y, w, h] (class is always 0, since we're not doing classification)
            # Target is originally front left, front right, back left and back right
            # Note: for boxes not aligned with the x-y axis, this will draw a box with the same center but a maximal width-height that *is* aligned
            # The original range is xy values from from -40 to 40. We also rescale so that x values are from 0 to 1
            target_yolo = torch.zeros(0,6)
            for i, obs in enumerate(target):
                boxes = (obs['bounding_box'] + 40)/80
                boxes_yolo = torch.zeros(boxes.shape[0], 6)
                for box in range(boxes.shape[0]):
                    cls = 0
                    x_center = 0.5*(boxes[box, 0, 0] + boxes[box, 0, 3])
                    y_center = 0.5*(boxes[box, 1, 0] + boxes[box, 1, 3])
                    width = max(boxes[box, 0, :]) - min(boxes[box, 0, :])
                    height = max(boxes[box, 1, :]) - min(boxes[box, 1, :])
                    boxes_yolo[box] = torch.tensor([i, cls, x_center, y_center, width, height])
                target_yolo = torch.cat((target_yolo, boxes_yolo), 0)

            # Send to device
            sample = torch.stack(sample).to(device)
            target_yolo = target_yolo.to(device)

            # Make input the correct shape
            if sixinput==False:
                batch_size = sample.shape[0]
                sample = sample.view(batch_size, -1, 256, 306) # torch.Size([3, 18, 256, 306])

            # Run through model
            optimizer.zero_grad()
            output = model(sample)

            # Calculate loss and take step
            loss, loss_items = compute_loss(output, target_yolo, model, hyp) # Note: this is defined in yolov3.py
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss.')
            loss.backward()
            optimizer.step()

            # Log progress
            if batch_idx % 100 == 0:
                print('\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(sample), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                    
        return





    # Create an evaluation function
    def evaluate(val_loader, model, sixinput):
        model.eval()
        losses = []
        tp, fp, fn = 0, 0, 0
        for batch_idx, (sample, target, road_image) in enumerate(val_loader):
            
            # Rework target into expected format:
            # A tensor of size [B, 6]
            # where B is the total # of bounding boxes for all observvations in the batch 
            # and 6 is [id, class, x, y, w, h] (class is always 0, since we're not doing classification)
            # Target is originally front left, front right, back left and back right
            # Note: for boxes not aligned with the x-y axis, this will draw a box with the same center but a maximal width-height that *is* aligned
            # The original range is xy values from from -40 to 40. We also rescale so that x values are from 0 to 1
            # "Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide x_center and width by image width, and y_center and height by image height."
            target_yolo = torch.zeros(0,6)
            for i, obs in enumerate(target):
                boxes = (obs['bounding_box'] + 40)/80
                boxes_yolo = torch.zeros(boxes.shape[0], 6)
                for box in range(boxes.shape[0]):
                    cls = 0
                    x_center = 0.5*(boxes[box, 0, 0] + boxes[box, 0, 3])
                    y_center = 0.5*(boxes[box, 1, 0] + boxes[box, 1, 3])
                    width = max(boxes[box, 0, :]) - min(boxes[box, 0, :])
                    height = max(boxes[box, 1, :]) - min(boxes[box, 1, :])
                    boxes_yolo[box] = torch.tensor([i, cls, x_center, y_center, width, height])
                target_yolo = torch.cat((target_yolo, boxes_yolo), 0)
            
            # Send to device
            sample = torch.stack(sample).to(device)
            target_yolo = target_yolo.to(device)
             
            # Make input the correct shape
            if sixinput==False:
                batch_size = sample.shape[0]
                sample = sample.view(batch_size, -1, 256, 306) # torch.Size([3, 18, 256, 306])
            
            # Run through model
            with torch.no_grad():
                output = model(sample)
            # Calculate loss
            #print(output[0].shape)
            loss, loss_items = compute_loss(output[1], target_yolo, model, hyp) # Note: this is defined in yolov3.py
            losses.append(loss)
        
        # Calculate metrics
        loss = sum(losses)/len(losses)
        
        return loss




    # Train
    min_val_loss = np.inf
    #val_threat_score_hist = []
    val_loss_hist = []

    for epoch in range(30):
        
        # Train for one epoch
        train(trainloader, model, optimizer, None, epoch, sixinput=False)
        
        # Evaluate at the end of the epoch
        print("Evaluating after Epoch {}:".format(epoch))
        #val_loss, val_threat_score = evaluate(valloader, model, loss, sixinput=False)
        #print("Val loss is {:.6f}, threat score is {:.6f}".format(val_loss, val_threat_score))
        model.training= False
        val_loss = evaluate(valloader, model, sixinput=False)
        model.training=True
        print("Val loss is {}".format(val_loss.cpu().detach()))
        
        # If this is the best model so far, save it
        if val_loss < min_val_loss:
            torch.save({
                'epoch': epoch,
                'config': config,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                }, 'models/best_bounding_box_iou_%f_lr_%f_depth_%d.pt' % (opt.iou, opt.lr, opt.depth))
        
        # Save loss 
        val_loss_hist.append(val_loss)
        #val_threat_score_hist.append(val_threat_score)

    checkpoint = torch.load('models/best_bounding_box_iou_%f_lr_%f_depth_%d.pt' % (opt.iou, opt.lr, opt.depth))
    checkpoint['val_loss_hist'] = val_loss_hist
    #checkpoint['val_threat_score_hist'] = val_threat_score_hist
    torch.save(checkpoint, 'models/best_bounding_box_iou_%f_lr_%f_depth_%d.pt' % (opt.iou, opt.lr, opt.depth))