import os
import json
import cv2
import numpy as np
import torch
import torchvision
from dataset.dataset import vehicledata
from loss.loss import cross_entropy2d
from tqdm import tqdm
from copy import deepcopy
from Core.functions import *
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt
from distutils.version import LooseVersion
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch.nn.functional as F


except_classes = ['motorcycle', 'bicycle', 'twowheeler', 'pedestrian', 'rider', 'sidewalk', 'crosswalk', 'speedbump', 'redlane', 'stoplane', 'trafficlight']

CLASSES = [
    'background', 'vehicle', 'bus', 'truck', 'policeCar', 'ambulance', 'schoolBus', 'otherCar',
    'freespace', 'curb', 'safetyZone', 'roadMark', 'whiteLane',
    'yellowLane', 'blueLane', 'constructionGuide', 'trafficDrum',
    'rubberCone', 'trafficSign', 'warningTriangle', 'fence'
]

class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = self.setup_device()
        self.model = self.setup_network()
        self.optimizer = self.setup_optimizer()
        self.train_loader = self.get_dataloader()
        self.val_loader = self.get_val_dataloader()
        self.loss = self.setup_loss()
        self.scheduler = self.setup_scheduler()
        self.global_step = 0
        self.save_path = self.cfg['model']['save_dir']
        self.writer = SummaryWriter(log_dir=self.save_path)
        self.load_weight()

    def setup_device(self):
        if self.cfg['args']['gpu_id'] is not None:
            device = torch.device("cuda:{}".format(self.cfg['args']['gpu_id']) if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        return device

    def get_dataloader(self):
        if self.cfg['dataset']['name'] == 'vehicledata':
            dataset = vehicledata(self.cfg['dataset']['img_path'], self.cfg['dataset']['ann_path'],
                                  self.cfg['dataset']['num_class'], self.cfg['dataset']['size'])
        else:
            raise ValueError("Invalid dataset name...")
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg['args']['batch_size'], shuffle=True,
                                             num_workers=self.cfg['args']['num_workers'])

        return loader

    def get_val_dataloader(self):
        if self.cfg['dataset']['name'] == 'vehicledata':
            val_dataset = vehicledata(self.cfg['dataset']['val_path'], self.cfg['dataset']['val_ann_path'],
                                      self.cfg['dataset']['num_class'], self.cfg['dataset']['size'])
        else:
            raise ValueError("Invalid dataset name...")

        loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             num_workers=self.cfg['args']['num_workers'])
        return loader


    def setup_network(self):
        if self.cfg['args']['network_name'] == "fcn8":
            from model.FCN8s import FCN8s
            return FCN8s(num_class=self.cfg['dataset']['num_class']).to(self.device)
        elif self.cfg['args']['network_name'] == "fcn16":
            from model.FCN16s import FCN16s
            return FCN16s(num_class=self.cfg['dataset']['num_class']).to(self.device)
        elif self.cfg['args']['network_name'] == "fcn32":
            from model.FCN32s import FCN32s
            return FCN32s(num_class=self.cfg['dataset']['num_class']).to(self.device)

    def setup_optimizer(self):
        if self.cfg['solver']['optimizer'] == "sgd":
            optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.cfg['solver']['lr'],
                                        weight_decay=self.cfg['solver']['weight_decay'])
        elif self.cfg['solver']['optimizer'] == "adam":
            optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.cfg['solver']['lr'],
                                         weight_decay=self.cfg['solver']['weight_decay'])
        else:
            raise NotImplementedError("Not Implemented {}".format(self.cfg['solver']['optimizer']))

        return optimizer

    def setup_scheduler(self):
        if self.cfg['solver']['scheduler'] == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.cfg['solver']['step_size'],
                                                        self.cfg['solver']['gamma'])
        elif self.cfg['solver']['scheduler'] == 'cycliclr':
            scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=1e-7,
                                                          max_lr=self.cfg['solver']['lr'],
                                                          step_size_up=int(self.cfg['args']['epochs'] / 5 * 0.7),
                                                          step_size_down=int(self.cfg['args']['epochs'] / 5) - int(self.cfg['args']['epochs'] / 5 * 0.7),
                                                          cycle_momentum=False,
                                                          gamma=0.9)


        return scheduler

    def setup_loss(self):
        if self.cfg['solver']['loss'] == 'crossentropy':
            loss = torch.nn.CrossEntropyLoss(reduction='sum')
        elif self.cfg['solver']['loss'] == 'bceloss':
            loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
        else:
            raise NotImplementedError("Not Implemented {}".format(self.cfg['solver']['loss']))

        return loss

    def load_weight(self):
        if self.cfg['model']['mode'] == 'train':
            pass
        elif self.cfg['model']['mode'] == 'test':
            file_path = self.cfg['model']['resume']
            assert os.path.exists(file_path), f'There is no checkpoints file!'
            print("Loading saved weighted {}".format(file_path))
            ckpt = torch.load(file_path, map_location=self.device)
            resume_state_dict = ckpt['model'].state_dict()

            self.model.load_state_dict(resume_state_dict, strict=True)  # load weights
        else:
            raise NotImplementedError("Not Implemented {}".format(self.cfg['dataset']['mode']))

    def training(self):
        print("start training_model_{}".format(self.cfg['args']['network_name']))
        self.model.train()

        #
        for curr_epoch in range(self.cfg['args']['epochs']):
            #
            if (curr_epoch + 1) % 3 == 0:
                avr_ious, total_accs, cls, org_cls, target_crop_image, pred_crop_image, avr_precision, avr_recall = self.validation()

                for i in range(len(avr_ious)):
                    self.writer.add_scalar(tag='total_ious/{}'.format(cls[i]), scalar_value=avr_ious[i], global_step = self.global_step)

                # Crop Image
                for i in range(len(target_crop_image)):
                    self.writer.add_image('target /' + org_cls[i], trg_to_class_rgb(target_crop_image[i], org_cls[i]),
                                          dataformats='HWC', global_step=self.global_step)
                    self.writer.add_image('pred /' + org_cls[i], pred_to_class_rgb(pred_crop_image[i], org_cls[i]),
                                          dataformats='HWC', global_step=self.global_step)
                # Pixel Acc
                for i in range(len(cls)):
                    self.writer.add_scalar(tag='pixel_accs/{}'.format(cls[i]), scalar_value=total_accs[cls[i]], global_step=self.global_step)

                # precision & recall
                for i in range(len(cls)):
                    self.writer.add_scalar(tag='precision/{}'.format(cls[i]), scalar_value=avr_precision[cls[i]], global_step=self.global_step)
                for i in range(len(cls)):
                    self.writer.add_scalar(tag='recall/{}'.format(cls[i]), scalar_value=avr_recall[cls[i]], global_step=self.global_step)

            #
            for batch_idx, (data, target, label, json) in tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                                                         desc="[Epoch][{}/{}]".format(curr_epoch + 1, self.cfg['args']['epochs']), ncols=80,
                                                         leave=False):
                self.global_step += 1

                data = data.to(self.device)
                target = target.to(self.device)
                label = label.to(self.device)
                label = label.type(torch.long)
                #
                out = self.model(data)
                #
                loss = self.loss(out, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.global_step % self.cfg['solver']['print_freq'] == 0:
                    self.writer.add_scalar(tag='train/loss', scalar_value=loss, global_step=self.global_step)
                if self.global_step % (10 * self.cfg['solver']['print_freq']) == 0:
                    self.writer.add_image('train/train_image', matplotlib_imshow(data[0].to('cpu')),
                                          dataformats='HWC', global_step=self.global_step)
                    self.writer.add_image('train/predict_image',
                                          pred_to_rgb(out[0]),
                                          dataformats='HWC', global_step=self.global_step)
                    self.writer.add_image('train/target_image',
                                          trg_to_rgb(target[0]),
                                          dataformats='HWC', global_step=self.global_step)


            self.scheduler.step()
            self.writer.add_scalar(tag='train/lr', scalar_value=self.optimizer.param_groups[0]['lr'],
                                   global_step=curr_epoch)

            if self.global_step % 2 == 0:
                self.save_model(self.cfg['model']['checkpoint'])
        #

    def validation(self):
        self.model.eval()
        total_ious = []
        total_accs = {}
        cls = []
        cls_count = []
        p_threshold = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15,
                       0.1, 0.05]
        #
        for i in range(len(CLASSES)):
            CLASSES[i] = CLASSES[i].lower()
        for c in CLASSES:
            if c in except_classes:
                pass
            else:
                total_accs[c] = []
        #
        for iter, (data, target, label, idx) in enumerate(self.val_loader):
            cls = []
            total_ious = []
            total_accs = {}
            avr_precision = {}
            avr_recall = {}
            #
            self.global_step += 1
            #
            data = data.to(self.device)
            target = target.to(self.device)
            label = label.to(self.device)

            logits = self.model(data)
            pred = logits.softmax(dim=1).argmax(dim=1).to('cpu')
            pred_ = pred.to(self.device)
            pred_softmax = logits.softmax(dim=1)
            target_ = target.softmax(dim=1).argmax(dim=1).to('cpu')
            file, json_path = load_json_file(int(idx))
            # Iou
            iou = make_bbox(json_path, target_, pred)
            # Crop image
            target_crop_image, pred_crop_image, org_cls = crop_image(target[0], logits[0], json_path)

            for i in range(len(iou)):
                for key, val in iou[i].items():
                    if key in cls:
                        a = cls.index(key)
                        total_ious[a] += val
                        cls_count[a] += 1
                    else:
                        cls.append(key)
                        total_ious.append(val)
                        cls_count.append(1)

            avr_ious = [total / count for total, count in zip(total_ious, cls_count)]
            cls_count.clear()

            # Pixel Acc

            x = pixel_acc_cls(pred[0].cpu(), label[0].cpu(), json_path)
            for key, val in x.items():
                if len(val) > 1:
                    total_accs[key] = sum(val) / len(val)
                    c = CLASSES.index(key)

                else:
                    total_accs[key] = val[0]
                    c = CLASSES.index(key)

            #
            precision, recall = precision_recall(target[0], pred_softmax[0], json_path, threshold=p_threshold)
            for key, val in precision.items():
                for key2, val2 in val.items():
                    if key == 0.5:
                        if len(val2) > 1:
                            avr_precision[key2] = sum(val2) / len(val2)
                        else:
                            avr_precision[key2] = val2[0]

            #
            for key, val in recall.items():
                for key2, val2 in val.items():
                    if key == 0.5:
                        if len(val2) > 1:
                            avr_recall[key2] = sum(val2) / len(val2)
                        else:
                            avr_recall[key2] = val2[0]



        self.model.train()

        return avr_ious, total_accs, cls, org_cls, target_crop_image, pred_crop_image, avr_precision, avr_recall


    def save_model(self, save_path):
        save_file = 'fcn_epochs:{}_optimizer:{}_lr:{}_model{}.pth'.format(self.cfg['args']['epochs'],
                                                                  self.cfg['solver']['optimizer'],
                                                                  self.cfg['solver']['lr'],
                                                                  self.cfg['args']['network_name'])
        path = os.path.join(save_path, save_file)
        torch.save({'model': deepcopy(self.model), 'optimizer': self.optimizer.state_dict()}, path)
        print("Success save")