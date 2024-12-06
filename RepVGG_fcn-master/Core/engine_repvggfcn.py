import os
import json
import cv2
import numpy as np
import torch
import torchvision

from Core.functions import *
from dataset.dataset import vehicledata
from tqdm import tqdm
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt
from distutils.version import LooseVersion
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch.nn.functional as F
from setproctitle import *

except_classes = ['motorcycle', 'bicycle', 'twowheeler', 'pedestrian', 'rider', 'sidewalk', 'crosswalk', 'speedbump', 'redlane', 'stoplane', 'trafficlight', 'background']

CLASSES = [
    'background', 'vehicle', 'bus', 'truck', 'policeCar', 'ambulance', 'schoolBus', 'otherCar',
    'freespace', 'curb', 'safetyZone', 'roadMark', 'whiteLane',
    'yellowLane', 'blueLane', 'constructionGuide', 'trafficDrum',
    'rubberCone', 'trafficSign', 'warningTriangle', 'fence'
]

p_threshold = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]

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
    #
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

    def get_test_dataloader(self):
        if self.cfg['dataset']['name'] == 'vehicledata':
            val_dataset = vehicledata(self.cfg['dataset']['test_path'], self.cfg['dataset']['test_ann_path'],
                                      self.cfg['dataset']['num_class'], self.cfg['dataset']['size'])
        else:
            raise ValueError("Invalid dataset name...")

        loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             num_workers=self.cfg['args']['num_workers'])
        return loader

    def setup_network(self):
        optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
        g2_map = {l: 2 for l in optional_groupwise_layers}
        g4_map = {l: 4 for l in optional_groupwise_layers}
        if self.cfg['args']['network'] == 'fcn8':
            from model.RepVGG_fcn8 import RepVGG
            if self.cfg['args']['network_name'] == "a0":
                train_model = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=self.cfg['dataset']['num_class'],
                              width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "a1":
                train_model = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=self.cfg['dataset']['num_class'],
                              width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "a2":
                train_model = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=self.cfg['dataset']['num_class'],
                              width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b0":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                              width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b1":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                              width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b1g2":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                              width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b1g4":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                              width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b2":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                              width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b2g2":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                              width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b2g4":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                              width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b3":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                              width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b3g2":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                              width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=self.cfg['solver']['deploy']).to(self.device)

                return train_model
            elif self.cfg['args']['network_name'] == "b3g4":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                              width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "d2se":
                train_model = RepVGG(num_blocks=[8, 14, 24, 1], num_classes=self.cfg['dataset']['num_class'],
                              width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model

        elif self.cfg['args']['network'] == 'fcn16':
            from model.RepVGG_fcn16 import RepVGG
            if self.cfg['args']['network_name'] == "a0":
                train_model = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "a1":
                train_model = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[1, 1, 1, 2.5], override_groups_map=None,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "a2":
                train_model = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b0":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[1, 1, 1, 2.5], override_groups_map=None,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b1":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[2, 2, 2, 4], override_groups_map=None,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b1g2":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b1g4":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b2":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b2g2":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b2g4":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)

                return train_model
            elif self.cfg['args']['network_name'] == "b3":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[3, 3, 3, 5], override_groups_map=None,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b3g2":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)

                return train_model
            elif self.cfg['args']['network_name'] == "b3g4":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)

                return train_model
            elif self.cfg['args']['network_name'] == "d2se":
                train_model = RepVGG(num_blocks=[8, 14, 24, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
        elif self.cfg['args']['network'] == 'fcn32':
            from model.RepVGG_fcn32 import RepVGG
            if self.cfg['args']['network_name'] == "a0":
                train_model = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "a1":
                train_model = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[1, 1, 1, 2.5], override_groups_map=None,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "a2":
                train_model = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b0":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[1, 1, 1, 2.5], override_groups_map=None,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b1":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[2, 2, 2, 4], override_groups_map=None,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b1g2":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b1g4":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b2":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b2g2":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b2g4":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)

                return train_model
            elif self.cfg['args']['network_name'] == "b3":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[3, 3, 3, 5], override_groups_map=None,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model
            elif self.cfg['args']['network_name'] == "b3g2":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)

                return train_model
            elif self.cfg['args']['network_name'] == "b3g4":
                train_model = RepVGG(num_blocks=[4, 6, 16, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)

                return train_model
            elif self.cfg['args']['network_name'] == "d2se":
                train_model = RepVGG(num_blocks=[8, 14, 24, 1], num_classes=self.cfg['dataset']['num_class'],
                                     width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None,
                                     deploy=self.cfg['solver']['deploy']).to(self.device)
                return train_model

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
        setproctitle(self.cfg['model']['tensor_name'])
        print("start_training_RepVGG_{}".format(self.cfg['args']['network_name']))
        self.model.train()
        #
        tmp = 0  # for prob mAP
        tmp2 = 0  # for avr_mAP
        for curr_epoch in range(self.cfg['args']['epochs']):
            #
            if (curr_epoch + 1) % 3 == 0:
                total_ious, total_accs, cls, org_cls, target_crop_image, pred_crop_image, avr_precision, avr_recall, mAP = self.validation()

                for key, val in total_ious.items():
                    self.writer.add_scalar(tag='total_ious/{}'.format(key), scalar_value=val,
                                           global_step=self.global_step)

                # Crop Image
                for i in range(len(target_crop_image)):
                    self.writer.add_image('target /' + org_cls[i], trg_to_class_rgb(target_crop_image[i], org_cls[i]),
                                          dataformats='HWC', global_step=self.global_step)
                    self.writer.add_image('pred /' + org_cls[i], pred_to_class_rgb(pred_crop_image[i], org_cls[i]),
                                          dataformats='HWC', global_step=self.global_step)
                # Pixel Acc
                for key, val in total_accs.items():
                    self.writer.add_scalar(tag='pixel_accs/{}'.format(key), scalar_value=val,
                                           global_step=self.global_step)

                # precision & recall
                for key, val in avr_precision.items():
                    self.writer.add_scalar(tag='precision/{}'.format(key), scalar_value=val,
                                           global_step=self.global_step)
                for key, val in avr_recall.items():
                    self.writer.add_scalar(tag='recall/{}'.format(key), scalar_value=val,
                                           global_step=self.global_step)
                # mAP
                z = []
                for i in range(len(p_threshold)):
                    self.writer.add_scalar(tag='mAP/{}'.format(str(p_threshold[i])), scalar_value=mAP[str(p_threshold[i])],
                                           global_step=self.global_step)
                    z.append(mAP[str(p_threshold[i])])
                max_z = max(z)

                # for save 1 -> max_prob_mAP
                if max_z > tmp :
                    tmp = max_z
                    self.save_model(self.cfg['model']['checkpoint'])
                # for save 2 -> avr_mAP
                if (sum(z) / len(z)) > tmp2:
                    tmp2 = sum(z) / len(z)
                    self.save_model2(self.cfg['model']['checkpoint'])
            #
            for batch_idx, (data, target, label, json) in tqdm(enumerate(self.train_loader),total=len(self.train_loader), desc="[Epoch][{}/{}]".format(curr_epoch + 1, self.cfg['args']['epochs']), ncols=80, leave=False):
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

        #

    def validation(self):
        self.model.eval()
        total_ious = {}
        total_accs = {}
        avr_precision = {}
        avr_recall = {}
        mAP = {}
        cls = []
        cls_count = []

        p_threshold = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15,
                       0.1, 0.05]

        #
        for iter, (data, target, label, idx) in enumerate(self.val_loader):
            cls = []
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
                    if key in except_classes:
                        pass
                    else:
                        total_ious.setdefault(key, []).append(val)


            # Pixel Acc
            x = pixel_acc_cls(pred[0].cpu(), label[0].cpu(), json_path)
            for key, val in x.items():
                if key in except_classes:
                    pass
                else:
                    if len(val) > 1:
                        total_accs.setdefault(key, []).append(sum(val) / len(val))

                    else:
                        total_accs.setdefault(key, []).append(val[0])

            #
            precision, recall = precision_recall(target[0], pred_softmax[0], json_path, threshold=p_threshold)
            for key, val in precision.items():
                for key2, val2 in val.items():
                    if key == 0.5:
                        if key2 in except_classes:
                            pass
                        else:
                            if len(val2) > 1:
                                avr_precision.setdefault(key2, []).append(sum(val2) / len(val2))
                            else:
                                avr_precision.setdefault(key2, []).append(val2[0])

            #
            for key, val in recall.items():
                for key2, val2 in val.items():
                    if key == 0.5:
                        if key2 in except_classes:
                            pass
                        else:
                            if len(val2) > 1:
                                avr_recall.setdefault(key2, []).append(sum(val2) / len(val2))
                            else:
                                avr_recall.setdefault(key2, []).append(val2[0])


            # mAP
            for key, val in precision.items():
                x = 0
                for key2, val2 in val.items():
                    if len(val2) > 1:
                        x =  x + sum(val2) / len(val2)
                    else:
                        x = x + val2[0]
                mAP.setdefault(str(key), []).append(x / len(key2))


        #
        for k, v in total_ious.items():
            if len(v) > 1:
                total_ious[k] = sum(v) / len(v)
            else:
                total_ious[k] = v
        #
        for k, v in total_accs.items():
            if len(v) > 1:
                total_accs[k] = sum(v) / len(v)
            else:
                total_accs[k] = v
        #
        for k, v in avr_precision.items():
            if len(v) > 1:
                avr_precision[k] = sum(v) / len(v)
            else:
                avr_precision[k] = v
        #
        for k, v in avr_recall.items():
            if len(v) > 1:
                avr_recall[k] = sum(v) / len(v)
            else:
                avr_recall[k] = v
        #
        for k, v in mAP.items():
            if len(v) > 1:
                mAP[k] = sum(v) / len(v)
            else:
                mAP[k] = v

        self.model.train()
        return total_ious, total_accs, cls, org_cls, target_crop_image, pred_crop_image, avr_precision, avr_recall, mAP

    def save_model(self, save_path):
        save_file = 'RepVGG_fcn_epochs:{}_optimizer:{}_lr:{}_model{}_max_prob_mAP.pth'.format(self.cfg['args']['epochs'],
                                                                          self.cfg['solver']['optimizer'],
                                                                          self.cfg['solver']['lr'],
                                                                          self.cfg['args']['network_name'])
        path = os.path.join(save_path, save_file)
        torch.save({'model': deepcopy(self.model.repvgg_model_convert().state_dict()), 'optimizer': self.optimizer.state_dict()}, path)
        print("Success save_max_prob_mAP")

    def save_model2(self, save_path):
        save_file = 'RepVGG_fcn_epochs:{}_optimizer:{}_lr:{}_model{}_total_mAP.pth'.format(self.cfg['args']['epochs'],
                                                                          self.cfg['solver']['optimizer'],
                                                                          self.cfg['solver']['lr'],
                                                                          self.cfg['args']['network_name'])
        path = os.path.join(save_path, save_file)
        torch.save({'model': deepcopy(self.model.repvgg_model_convert().state_dict()), 'optimizer': self.optimizer.state_dict()}, path)
        print("Success save_avr_mAP")

