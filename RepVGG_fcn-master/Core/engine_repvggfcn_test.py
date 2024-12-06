import os
import json
import cv2
import numpy as np
import torch
import torchvision
from dataset.dataset import vehicledata
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
from setproctitle import *

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
        self.test_loader = self.get_test_dataloader()
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


    def test(self):
        setproctitle(self.cfg['model']['tensor_name'])
        self.model.eval()
        print("start testing_model_{}".format(self.cfg['args']['network_name']))

        cls_count = []
        total_avr_acc = {}
        total_avr_iou = {}
        total_avr_precision = {}
        total_avr_recall = {}
        #
        for i in range(len(CLASSES)):
            CLASSES[i] = CLASSES[i].lower()
        #
        for iter, (data, target, label, idx) in enumerate(self.test_loader):
            cls = []
            total_ious = []
            total_accs = {}
            avr_precision = {}
            avr_recall = {}
            p_threshold = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
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
            for i in range(len(avr_ious)):
                total_avr_iou.setdefault(cls[i], []).append(avr_ious[i])
            cls_count.clear()

            # Pixel Acc
            x = pixel_acc_cls(pred[0].cpu(), label[0].cpu(), json_path)

            for key, val in x.items():
                if len(val) > 1:
                    total_accs[key] = sum(val) / len(val)
                    c = CLASSES.index(key)
                    total_avr_acc.setdefault(key, []).append(sum(val) / len(val))
                else:
                    total_accs[key] = val[0]
                    c = CLASSES.index(key)
                    total_avr_acc.setdefault(key, []).append(val[0])

            #
            precision, recall = precision_recall(target[0], pred_softmax[0], json_path, threshold = p_threshold)
            for key, val in precision.items():
                for key2, val2 in val.items():
                    if key == 0.5:
                        if len(val2) > 1:
                            avr_precision[key2] = sum(val2) / len(val2)
                        else:
                            avr_precision[key2] = val2[0]
                    total_avr_precision.setdefault(key2, {}).setdefault(key, []).append(val2[0].cpu())
            #
            for key, val in recall.items():
                for key2, val2 in val.items():
                    if key == 0.5:
                        if len(val2) > 1:
                            avr_recall[key2] = sum(val2) / len(val2)
                        else:
                            avr_recall[key2] = val2[0]
                    total_avr_recall.setdefault(key2, {}).setdefault(key, []).append(val2[0].cpu())


            # if self.global_step % 10 == 0:
            #     #
            #     for i in range(len(avr_ious)):
            #         self.writer.add_scalar(tag='total_ious/{}'.format(cls[i]), scalar_value=avr_ious[i], global_step = self.global_step)
                # Crop Image
                # for i in range(len(target_crop_image)):
                #     self.writer.add_image('target /' + org_cls[i], trg_to_class_rgb(target_crop_image[i], org_cls[i]),
                #                           dataformats='HWC', global_step=self.global_step)
                #     self.writer.add_image('pred /' + org_cls[i], pred_to_class_rgb(pred_crop_image[i], org_cls[i]),
                #                           dataformats='HWC', global_step=self.global_step)
                # # Pixel Acc
                # for i in range(len(cls)):
                #     self.writer.add_scalar(tag='pixel_accs/{}'.format(cls[i]), scalar_value=total_accs[cls[i]], global_step=self.global_step)
                #
                # # precision & recall
                # for i in range(len(cls)):
                #     self.writer.add_scalar(tag='precision/{}'.format(cls[i]), scalar_value=avr_precision[cls[i]], global_step=self.global_step)
                # for i in range(len(cls)):
                #     self.writer.add_scalar(tag='recall/{}'.format(cls[i]), scalar_value=avr_recall[cls[i]], global_step=self.global_step)

                #
                # self.writer.add_image('train/predict_image',
                #                       pred_to_rgb(logits[0]),
                #                       dataformats='HWC', global_step=self.global_step)
                # #
                # self.writer.add_image('train/target_image',
                #                       trg_to_rgb(target[0]),
                #                       dataformats='HWC', global_step=self.global_step)


        # class_per_histogram(total_avr_acc, total_avr_iou, total_avr_precision, total_avr_recall)

        # for key ,val in total_avr_iou.items():
        #     self.writer.add_scalar(tag='total_average_ious/{}'.format(key), scalar_value=sum(val) / len(val),
        #                            global_step=1)
        # for key, val in total_avr_acc.items():
        #     self.writer.add_scalar(tag='total_average_acc/{}'.format(key), scalar_value=sum(val) / len(val),
        #                            global_step=1)
        # for key, val in total_avr_precision.items():
        #     self.writer.add_scalar(tag='total_average_precision/{}'.format(key), scalar_value=sum(val) / len(val),
        #                            global_step=1)
        # for key, val in total_avr_recall.items():
        #     self.writer.add_scalar(tag='total_average_recall/{}'.format(key), scalar_value=sum(val) / len(val),
        #                            global_step=1)

        for key, val in total_avr_precision.items():
            for key2, val2 in val.items():
                path = "/storage/sjpark/vehicle_data/precision_recall_per_class_p_threshold/Repvgg/a0/max_mAP/256/fcn8s/precision/{}/{}_{}.txt".format(key, key, key2)
                np.savetxt(path, total_avr_precision[key][key2], fmt= '%f')

        for key, val in total_avr_recall.items():
            for key2, val2 in val.items():
                path = "/storage/sjpark/vehicle_data/precision_recall_per_class_p_threshold/Repvgg/a0/max_mAP/256/fcn8s/recall/{}/{}_{}.txt".format(key, key, key2)
                np.savetxt(path, total_avr_recall[key][key2], fmt='%f')
