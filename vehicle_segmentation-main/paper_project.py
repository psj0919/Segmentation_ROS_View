import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from Config.config import get_config_dict
from dataset.dataset import vehicledata
from model.FCN8s import FCN8s
from model.FCN16s import FCN16s
from model.FCN32s import FCN32s

def get_dataloader():
    if cfg['dataset']['name'] == 'vehicledata':
        dataset = vehicledata(cfg['dataset']['img_path'], cfg['dataset']['ann_path'],
                              cfg['dataset']['num_class'])
    else:
        raise ValueError("Invalid dataset name...")
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['args']['batch_size'], shuffle=True,
                                         num_workers= cfg['args']['num_workers'])
    return loader

def pred_to_rgb(pred):
    assert len(pred.shape) == 3
    #
    pred = pred.to('cpu').softmax(dim=0).argmax(dim=0).to('cpu')
    #
    pred = pred.detach().cpu().numpy()
    #
    pred_rgb = np.zeros_like(pred, dtype=np.uint8)
    pred_rgb = np.repeat(np.expand_dims(pred_rgb[:, :], axis=-1), 3, -1)
    #
    color_table = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (0, 0, 128), 4: (128, 128, 0),
                   5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128), 8: (0, 64, 64),
                   9: (64, 64, 64), 10: (0, 0, 192), 11: (192, 0, 192), 12: (0, 192, 192),
                   13: (192, 192, 192), 14: (64, 128, 0), 15: (192, 0, 128), 16: (64, 128, 128),
                   17: (192, 128, 128), 18: (128, 64, 0), 19: (128, 192, 0), 20: (0, 64, 128)}
    #
    for i in range(0, 21):
        pred_rgb[pred == i] = np.array(color_table[i])

    plt.imshow(pred_rgb)

    return pred_rgb

def trg_to_rgb(target):
    assert len(target.shape) == 3
    #
    target = target.to('cpu').softmax(dim=0).argmax(dim=0).to('cpu')
    #
    target = target.detach().cpu().numpy()
    #
    target_rgb = np.zeros_like(target, dtype=np.uint8)
    target_rgb = np.repeat(np.expand_dims(target_rgb[:, :], axis=-1), 3, -1)
    #
    color_table = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (0, 0, 128), 4: (128, 128, 0),
                   5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128), 8: (0, 64, 64),
                   9: (64, 64, 64), 10: (0, 0, 192), 11: (192, 0, 192), 12: (0, 192, 192),
                   13: (192, 192, 192), 14: (64, 128, 0), 15: (192, 0, 128), 16: (64, 128, 128),
                   17: (192, 128, 128), 18: (128, 64, 0), 19: (128, 192, 0), 20: (0, 64, 128)}
    #
    for i in range(21):
        target_rgb[target == i] = np.array(color_table[i])

    plt.imshow(target_rgb)

    return target_rgb


def matplotlib_imshow(img):

    npimg = img.numpy()

    npimg = (np.transpose(npimg, (1, 2, 0))[:, :, ::-1] * 255).astype(np.uint8)
    plt.imshow(npimg)


if __name__=='__main__':
    cfg = get_config_dict()


    train_loader = get_dataloader()

    for curr_epoch in range(0, 100):
        for batch_idx, (data, target, label) in enumerate(train_loader):
            matplotlib_imshow(data[0])
            trg_to_rgb(target[0])

