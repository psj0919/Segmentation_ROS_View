import yaml
from pathlib import Path
import os


def Segmenter_param(model):
    if model == 'Seg-S':
        patch_size = 16
        d_model = 384
        n_heads = 6
        n_layers = 12

    elif model == 'Seg-B':
        patch_size = 16
        d_model = 768
        n_heads = 12
        n_layers = 12
        
    elif model == 'Seg-BP8':
        patch_size = 8
        d_model = 768
        n_heads = 12
        n_layers = 12
                
    elif model == 'Seg-L':
        patch_size = 16
        d_model = 1024
        n_heads = 16
        n_layers = 24

    return patch_size, d_model, n_heads, n_layers

def get_config_dict():
    network_name = 'Seg-L'
    patch_size, d_model, n_heads, n_layers = Segmenter_param(network_name)
    dataset = dict(
        network_name = network_name,
        num_class = 21,
        eval_freq =4,
        batch_size = 1,
        image_size= 256,
    )
    model = dict(
        backbone = dict(
            name= " vit_base_patch8_384",
            image_size= (256, 256),
            patch_size= patch_size,  
            d_model= d_model,    
            n_heads= n_heads,
            n_layers= n_layers,
            d_ff = 0,
            normalization= "vit",
            n_cls= dataset['num_class'],
            distilled= False,
        ),
    )

    decoder = dict(
        name = "mask_transformer",
        drop_path_rate = 0.0,
        dropout = 0.1,
        n_layers = 2,
        n_cls = dataset['num_class'],

    )
    config = dict(
        dataset = dataset,
        model = model,
        decoder = decoder,
    )

    return config



