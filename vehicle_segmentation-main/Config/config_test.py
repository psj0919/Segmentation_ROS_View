def dataset_info(dataset_name='vehicledata'):
    if dataset_name == 'vehicledata':
        train_path = "/storage/sjpark/vehicle_data/Dataset/train_image/"
        ann_path = "/storage/sjpark/vehicle_data/Dataset/ann_train/"
        val_path = '/storage/sjpark/vehicle_data/Dataset/val_image/'
        val_ann_path = '/storage/sjpark/vehicle_data/Dataset/ann_val/'
        test_path = '/storage/sjpark/vehicle_data/Dataset/test_image/'
        test_ann_path = '/storage/sjpark/vehicle_data/Dataset/ann_test/'
        json_file = '/storage/sjpark/vehicle_data/Dataset/json_file/'
        num_class = 21
    else:
        raise NotImplementedError("Not Implemented dataset name")

    return dataset_name, train_path, ann_path, val_path, val_ann_path, test_path, test_ann_path, num_class


def get_test_config_dict():
    dataset_name = "vehicledata"
    name, img_path, ann_path, val_path, val_ann_path, test_path, test_ann_path, num_class, = dataset_info(dataset_name)


    dataset = dict(
        name=name,
        img_path=img_path,
        ann_path=ann_path,
        val_path=val_path,
        val_ann_path=val_ann_path,
        test_path=test_path,
        test_ann_path=test_ann_path,
        num_class=num_class,
        image_size = 256,
        size= (256, 256)
    )
    args = dict(
        gpu_id='1',
        network_name="fcn32",
        num_workers=6
    )
    model = dict(
        resume='/storage/sjpark/vehicle_data/checkpoints/FCN32/256/fcn_epochs:200_optimizer:adam_lr:0.0001_modelfcn32.pth',  # weight_file
        mode='test',
        save_dir='/storage/sjpark/vehicle_data/runs/FCN/test/fcn32/256/test_afterremove',   # runs_file
    )
    config = dict(
        args=args,
        dataset=dataset,
        model=model
    )

    return config
