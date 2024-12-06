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
        image_size=256,
        size=(256, 256)
    )
    args = dict(
        gpu_id='0',
        network='fcn8',
        network_name="a0",
        num_workers=6
    )
    solver = dict(
        deploy= True
    )
    model = dict(
        resume='/storage/sjpark/vehicle_data/checkpoints/repvggfcn/256/a0/RepVGG_fcn_epochs:50_optimizer:adam_lr:0.0001_modela0_max_prob_mAP.pth',  # weight_file
        mode='test',
        save_dir='/storage/sjpark/vehicle_data/runs/repvggfcn/test/fcn8/256/{}'.format('a0'),
        tensor_name='test_RepVGG_{}'.format('a0')
    )
    config = dict(
        args=args,
        dataset=dataset,
        solver = solver,
        model=model
    )

    return config
