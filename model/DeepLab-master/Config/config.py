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


def get_config_dict():
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
        gpu_id='0',
        batch_size=8,
        epochs=100,
        num_workers=6,
        network_name='DeepLabV3+'
    )
    solver = dict(
        backbone = 'resnet50',
        output_stride = 16,
        optimizer="adam",
        scheduler='cycliclr',
        step_size=5,
        gamma=0.95,
        loss="crossentropy",
        lr=1e-4,
        weight_decay=5e-4,
        print_freq=20,
    )
    model = dict(
        resume='',  # weight_file
        mode='train',
        save_dir='/storage/sjpark/vehicle_data/runs/deeplab/train/256/resnet50',
        checkpoint='/storage/sjpark/vehicle_data/checkpoints/deeplab/256'  # checkpoint_path
    )
    config = dict(
        args=args,
        dataset=dataset,
        solver=solver,
        model=model
    )

    return config
