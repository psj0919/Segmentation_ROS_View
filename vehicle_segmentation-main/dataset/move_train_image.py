import os
import shutil


def move_dataset(sub_path):
    dir_list = os.listdir(sub_path)
    for i in dir_list:
        path = os.path.join(sub_path, i+'/sensor_raw_data/camera')
        for subdir, _, files in os.walk(path):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    src_path = os.path.join(subdir, file)
                    dst_path = os.path.join('/storage/sjpark/vehicle_data/Dataset/val_image', file)
                    shutil.copy(src_path, dst_path)


def total_image(sub_path):
    pre_images = 0
    cur_imges = 0

    # pre_num_image
    dir_list = os.listdir(sub_path)
    for i in dir_list:
        path = os.path.join(sub_path, i + '/sensor_raw_data/camera')
        for subdir, _, files in os.walk(path):
            for file in files:
                pre_images += 1

    cur_imges = len(os.listdir('/storage/sjpark/vehicle_data/Dataset/val_image'))

    print("pre_num_images: {}".format(pre_images))
    print("cur_num_images: {}".format(cur_imges))


def move_json_file(path):

    dir_list = os.listdir(path)
    for i in dir_list:
        files = os.path.join(path, i+'/sensor_raw_data/camera')
        for subdir, _, files in os.walk(files):
            for file in files:
                src_path = os.path.join(subdir, file)
                dst_path = os.path.join('/storage/sjpark/vehicle_data/Dataset/json_file', file)
                shutil.copy(src_path, dst_path)




if __name__=='__main__':

    path = '/storage/sjpark/vehicle_data/Dataset2/093.상용_자율주행차_주간_도심도로_데이터/01-1.정식개방데이터/Validation/02.라벨링데이터/VL'
    dst_path = '/storage/sjpark/vehicle_city_data/test_label_json'

    dir_list = os.listdir(path)
    for i in dir_list:
        files = os.path.join(path, i+'/sensor_raw_data/camera')
        for subdir, _, files in os.walk(files):
            for file in files:
                src_path = os.path.join(subdir, file)
                shutil.copy(src_path, dst_path)





