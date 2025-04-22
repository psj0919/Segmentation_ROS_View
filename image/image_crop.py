import cv2
import os





if __name__ == '__main__':
    image_folder = '/Users/parksungjun/Downloads/New_sample/20200731_145104/2'
    output_folder = '/Users/parksungjun/Downloads/New_sample/20200731_145104/2_crop'

    crop_height = 750
    images = sorted([img for img in os.listdir(image_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Could not read image: {image_name}")
            continue

        height, width = img.shape[:2]

        if width == 1920 and height == 1080:
            # 위에서 crop (0 ~ 750)
            cropped = img[0:crop_height, 0:1920]
            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, cropped)
        else:
            print(f"Skipping {image_name}: size is not 1920x1080")
