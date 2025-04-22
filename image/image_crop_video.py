import cv2
import os


if __name__=='__main__':
    image_folder = '/Users/parksungjun/Downloads/New_sample/20200731_145104/2'
    output_video = '/Users/parksungjun/Downloads/New_sample/output_video5.mp4'

    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()

    output_size = (1920, 750)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, 5, output_size)  # FPS = 20

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        cropped_frame = frame[0:750, 0:1920]

        video.write(cropped_frame)
    video.release()
    cv2.destroyAllWindows()

    print(f"Video saved as {output_video}")







