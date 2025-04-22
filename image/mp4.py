import cv2
import glob
import os

input_path = '/Volumes/NO NAME/BlackBox'
output_path_root = '/Volumes/NO NAME/BlackBox2'

# .avi 파일 전체 가져오기
avi_files = glob.glob(os.path.join(input_path, '*.avi'))

for avi_file in avi_files:
    cap = cv2.VideoCapture(avi_file)
    filename = os.path.basename(avi_file).split('.')[0]
    output_path = os.path.join(output_path_root, filename + '.mp4')

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30  # fallback to 30 if fps is 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f'✔ 변환 완료: {output_path}')