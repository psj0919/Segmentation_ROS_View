import cv2

input_path = '/Users/parksungjun/Desktop/박성준/Segmentation_연구자료/미팅자료/2025 미팅자료/3:31 미팅자료/final_video/final2.mp4'
output_path = '/Users/parksungjun/Desktop/박성준/Segmentation_연구자료/미팅자료/2025 미팅자료/3:31 미팅자료/final_video/fps10_final2.mp4'

cap = cv2.VideoCapture(input_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = 10
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

cap.release()
out.release()
print("✔ FPS 5로 변환 완료")
