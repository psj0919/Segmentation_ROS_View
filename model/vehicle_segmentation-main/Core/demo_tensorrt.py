import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import sys
import time
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# TensorRT 엔진 로드 및 설정
class TensorRTModel:
    def __init__(self, engine_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate device memory for inputs and outputs
        self.input_shape = (1, 3, 256, 256)  # Modify based on your model
        self.output_shape = (1, 256, 256)   # Modify based on your model's output
        self.d_input = cuda.mem_alloc(np.prod(self.input_shape) * np.float32().nbytes)
        self.d_output = cuda.mem_alloc(np.prod(self.output_shape) * np.float32().nbytes)
        self.bindings = [int(self.d_input), int(self.d_output)]

    def preprocess(self, image):
        # Resize and normalize the image
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image

    def postprocess(self, output):
        # Get the predicted class map
        pred = np.argmax(output, axis=1)[0]
        return pred

    def infer(self, image):
        # Preprocess image
        input_tensor = self.preprocess(image)

        # Copy data to device
        cuda.memcpy_htod(self.d_input, input_tensor)

        # Execute inference
        self.context.execute_v2(self.bindings)

        # Copy result back to host
        output_tensor = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output_tensor, self.d_output)

        # Postprocess output
        return self.postprocess(output_tensor)

# ROS 노드 클래스
class DetectedClassNode:
    def __init__(self, model):
        self.model = model
        self.bridge = CvBridge()
        self.input_image = rospy.Subscriber("/fps_controller/image_raw", Image, self.callback)
        self.result_image = rospy.Publisher("/detected_class/result_image", Image, queue_size=1)

        # Color table for visualization
        self.color_table = {i: (i * 10, i * 20, i * 30) for i in range(21)}  # Example color map

    def callback(self, data):
        try:
            # Convert ROS image to OpenCV format
            cv_input_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            # Run inference
            start_time = time.time()
            pred_class_map = self.model.infer(cv_input_image)
            print(f"Inference time: {time.time() - start_time:.4f} seconds")

            # Convert prediction to RGB
            pred_rgb = self.pred_to_rgb(pred_class_map)

            # Combine input image with prediction for visualization
            result_image = cv2.addWeighted(cv_input_image, 0.6, pred_rgb, 0.4, 0)

            # Publish the result
            msg_result_image = self.bridge.cv2_to_imgmsg(result_image, encoding="bgr8")
            self.result_image.publish(msg_result_image)

        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")

    def pred_to_rgb(self, pred):
        # Map each class to its color
        h, w = pred.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in self.color_table.items():
            rgb_image[pred == cls_id] = color
        return rgb_image

# Main function
def main():
    rospy.init_node("class_detector", anonymous=True)

    # Load TensorRT engine
    engine_path = "/home/parksungjun/ros_ws/src/vehice_project/src/model_fixed_batch.onnx"
    trt_model = TensorRTModel(engine_path)

    # Start ROS node
    detector_node = DetectedClassNode(trt_model)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")

if __name__ == "__main__":
    main()
