#! /usr/bin/env python3
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
import cv2

class video_publish():
    def __init__(self, video_path):
        self.video_path = video_path
        self.image_pub = rospy.Publisher('video_publish/video_frames', Image, queue_size=1)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(video_path)
    
        if not self.cap.isOpened():            
            rospy.logger("Not Open Video")
            return
    def publish_frame():
        try:        
            ret, frame = self.cap.read()
            if not ret:
                rospy.loginfo("End of video")
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.image_pub.publish(ros_image)
        except:
            print("Error video publish")            
            

def main():
    video_path = '/home/parksungjun/output_video.mp4'
    video_node = video_publish(video_path)
    rospy.init_node('video_publish', anonymous=True)    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shut down")
        cv2.destroyAllWindows()


if __name__== '__main__':
    main()
