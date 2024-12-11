import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import torch

# 初始化 ROS 节点
rospy.init_node('yolov8_realsense_detection', anonymous=True)

# 初始化 CvBridge
bridge = CvBridge()

# 加载 YOLOv8 模型到 GPU
model = YOLO('/home/nvidia/yolov8n.pt').to('cuda')

# 定义回调函数，处理图像数据
def image_callback(msg):
    try:
        # 将 ROS 图像消息转换为 OpenCV 图像
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # 对图像进行目标检测
        results = model(frame)
        
        # 获取标注后的图像（包含检测框）
        annotated_frame = results[0].plot()  # 绘制检测框和标签

        # 可选：显示检测结果
        cv2.imshow('YOLOv8 Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 退出
            rospy.signal_shutdown("User requested shutdown.")
    
    except Exception as e:
        rospy.logerr(f"Error processing image: {e}")

# 订阅 RealSense 相机的图像话题
image_topic = "/camera/color/image_raw"  # 根据你的相机配置，确保这个话题正确
rospy.Subscriber(image_topic, Image, image_callback)

# ROS 主循环
rospy.spin()

# 释放 OpenCV 显示窗口
cv2.destroyAllWindows()

