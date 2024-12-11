import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os
import time

# 设置环境变量来选择GPU并限制使用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 选择 GPU 0，因为只有一个设备（Orin）
torch.cuda.set_device(0)  # 设置为 GPU 0
torch.cuda.set_per_process_memory_fraction(0.3, device=0)  # 设置最大内存为 50% (4GB 在 8GB GPU 中)

# 初始化 ROS 节点
rospy.init_node('yolov8_realsense_detection', anonymous=True)

# 初始化发布器
distance_pub = rospy.Publisher('distance', String, queue_size=10)
image_pub = rospy.Publisher('/yolov8_raw', Image, queue_size=10)  # 发布检测后的图像

# 初始化 CvBridge
bridge = CvBridge()

# 加载 YOLOv8 模型到 GPU
model = YOLO('/home/nvidia/yolov8n.pt').to('cuda')

# 深度图数据
depth_frame = None

# 相机内参：焦距f，假设已知
focal_length = 1000  # 以像素为单位，具体值需根据相机标定得出
car_height = 1.6  # 车高，单位：米
person_height = 1.7  # 人高，单位：米

# 为每个目标维护历史距离（动态更新）
history_distances = {}

# 加权移动平均法计算
def weighted_moving_average(new_value, history, alpha=0.7):
    """alpha为加权因子，较大的值使得新值影响更大"""
    if len(history) > 0:
        return alpha * new_value + (1 - alpha) * np.mean(history)
    return new_value

# 回调函数处理深度图
def depth_callback(msg):
    global depth_frame
    try:
        depth_frame = bridge.imgmsg_to_cv2(msg, "32FC1")
    except Exception as e:
        rospy.logerr(f"Error processing depth image: {e}")

# 针孔模型计算距离
def calculate_distance_from_pinhole_model(detection_height, object_height):
    """通过针孔模型估算距离"""
    if detection_height > 0:
        # 估算距离 Z = (f * object_height) / detection_height
        return (focal_length * object_height) / detection_height
    return None

# 获取框的非重叠部分的深度数据
def get_non_overlapping_depth(depth_frame, x1, y1, x2, y2, prev_x1, prev_y1, prev_x2, prev_y2):
    """返回两个框不重叠部分的深度值"""
    # 计算重叠部分的坐标
    overlap_x1 = max(x1, prev_x1)
    overlap_y1 = max(y1, prev_y1)
    overlap_x2 = min(x2, prev_x2)
    overlap_y2 = min(y2, prev_y2)

    # 如果有重叠部分
    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
        # 提取不重叠的深度区域
        depth_region_current = depth_frame[y1:y2, x1:x2]
        depth_region_prev = depth_frame[prev_y1:prev_y2, prev_x1:prev_x2]

        # 排除重叠区域
        depth_region_current = np.delete(depth_region_current, slice(overlap_x1, overlap_x2), axis=1)
        depth_region_prev = np.delete(depth_region_prev, slice(overlap_x1, overlap_x2), axis=1)

        return np.concatenate((depth_region_current, depth_region_prev), axis=0)
    else:
        # 如果没有重叠部分，返回当前框的深度区域
        return depth_frame[y1:y2, x1:x2]

# 回调函数处理彩色图像
def image_callback(msg):
    global depth_frame, history_distances
    start_time = time.time()  # 记录每帧处理的开始时间

    try:
        # 将 ROS 图像消息转换为 OpenCV 格式
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # 使用 YOLO 模型进行检测
        results = model(frame)

        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # 检测框
            cls = result.cls[0].item()               # 类别索引
            conf = result.conf[0].item()             # 置信度
            label = model.names[int(cls)]           # 类别名称

            # 提取检测框区域
            detection_box = frame[y1:y2, x1:x2]

            # 提取深度图中检测框对应区域的深度值
            if depth_frame is not None and 0 <= x1 < depth_frame.shape[1] and 0 <= y1 < depth_frame.shape[0]:
                depth_region = depth_frame[y1:y2, x1:x2]

                # 过滤掉无效的深度值 (值为 0 或者其他异常值)
                valid_depth = depth_region[depth_region > 0]

                if len(valid_depth) > 0:  # 如果有有效的深度值
                    # 选择最近的深度值
                    min_distance = np.min(valid_depth) / 1000  # 转换为米

                    # 对于0-8米的范围，直接使用深度信息
                    if min_distance <= 8.0:
                        final_distance = min_distance
                    # 对于8-10米的范围，使用针孔模型估算
                    elif 8.0 < min_distance <= 10.0:
                        # 计算物体在图像上的高度
                        detection_height = y2 - y1  # 检测框的高度（像素值）
                        # 根据目标类别来选择目标的实际高度（车高或人高）
                        object_height = car_height if label == "car" else person_height
                        # 使用针孔模型估算距离
                        estimated_distance = calculate_distance_from_pinhole_model(detection_height, object_height)

                        # 如果估算的距离有效，则结合加权平均计算最终距离
                        if estimated_distance is not None:
                            final_distance = weighted_moving_average(estimated_distance, history_distances.get(label, []))
                        else:
                            final_distance = min_distance
                    else:
                        final_distance = min_distance
                    
                    # 判断深度值是否在有效范围内，并排除异常值
                    if 0.15 <= final_distance <= 10.0:
                        # 更新目标的历史距离
                        if label not in history_distances:
                            history_distances[label] = []
                        history_distances[label].append(final_distance)

                        # 格式化消息并发布
                        message = f"Label: {label}, Confidence: {conf:.2f}, Distance: {final_distance:.2f}m"
                        distance_pub.publish(message)

                        # 在图像上绘制检测框和信息
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {final_distance:.2f}m", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                else:
                    rospy.logwarn(f"No valid depth data for {label} in the detected region.")
            else:
                rospy.logwarn(f"{label} detected, but depth is unavailable!")

        # 计算帧率
        end_time = time.time()  # 获取结束时间
        fps = 1 / (end_time - start_time)  # 计算每秒钟处理的帧数

        # 在图像上绘制帧率
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 将处理后的图像转换为ROS消息并发布
        ros_image = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image_pub.publish(ros_image)

    except Exception as e:
        rospy.logerr(f"Error processing image: {e}")

# 订阅图像话题
rospy.Subscriber("/camera/color/image_raw", Image, image_callback)
rospy.Subscriber("/camera/depth/image_raw", Image, depth_callback)

# 主循环
rospy.spin()

