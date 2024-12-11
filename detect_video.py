from ultralytics import YOLO

import cv2

import torch



# 检查 GPU 是否可用

if not torch.cuda.is_available():

    print("CUDA GPU 不可用，程序将在 CPU 上运行，速度会较慢。")



# 加载 YOLOv8 模型到 GPU

model = YOLO('/home/nvidia/下载/yolov8n.pt').to('cuda')
#model.set_classes["person", "bicycle", "car", "motorcycle", "bus", "truck", "fire hydrant", "stop sign", "bench", "cat", "dog"]


# 打开视频文件

video_path = '/home/nvidia/下载/1.mp4'

cap = cv2.VideoCapture(video_path)



# 获取视频的帧宽、高和 FPS

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = int(cap.get(cv2.CAP_PROP_FPS))



# 设置输出视频编码格式

output_path = 'output.mp4'  # 保存检测结果的视频

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码器

out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))



# 逐帧处理

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:

        break  # 视频读取完毕



    # 对帧进行目标检测

    results = model(frame)



    # 在帧上绘制检测结果

    annotated_frame = results[0].plot()  # 绘制检测框和标签



    # 保存帧到输出视频

    out.write(annotated_frame)



    # 可选：显示检测过程（实时查看）

    cv2.imshow('YOLOv8 Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 退出

        break



# 释放资源

cap.release()

out.release()

cv2.destroyAllWindows()


