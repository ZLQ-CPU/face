import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import subprocess
import time
import os
import numpy as np
import torch

# 初始化模型
mtcnn = MTCNN(
    min_face_size=45,
    thresholds=[0.82, 0.85, 0.85],
    factor=0.709,
    keep_all=True
)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# 固定路径
base_dir = r"C:\Users\zhouliquan\Desktop\face_monitor"
lib_path = os.path.join(base_dir, "face_lib")
face_database = {}

# 加载人脸库
if os.path.exists(lib_path):
    for person_name in os.listdir(lib_path):
        person_folder = os.path.join(lib_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        feat_list = []
        for pic_name in os.listdir(person_folder):
            pic_path = os.path.join(person_folder, pic_name)
            img = cv2.imread(pic_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(img_rgb)
            if boxes is None or len(boxes) == 0:
                continue
            box = boxes[0].astype(int)
            x1, y1, x2, y2 = box
            # 边界保护，防止裁空
            h, w = img_rgb.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            face_crop = img_rgb[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            face_crop = cv2.resize(face_crop, (160, 160))
            # 构造模型输入
            face_tensor = torch.from_numpy(face_crop).permute(2, 0, 1).float() / 255.0
            with torch.no_grad():
                feat = resnet(face_tensor.unsqueeze(0)).detach().numpy()
            feat_list.append(feat)
        if len(feat_list) > 0:
            face_database[person_name] = np.mean(feat_list, axis=0)
else:
    print("警告：face_lib文件夹不存在，人脸比对关闭")

# 摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 配置参数
last_alert = 0
cool_down = 3
ps_window_time = 3
dist_threshold = 0.8
stable_frame = 0
need_stable = 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(frame_rgb)
    is_known_person = False
    show_name = "未知人员"

    if boxes is not None and len(face_database) > 0:
        for idx, (box, prob) in enumerate(zip(boxes, probs)):
            if prob < 0.78:
                continue
            x1, y1, x2, y2 = box.astype(int)
            # 边界截断，杜绝空白裁剪
            h_img, w_img = frame_rgb.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_img, x2)
            y2 = min(h_img, y2)
            face_crop = frame_rgb[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            face_crop = cv2.resize(face_crop, (160, 160))
            face_tensor = torch.from_numpy(face_crop).permute(2, 0, 1).float() / 255.0
            with torch.no_grad():
                current_feat = resnet(face_tensor.unsqueeze(0)).detach().numpy()
            # 人脸比对
            min_dist = 999.0
            match_name = "陌生人"
            for name, emb in face_database.items():
                dist = np.linalg.norm(current_feat - emb)
                if dist < min_dist:
                    min_dist = dist
                    match_name = name
            # 绘制框
            if min_dist < dist_threshold:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, match_name, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                is_known_person = True
                show_name = match_name
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "陌生人", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # 连续帧防抖
    if is_known_person:
        stable_frame += 1
    else:
        stable_frame = 0
    # 触发弹窗，修改提示文字
    if stable_frame >= need_stable:
        now = time.time()
        if now - last_alert > cool_down:
            powershell_cmd = f'Write-Host "已识别到人脸"; Start-Sleep {ps_window_time}'
            subprocess.Popen(
                ["powershell.exe", "-Command", powershell_cmd],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            last_alert = now

    cv2.imshow("人脸监控", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()