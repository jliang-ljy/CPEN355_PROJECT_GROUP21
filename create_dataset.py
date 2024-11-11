import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# 初始化 MediaPipe 手部模型
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
show_one = True
display_frequency = 1498;
counter = 0;

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        # 读取图像并转换为 RGB 格式
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 处理图像以检测手部关键点
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 提取并归一化手部关键点坐标
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_)) 

                # 在图像上绘制手部关键点及连接线
                if (counter%display_frequency) == 0:
                    if show_one == True:
                        mp_drawing.draw_landmarks(
                            img_rgb,  # 图像
                            hand_landmarks,  # 手部关键点
                            mp_hands.HAND_CONNECTIONS,  # 连接关系
                            mp_drawing_styles.get_default_hand_landmarks_style(),  # 默认关键点样式
                            mp_drawing_styles.get_default_hand_connections_style()  # 默认连接线样式
                        )
                    # 显示绘制后的图像
                        plt.figure()
                        plt.imshow(img_rgb)
                        plt.axis('off')  # 隐藏坐标轴
                        plt.show()
                        show_one = False
                    

                counter+=1

            # 保存归一化的关键点数据
            data.append(data_aux)
            labels.append(dir_)

# 保存数据到 pickle 文件
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
