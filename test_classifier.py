import pickle
import os, cv2
import mediapipe as mp
import numpy as np

# 加载模型
##model_dict = pickle.load(open('./model.p', 'rb'))
##model = model_dict['model']
# 加载模型（直接从 modelrfc.p 文件中加载 RandomForest 模型）
with open('./modelknn.p', 'rb') as file:
    modelknn = pickle.load(file)

# 将模型赋值给变量 model
model = modelknn

# 打开摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 初始化 MediaPipe 手部识别模块
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# 标签字典
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 6: 'G', 8: 'I', 9: 'J', 11: 'L', 12: 'M', 14: 'O', 15: 'P', 16: 'Q', 18: 'S', 19: 'T', 23: 'X', 24: 'Y', 25: 'Z'}

while True:

    data_aux = []
    x_ = []
    y_ = []

    # 读取摄像头帧
    ret, frame = cap.read()
    H, W, _ = frame.shape

    # 检查摄像头是否成功读取到帧
    if not ret:
        print("未能从摄像头读取图像。请检查摄像头连接或设备编号。")
        continue  # 跳过本次循环，继续尝试读取

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 处理图像，检测手部位置
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
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

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10


        # 使用模型进行预测
        prediction = model.predict([np.asarray(data_aux)])

        # 处理预测结果，如果标签不在字典中，标记为“未知”
        try:
            predicted_character = labels_dict[int(prediction[0])]
        except KeyError:
            print(f"预测结果不在标签字典中：{int(prediction[0])}")
            predicted_character = "u"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # 显示图像
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下 'q' 键退出
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()
