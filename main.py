
"""
2021-08-01 가위바위보 AI 제작기입니다.

main.py : PC -> LEGO로 적합한 동작을 지시합니다.
lego.py : 신호를 받고 그에 맞는 상태로 모터를 움직입니다.
ai.py : 가위바위보 예측 분류기를 학습합니다.

"""
# %%

import json
import time
import random
import threading
import cv2 as cv
import winsound as sd
import mediapipe as mp
import numpy as np

from keras.models import load_model
model = load_model("ai_model")

# %%
# window에서 serial 통신하기 위해서는 이래 2개의 패키지를 설치해야 한다.
# pip install serial
# pip install pyserial
# 블루투스 장비의 COM 포트는 "추가 Bluetooth 옵션"에서 확인 가능하다
import serial
ser = serial.Serial("COM6", 115200)

# %%
cap = cv.VideoCapture(1)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
# %%

def angle_between(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    v21 = (x1 - x2, y1 - y2)
    v23 = (x3 - x2, y3 - y2)
    dot = v21[0] * v23[0] + v21[1] * v23[1]
    det = v21[0] * v23[1] - v21[1] * v23[0]
    theta = np.rad2deg(np.arctan2(det, dot))
    return theta

 
def beepsound(time):
    fr = 2000           # range : 37 ~ 32767
    du = int(time*1000) # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)


"""
class RockPaperScissors : 가위바위보를 위한 class 입니다.
"""
class RockPaperScissors:
    def __init__(self, index):

        self.index = index

        self.open_count = 0
        self.img = None
        self.img_bak = None

        self.points = None

        self.ai_input = None
        self.ai_input_bak = None

    # wait : 게임 시작을 기다림, 키보드 입력... 후에 카메라 손 들어오는걸로 변경?
    def wait(self):
        ser.write(f'f{340}f{359}f{359}f{359}FIN'.encode('utf-8')) # 바위
        time.sleep(1.5)
        # input("게임을 시작하려면 Enter를 눌러주세요.")

    # hadn_tracking : 쓰레드를 통해 웹캠 데이터를 별도로 처리하며 전역변수로 데이터를 전달
    def hand_tracking(self):

        fourcc = cv.VideoWriter_fourcc(*'DIVX')
        out = cv.VideoWriter(f'output/video{self.index}.avi', fourcc, 25.0, (640,480))

        while self.tracking_flag:
            success, img = cap.read()
            self.img = img
            imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            points = {}
            ai_input = []
            open_count = 0

            if results.multi_hand_landmarks:

                for handLms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

                        points[id] = (cx, cy)

                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                

            if points:

                fingers = []
                fingers.append( abs(angle_between(points[1], points[2], points[4])) )
                fingers.append( abs(angle_between(points[5], points[6], points[8])) )
                fingers.append( abs(angle_between(points[9], points[10], points[12])) )
                fingers.append( abs(angle_between(points[13], points[14], points[16])) )
                fingers.append( abs(angle_between(points[17], points[18], points[20])) )

                for _, point in points.items():
                    ai_input.append((point[0] - points[0][0]) / 640)
                    ai_input.append((point[1] - points[0][1]) / 480)

                for finger in fingers:
                    ai_input.append(finger / 180)
                    if finger > 165:
                        open_count += 1

                # print(f'열린 손: {open_count} 엄지: {finger1} 검지: {finger2} 중지: {finger3} 약지: {finger4} 새끼: {finger5}')

            self.open_count = open_count
            self.ai_input = ai_input
            
            cv.putText(img, self.msg, (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            cv.imshow("Image", img)
            out.write(img)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            
        out.release()
        cv.destroyAllWindows()

    # ready : 3, 2, 1 카운트 해주고 어떤 값을 보낼지 판단
    def ready(self):

        def beep02():
            beepsound(0.4)

        print("3", self.open_count)
        self.msg = "Ready 3"
        beep02()
        time.sleep(0.6)
        
        print("2", self.open_count)
        self.msg = "Ready 2"
        beep02()
        time.sleep(0.4)
        
        print("1", self.open_count)
        self.msg = "Ready 1"

        t1 = threading.Thread(target=beep02)
        t1.daemon = True
        t1.start()
        # beep02()
        
        time.sleep(0.25)
        self.img_bak = self.img
        self.ai_input_bak = self.ai_input

        # ai = random.choice("RPS")

        user_predict = np.argmax(model.predict(np.array([self.ai_input_bak])))
        user = "RPS"[user_predict]
        ai = "RPS"[(user_predict + 1) % 3]

        print('user', user, 'ai', ai)
        
        time.sleep(0.2)
        # if self.open_count == 0:
        #     ai = "P"
        # elif self.open_count > 3:
        #     ai = "S"
        # else:
        #     ai = "R"
        return ai

    # send_signal : LEGO rps 값 전달
    # rps in ["R", "P", "S"]
    def send_signal(self, ai):
        if ai == "R":
            ser.write(f'f{340}f{359}f{359}f{359}FIN'.encode('utf-8')) # 바위
        elif ai == "P":
            ser.write(f'f{270}f{20}f{20}f{20}FIN'.encode('utf-8')) # 보
        else:
            ser.write(f'f{340}f{20}f{20}f{359}FIN'.encode('utf-8')) # 가위

    # judge : 사용자의 손을 인신해서 가위, 바위, 보를 판단합니다.
    def judge(self, ai):

        if self.open_count == 0:
            user = "R"
            self.ai_input_bak.append(0)
        elif self.open_count > 3:
            user = "P"
            self.ai_input_bak.append(1)
        else:
            user = "S"
            self.ai_input_bak.append(2)

        index = ("RPS".index(ai) - "RPS".index(user)) % 3
        if index == 0:
            self.msg = f"DRAW!! AI: {ai} USER: {user}"
            self.winner = "DRAW"
        elif index == 1:
            self.msg = f"AI WIN!! AI: {ai} USER: {user}"
            self.winner = "AI"
        else:
            self.msg = f"USER WIN!! AI: {ai} USER: {user}"
            self.winner = "USER"

        print(self.msg)
        # 0 R 바위
        # 1 P 보
        # 2 S 가위

    # game : game 진행
    # wait (키보드 인터럽트) > ready (3 2 1 부저) > send_signal > 승패 판단
    def game_start(self):

        self.msg = "Ready"
        self.tracking_flag = True

        t1 = threading.Thread(target=self.hand_tracking)
        t1.daemon = True
        t1.start()
        
        self.wait()
        ai = self.ready()
        self.send_signal(ai)
        self.judge(ai)

        time.sleep(1)
        self.tracking_flag = False

        time.sleep(1)
        
        # cv.imshow("img_bak", self.img_bak)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

# %%

# if __name__ == "__main__":

num_of_games = 30
ai_win_count = 0
user_win_count = 0

ai_input = []

# time.sleep(10)

for i in range(num_of_games):
    env = RockPaperScissors(i)
    env.game_start()

    print( '--- Round', i, env.winner, '---')
    ai_input.append(env.ai_input_bak)
    print(env.ai_input_bak)
    
    if env.winner == 'AI':
        ai_win_count += 1
    elif env.winner == 'USER':
        user_win_count += 1

print(f'AI {ai_win_count} 승 {user_win_count} 패 {num_of_games - (ai_win_count + user_win_count)} 무')
print(f'AI 승률 : {round(100 * ai_win_count / num_of_games, 2)}')

# print(ai_input)
# %%
with open('ai_data.json', 'w') as f:
    json.dump(ai_input, f)

# %%
