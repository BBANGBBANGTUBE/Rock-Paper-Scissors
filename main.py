"""
2021-08-01 가위바위보 AI 제작기입니다.

main.py : PC -> LEGO로 적합한 동작을 지시합니다.
lego.py : 신호를 받고 그에 맞는 상태로 모터를 움직입니다.

"""

import time
import random
import cv2 as cv

import winsound as sd

def beepsound(time):
    fr = 2000           # range : 37 ~ 32767
    du = int(time*1000) # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)


"""
class RockPaperScissors : 가위바위보를 위한 class 입니다.
"""
class RockPaperScissors:
    def __init__(self):
        pass

    # wait : 게임 시작을 기다림, 키보드 입력... 후에 카메라 손 들어오는걸로 변경?
    def wait(self):
        input("게임을 시작하려면 Enter를 눌러주세요.")

    # ready : 3, 2, 1 카운트 해주고 어떤 값을 보낼지 판단
    def ready(self):
        print("3")
        beepsound(0.2)
        time.sleep(0.8)
        
        print("2")
        beepsound(0.2)
        time.sleep(0.8)
        
        print("1")
        beepsound(0.5)

        return random.choice("RPS")

    # send_signal : LEGO rps 값 전달
    # rps in ["R", "P", "S"]
    def send_signal(self, rps):
        pass

    def judge(self, rps):
        user = input("RPS를 입력 후 Enter를 눌러주세요. ") # random.choice("RPS")

        # 0 R 바위
        # 1 P 보
        # 2 S 가위
        index = ("RPS".index(rps) - "RPS".index(user)) % 3
        if index == 0:
            print("무승부")
        elif index == 1:
            print(f"AI 승!! AI: {rps} 사람: {user}")
        else:
            print(f"사람 승!! AI: {rps} 사람: {user}")

    # game : game 진행
    # wait (키보드 인터럽트) > ready (3 2 1 부저) > send_signal > 승패 판단
    def game_start(self):

        self.wait()
        rps = self.ready()
        self.send_signal(rps)
        self.judge(rps)



if __name__ == "__main__":

    env = RockPaperScissors()
    env.game_start()