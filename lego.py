import hub
import time
from mindstorms import Motor

motor_name = ['A', 'C', 'B', 'D']
motor_init = [340, 359, 359, 359]
tobes = [340, 359, 359, 359]

motors = []
for i in range(4):
 motors.append(Motor(motor_name[i]))
 motors[i].run_to_position(motor_init[i], "clockwise", 100)

time.sleep(1)
motors_offset = []

def get_angles():
 return [hub.port.A.device.get(), hub.port.C.device.get(), hub.port.B.device.get(), hub.port.D.device.get()]

angles = get_angles()

for i in range(4):
 motors_offset.append(angles[i][1])

def speed_filter(asis, tobe):
 diff = tobe - asis
 mm = 80
 if abs(diff) <= 7: return 0
 elif abs(diff) <= 20:   mm = 30
 elif abs(diff) <= 30:   mm = 50
 elif abs(diff) <= 45:   mm = 60
 speed = diff * abs(diff)
 speed *= 20
 speed = min(mm, speed)
 speed = max(-mm, speed)
 return int(speed)

def gmove():
 angles = get_angles()
 for i in range(4):
  angle = angles[i][1] - motors_offset[i] + 360
  speed = speed_filter(angle, tobes[i])
  motors[i].start_at_power(speed)

tobes = [340, 359, 359, 359]

while True:
 msg = hub_runtime.BT_VCP.read()
 gmove()
 if msg:
  angle = msg.decode()
  angle = angle.split("FIN")[0].split("f")
  # print(angle)
  for i in range(1,5): tobes[i - 1] = int(angle[i])












