import cv2 as cv
import mediapipe as mp
import time
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math
import HandTrackingModule as htm


def set_speaker_volume(volume):
    # script = f"set volume output volume {volume}"
    # subprocess.call(["osascript", "-e", script])
     devices = AudioUtilities.GetSpeakers()
     interface = devices[0].Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
     volume_interface = cast(interface, POINTER(IAudioEndpointVolume))

        # Set the volume (0.0 to 1.0)
     volume_interface.SetMasterVolumeLevelScalar(volume / 100, None)


#def set_brightness(brightness):
 #   script = f"set brightness of display 1 to {brightness}"
  #  subprocess.call(["osascript", "-e", script])


# Example usage: Set the speaker volume to 50%
# set_speaker_volume(50)

# we will use thumb tip id=4 and index fingertip id =8 for volume control
#####################
wcam, hcam = 640, 480
####################


cap = cv.VideoCapture(0)
# cap.set(3,wcam)
# cap.set(4,hcam)
ptime = 0

detector = htm.handDetector(DetectConfidence=0.8)
devices=AudioUtilities.GetSpeakers()
interface=devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume=cast(interface,POINTER(IAudioEndpointVolume))
# tf=volume.GetVolumeRange()
# print(tf)
#-96.0, 0.0, 0.125
min_vol = -96
max_vol = 0.0
b_min=0
b_max=100

while True:
    success, img = cap.read()

    img = detector.findhands(img)
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist) != 0:
        print(lmlist[4], lmlist[8])
        tx1, ty2 = lmlist[4][1], lmlist[4][2]
        fx1, fy2 = lmlist[8][1], lmlist[8][2]
        cx, cy = (tx1 + fx1) // 2, (ty2 + fy2) // 2
        cv.circle(img, (tx1, ty2), 20, (0, 0, 0), cv.FILLED)
        cv.circle(img, (fx1, fy2), 20, (0, 0, 0), cv.FILLED)
        cv.circle(img, (cx, cy), 20, (255, 0, 0), cv.FILLED)
        cv.line(img, (tx1, ty2), (fx1, fy2), (255, 255, 255), 3)

        length = math.hypot(fx1 - tx1, fy2 - ty2)
        vol = np.interp(length, [25, 240], [min_vol, max_vol])
        volume.SetMasterVolumeLevel(vol,None)
        #brightness=np.interp(length,[50,350],[b_min,b_max])

        cv.putText(img, f'length:{int(length)}', (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
        cv.putText(img, f'vol:{int(vol)}', (700, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)
        #cv.putText(img, f'Brightness:{float(brightness)}', (70, 700), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)

        # set_speaker_volume(vol)
        #et_brightness(brightness//100)
        # hand range=50-350
        if length <= 50:
            cv.circle(img, (cx, cy), 20, (0, 255, 0), cv.FILLED)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv.putText(img, f'FPS: {int(fps)}', (40, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)

    cv.imshow("Controller", img)
    cv.waitKey(1)
