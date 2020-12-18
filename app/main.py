"""
Main Demo Module

Author: bacloud (Qinchen Wang, Sixuan Wu, Tingfeng Xia)
"""

import cv2
import pose
import numpy as np
import dlib
import vgg_model
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
import time


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    # from video stream
    cnt = 0
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
    model = vgg_model.VGG("VGG_ba_small")
    model.load_state_dict(torch.load("./models/vgg_ba_small_model.t7", map_location=torch.device('cpu'))['net'])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(44),
        transforms.CenterCrop(44),
        transforms.ToTensor()
        # transforms.Normalize((0.5),(0.3))
    ])
    t = []
    while True:
        cnt += 1
        # print(cnt)
        # cnt = cnt % 10
        # if cnt == 0:
        #     continue
        # cnt += 1
        ret, frame = cap.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # we really only need the shape, so neutral picked from random : )
        emoji = cv2.imread("./emojis/neutral.png")
        emoji_points = np.array([
            (int(emoji.shape[0] / 2), 430),  # philtrum
            (int(emoji.shape[0] / 2), 380 + 180),  # chin
            (190, 250),  # left eye
            (450, 250),  # righteye
            (250, 460),  # leftmouth
            (emoji.shape[0] - 250, 460)  # rightmouth
        ])
        # print(np.max(frame))
        start = time.time()
        res = pose.masking(frame, emoji_points, detector, predictor, model, transform)
        if res is None:
            print(res)
            continue
        t.append(time.time() - start)
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL) # adjustable window size
        out = cv2.hconcat([frame, res])
        cv2.imshow("Frame", out)
        # quit the frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # cleanup
    cv2.destroyAllWindows()
    print(len(t))
    print(np.mean(t))
