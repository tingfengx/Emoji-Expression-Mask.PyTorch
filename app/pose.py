"""
Pose Estimation Module

Author: bacloud (Qinchen Wang, Sixuan Wu, Tingfeng Xia)
"""

import cv2
import dlib
import numpy as np
import torch
from imutils import face_utils
import PIL.Image
import classification


def process_image(image, detector, predictor):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rectangles = detector(image, 0)

    if len(rectangles) == 0:
        return None

    shape0 = predictor(image, rectangles[0])
    shape0 = np.array(face_utils.shape_to_np(shape0))
    return shape0, rectangles[0]


def circle_mask(n, diff):
    a = n // 2
    b = n // 2
    r = n // 2 - diff

    y, x = np.ogrid[-a:n - a, -b:n - b]
    mask = x * x + y * y <= r * r

    array = np.ones((n, n))
    array[mask] = 255

    return array


def process_emoji(emoji, emoji_points):
    # emoji = cv2.cvtColor(emoji, cv2.COLOR_BGR2RGB)
    return emoji, emoji_points


def masking(image, emoji_points, detector, predictor, model, transforms):
    tmp = process_image(image, detector, predictor)
    if tmp is None:
        return None
    shape0, rect = tmp
    # im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im = image[max(rect.top(), 0):rect.bottom()+25, max(rect.left(), 0):rect.right()]
    im = PIL.Image.fromarray(im)
    im = transforms(im)
    im = torch.unsqueeze(im, 0)
    label = classification.classify(model, im)
    emoji = cv2.imread("./emojis/" + label + ".png")
    # emoji = cv2.imread("./emojis/" + "smile" + ".png")
    image_pts = np.array([(shape0[51, :]),
                          (shape0[8, :]),
                          (shape0[36, :]),
                          (shape0[45, :]),
                          (shape0[48, :]),
                          (shape0[54, :])], dtype="double")
    emoji, emoji_pts = process_emoji(emoji, emoji_points)
    homography, stats = cv2.findHomography(emoji_pts, image_pts)
    warped = cv2.warpPerspective(emoji, homography, (image.shape[1], image.shape[0]))
    warped_circle = cv2.warpPerspective(circle_mask(emoji.shape[0], 10), homography,
                                        (image.shape[1], image.shape[0]))
    warped_circle[warped_circle > 100] = 255.0
    warped_circle[warped_circle <= 100] = 0
    warped_circle = -1 * (warped_circle - 255)
    mask_warped = np.stack([warped_circle] * 3, axis=2)
    image = np.where(mask_warped, image, warped)
    # cv2.rectangle(image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 0, 0), 5)
    return image
