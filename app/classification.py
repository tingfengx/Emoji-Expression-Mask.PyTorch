"""
Classification Module

Author: bacloud (Qinchen Wang, Sixuan Wu, Tingfeng Xia)
"""

import vgg_model

def classify(model, image):
    labels = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}
    output = model(image)
    return labels[output.argmax(axis=1).item()]
