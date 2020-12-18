# Real-time Facial Expression Emoji Masking with Convolutional Neutral Networks and Homography.
**Authors: [Qinchen Wang](https://qcw171717.github.io/Personal-Website/), [Sixuan Wu](https://github.com/wsxwsx543/), [Tingfeng Xia](https://tingfengx.com)**

## | [Report](./writeup/ba_cloud_report.pdf) | [Code](https://github.com/tingfengx/Emoji-Expression-Mask.PyTorch) | [Demo](https://youtu.be/GCjtXw1y8Pw) | 
## Enviornment
You will need to install ```dlib```, ```imutils```, and ```cv2``` via:
``````
pip install dlib imutils opencv-python
# notice that you will need cmake if it is not yet installed on your machine, 
# you can do so on mac
brew install cmake
# or check https://www.linuxfordevices.com/tutorials/install-cmake-on-linux for linux
``````
See https://pytorch.org/get-started/locally/ for how to install ```torch``` and ```torchvision``` locally. 

## Local Quick Start
The ```app``` folder contains a standalone app that allows you test our pipeline with input from your own webcam! You can do so by 
``````
# clone this repo via git
cd CSC420-Project/app
python main.py 
# a window should pop up and you will see your beautiful face and the masked result. 
``````
Note: Once you start the app, if you don't see the window poping up, that's because we failed to detect your face, please adjust tha angle and make sure your face is complete in the camera. During the execution of the app, if you see that the video is stuck, that's also mostly likely because we didn't detect your face. 

## Pretrained Models
Our implementation uses VGG BA SMALL network, whose model weights are already included in the standalone app. Here are all our trained models, in case you wish to experiment. 
``````
# Xception pretrained weights
https://drive.google.com/drive/folders/1--onPPhUraZ66TynL-t0kkXCHB5xJEJD?usp=sharing
# VGG 11 pretrained weights
https://drive.google.com/drive/folders/1-7ZRhX5FebJipVX8QJrMUmyAit8HOlTd?usp=sharing
# VGG 13 pretrained weights
https://drive.google.com/drive/folders/1-_QBy3oetol16KHse9scxTTtDxWfj-j6?usp=sharing
# VGG 16 pretrained weights
https://drive.google.com/drive/folders/1-rBFFW3wE7yt7C1b2KLScglkvK4sdvD_?usp=sharing
# VGG 19 pretrained weights
https://drive.google.com/drive/folders/1-s15Vj3PvyLdFwypyMARGYDAK6ihccPD?usp=sharing
# VGG BA SMALL pretrained weights (the one that we are using in our report)
# see the report for a detailed discussion 
https://drive.google.com/drive/folders/107MvZVZgDMmjZGVai0edFsTRW2ww8fX6?usp=sharing
# Squeeze net pretrained weights
https://drive.google.com/file/d/1--7YW7iiLA8HRwlBhOgm1Dm9kek-knuW/view?usp=sharing
``````

## Training Your Own Model
Apart from our default VGG BA SMALL network implementation, we have also prepared trained models, in PyTorch, for VGG11, 13, 16, 19, Xception, and Shuffle Net. To load these models see the *Model Preparation* Section above. Alternatively, you can train your own weights for the expression categorization. We used FER2013 as our training data, and you can download it here: (please put them in the ```data``` folder once downloaded)
``````
https://drive.google.com/drive/folders/1sudFiDoV_-Ufie8UXULVA-x7Mz3i04fK?usp=sharing
``````
The ```data.h5``` file provided is a preprocessed version of the FER2013 dataset, which you can use the torch dataloader load directly. You can also customize your preprocessing by editing and then running the ```preprocess_fer2013.py``` file and using the ```fer2013.csv``` file. A ```data.h5``` file should be generated in the ```data/``` folder. It contains your freshly processed dataset. 

Check ```train_and_test.ipynb``` for how to train the model once you have the dataset ready. 

## Other Experiments
See the report for a comprehensive discussion of our experiments with VGG BA SMALL network. Below are some other experiments that we conducted. 

### ShuffleNet experiments
The main file that trains and validates the model is called ```shuffle_net.ipynb```. This file loads data from folders, and the version pushed to the branch experiments with the CK+ dataset. If you want to run the file with FER2013, please follow the guide in the section ```FER2013 dataset preparation``` on where the dataset is stored as image folders.

### SqueezeNet experiments
All experiments related to SqueezeNet are stored in the master branch. The main file that trains and validates the model is called ```csc420 squeezenet.ipynb```. If you would like to play with the model that achieved 97% accuracy on both the training and validation set for the CK+ dataset: 
``````
https://drive.google.com/file/d/1--7YW7iiLA8HRwlBhOgm1Dm9kek-knuW/view?usp=sharing.
``````

The model training, evaluation and saving are implemented in a file called train_and_test.ipynb. After training, the model is stored in a folder called ```FER2013_{model name}``` with filename ```validation_model.t7```. Our pretrained ```VGG_BA_SMALL``` model is stored in Google drive via the link: 
``````
https://drive.google.com/drive/folders/1-s15Vj3PvyLdFwypyMARGYDAK6ihccPD.
``````

#### Test model with your own input
The file named ```load_and_test_qinchen.ipynb``` currently loads images from the folder ```./qinchen``` and makes preditions usinig a pretrained network. You can replace the images in the folder ```./qinchen``` and run this file to see the model prediction on your own face expressions. Note, this is only the second module in our pipeline, so in order to get a satisfactory performance you will have to crop out the face yourself.
