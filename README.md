
You can get contents of the ```data``` folder here: 
``````
https://drive.google.com/drive/folders/1sudFiDoV_-Ufie8UXULVA-x7Mz3i04fK?usp=sharing
``````
You can get the trained models here: 
``````
https://drive.google.com/drive/folders/1--onPPhUraZ66TynL-t0kkXCHB5xJEJD?usp=sharing
https://drive.google.com/drive/folders/1-7ZRhX5FebJipVX8QJrMUmyAit8HOlTd?usp=sharing
https://drive.google.com/drive/folders/1-_QBy3oetol16KHse9scxTTtDxWfj-j6?usp=sharing
https://drive.google.com/drive/folders/1-rBFFW3wE7yt7C1b2KLScglkvK4sdvD_?usp=sharing
https://drive.google.com/drive/folders/1-s15Vj3PvyLdFwypyMARGYDAK6ihccPD?usp=sharing
https://drive.google.com/drive/folders/107MvZVZgDMmjZGVai0edFsTRW2ww8fX6?usp=sharing
``````

Check the top of the ```train_and_test.ipynb``` to see how you can establish the model, and see the bottom of the file on how you can load the model from pretrained. 

FER2013 dataset preparation:
To load from folders, you can create a shortcut to this link on your Google drive: https://drive.google.com/drive/folders/1hPfpU2M3WxdYeDRe2zVoNgLLcHAAfFdl?usp=sharing. It includes the image data from FER2013, already split into train, validation and test set.
To load from a csv, you can create a shortcut to this link on your Google drive: https://drive.google.com/file/d/16bCUgqzr5YSyCSZ_rBhDSSaUfe3bj9E_/view?usp=sharing. 
Alternatively, you can download these files to your local directory.

ShuffleNet experiments:
All experiments related to ShuffleNet are stored in the branch "543". The main file that trains and validates the model is called "shuffle_net.ipynb". This file loads data from folders, and the version pushed to the branch experiments with the CK+ dataset. If you want to run the file with FER2013, please follow the guide in the section "FER2013 dataset preparation" on where the dataset is stored as image folders.

SqueezeNet experiments:
All experiments related to SqueezeNet are stored in the master branch. The main file that trains and validates the model is called "csc420 squeezenet.ipynb". If you would like to play with the model that achieved 97% accuracy on both the training and validation set for the CK+ dataset: https://drive.google.com/file/d/1--7YW7iiLA8HRwlBhOgm1Dm9kek-knuW/view?usp=sharing.

Model preparation:
The model training, evaluation and saving are implemented in a file called train_and_test.ipynb. After training, the model is stored in a folder called FER2013_{model name} with filename "validation_model.t7". Our pretrained VGG_BA_SMALL model is stored in Google drive via the link: https://drive.google.com/drive/folders/1-s15Vj3PvyLdFwypyMARGYDAK6ihccPD.

Test model with your own input:
The file named "load_and_test_qinchen.ipynb" currently loads images from the folder "qinchen" and makes preditions usinig a pretrained network. You can replace the images in the folder "qinchen" and run this file to see the model prediction on your own face expressions. Note, this is only the second module in our pipeline, so in order to get a satisfactory performance you will have to crop out the face yourself.

Test model with the entire pipeline:
