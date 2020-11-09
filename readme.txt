To run the script type:

python3 run.py --image path_to_image

or

py run.py --image path_to_image

example python3 run.py --image image.jpg



It is just a code that perform the task of  recognition of mask on the face. It takes input as image and point out the face and tells that face with mask or without mask.
Steps to follow:
1 detect the face using haarcascade_frontalface_default.xml
2 preprocess the data and applied augmentation and make two folders for with mask and without mask
3 make model form CNN and compile
4 train model using train data
5 evaluate model using validation data or many methods.
Dataset is too large to upload, So I have uploaded only coding files.
