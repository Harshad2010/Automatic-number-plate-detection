# Automatic-number-plate-detection 

## Introduction:
This repository provides an implementation of License Plate Detection and Text Extraction using Optical Character Recognition (OCR) techniques. The goal of this project is to detect license plates in images and extract the alphanumeric characters from the detected plates. The second part of the project is to replace license plate by brand logo.

## Features
- Detects license plates in images using YOLO object detection algorithm.
- Extracts alphanumeric characters from license plates using OCR - easyocr.
- Replaces car number plate by the Brand logo as given.
- Works with jpg and png image format in this application.

## Instructions to run the Project:

1. Clone the repository.
```
git clone 
```
2. Create new enviroment with python==3.8.
```
conda create -p env_name python==3.8 -y
```
3. run command
```
python setup.py install
```
4. Activate the environment.
``` 
conda activate env_name/
```
5. Install the required dependencies mentioned in the requirements.txt file. (Make sure you're inside the environment). Please be patiet installation will take litte time.
``` 
pip install -r requirements.txt
```
6. Model is deployed using streamlit. Run "streamlit run app.py" command.
``` 
streamlit run app.py
```
7. Web app consist of two pages.
   - Page 1 - License plate detection and text generation using OCR.
   - Page 2 - Replacing license plate with brand logo as given

8. In web appplication, for Page1 - upload image from test image folder and get detection and text generation OCR.

9. In web appplication, for Page2 - upload car and logo image from test image folder to replace license plate by logo.

10. Please view all the detected images from web app  in detected images folder.

Please note - Streamlit accepts only jpg and png image format.

## Additonal Information:

1. In src complete pipeline is created :
   - data preparation
   - model training 
   - prediction. 
   
2. I've integrated these pipeline with web application.

## Refrence video to guide for web application:


[![video](https://github.com/Harshad2010/Automatic-number-plate-detection/blob/main/detected%20images/License%20plate.mp4
)


## Below are some results:

<img src = "https://github.com/Harshad2010/Automatic-number-plate-detection/blob/main/detected%20images/4.jpg" alt="MLBC">

<img src = "https://github.com/Harshad2010/Automatic-number-plate-detection/blob/main/detected%20images/4.1.jpg" alt="MLBC">

<img src = "https://github.com/Harshad2010/Automatic-number-plate-detection/blob/main/detected%20images/6.jpg" alt="MLBC">

<img src = "https://github.com/Harshad2010/Automatic-number-plate-detection/blob/main/detected%20images/6.1.jpg" alt="MLBC">

