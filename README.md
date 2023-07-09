# Automatic-number-plate-detection

## Introduction:
This project is focused on detecting license plate in images computer vision techniques. The second part of the project is to replace license plate by brand logo.

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

5. Install the required libraries mentioned in the requirements.txt file. (Make sure you're inside the environment)
``` 
pip install -r requirements.txt
```
6. Model is deployed using streamlit. Run "streamlit run app.py" command.

7. In src all the pipeline is created - data preparation, model training and prediction. I've integrated these pipeline with web app.

8. Web app consist of two pages.
   - Page 1 - License plate detection and text generation using OCR.
   - Page 2 - Replacing license plate with brand logo as given

9. In web appplication, for Page1 - upload image from test image folder and get detection and text generation.

10. In web appplication, for Page2 - upload car and logo image from test image folder to replace license plate by logo.


## Demo Video:


## Below are some results:



