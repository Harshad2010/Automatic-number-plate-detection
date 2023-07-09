import numpy as np
import pandas as pd
from glob import glob
import xml.etree.ElementTree as xet #based upon xml read an image or get the filename
import cv2
import os
from shutil import copy
from shutil import move
import warnings
warnings.filterwarnings("ignore")
import sys
from src.exception import ObjectDetectionException
from src.logger import logging

class DataPreprocess :
    def __init__() :
        pass
    
    def preprocess():
        "Preprocesses all data - convert files from XML to text format"
        try:
            # get path of each xml file.
            xmlfiles = glob("src/data_images/*.xml")
            # replace double backward slash(\\) with forward slash(/)
            xmlfiles = [i.replace("\\","/") for i in xmlfiles]
            
            # read xml files
            # from each xml file we need to extract
            # filename, size(width, height), object(name, xmin, xmax, ymin, ymax)
            def xml_to_csv():   
                path = glob('./images/*.xml')
                labels_dict = dict(filepath=[],xmin=[],xmax=[],ymin=[],ymax=[])
                for filename in path:

                    #filename = path[0]
                    info = xet.parse(filename)
                    root = info.getroot()
                    member_object = root.find('object')
                    labels_info = member_object.find('bndbox')
                    xmin = int(labels_info.find('xmin').text)
                    xmax = int(labels_info.find('xmax').text)
                    ymin = int(labels_info.find('ymin').text)
                    ymax = int(labels_info.find('ymax').text)
                    #print(xmin,xmax,ymin,ymax)
                    labels_dict['filepath'].append(filename)
                    labels_dict['xmin'].append(xmin)
                    labels_dict['xmax'].append(xmax)
                    labels_dict['ymin'].append(ymin)
                    labels_dict['ymax'].append(ymax)
                df = pd.DataFrame(labels_dict)
                return df.to_csv('labels.csv',index=False)
            
            # parsing
            def parsing(path):
                parser = xet.parse(path).getroot()
                name = parser.find('filename').text
                filename = f'./images/{name}'# ./images/N1.jpeg

                # width and height
                parser_size = parser.find('size')
                width = int(parser_size.find('width').text)
                height = int(parser_size.find('height').text)
    
                return filename, width, height # ./images/N1.jpeg,1920,1080
            
            # Save each image in train/test folder and respective labels in .txt
            def save_data(folder_name, df):
                values = df[['filename','center_x','center_y','bb_width','bb_height']].values
                for fname, x,y, w, h in values:
                    image_name = os.path.split(fname)[-1]                #filename= "./images/N1.jpeg"
                    txt_name = os.path.splitext(image_name)[0]
                    
                    dst_image_path = os.path.join(folder_name,image_name) #image_name=N1.jpeg
                    dst_label_file = os.path.join(folder_name,txt_name+'.txt') #txt_name=N1.txt
                    
                    # copy each image into the folder
                    copy(fname,dst_image_path)
                    
                    # generate .txt which has label info
                    label_txt = f'0 {x} {y} {w} {h}'
                    with open(dst_label_file,mode='w') as f:
                        f.write(label_txt)
                        f.close()
            
            ### Labels yolo
            ### center_x, center_y, width , height
            
            df = pd.read_csv('labels.csv')
            
            # Adding columns in existing df
            # Columns added - filename, width, height, center_x, center_y, bb_width, bb_height
            
            df[['filename','width','height']] = df['filepath'].apply(parsing).apply(pd.Series)
            
            # center_x, center_y, width , height
            df['center_x'] = (df['xmax'] + df['xmin'])/(2*df['width'])
            df['center_y'] = (df['ymax'] + df['ymin'])/(2*df['height'])

            df['bb_width'] = (df['xmax'] - df['xmin'])/df['width']
            df['bb_height'] = (df['ymax'] - df['ymin'])/df['height']
            
            ### split the data into train and test
            df_train = df.iloc[:200]
            df_test = df.iloc[200:]
            
            train_folder = 'src/model_training/yolov5/data_images/train'
            test_folder = 'src/model_training/yolov5/data_images/test'
            
            # Generate text files in train and test folder
            save_data(train_folder,df_train)
            save_data(test_folder,df_test)
            logging.info("preprocessing completed")
            
        except Exception as e:
            raise ObjectDetectionException(e,sys)
            
            
            
                        
                        
                
            
            
            
            
            
            
            
                    
    
    
