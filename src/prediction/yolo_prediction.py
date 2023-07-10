
# Importing dependecies
import os
import sys
import yaml
import numpy as np
import cv2
from PIL import Image
import easyocr
from yaml.loader import SafeLoader
from src.exception import ObjectDetectionException


# Creating object detection class
class YOLO_Pred():
    try :
        def __init__(self, onnx_model, data_yaml):
            # Load the YAML
            with open(data_yaml,"r") as f:
                data_yaml = yaml.load(f,Loader=SafeLoader)

            self.labels = data_yaml['names']
            self.nc = data_yaml['nc']

            # Load the yolo model
            self.yolo = cv2.dnn.readNetFromONNX(onnx_model) #----check
            self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        def predictions(self,image):
            """Returns predictions/detections from the Object detection algorithm"""
            
            row, column, d = image.shape
            ## Step1 : Convert image into yolo format.
            max_rc = max(row,column)
            input_image = np.zeros((max_rc,max_rc,3), dtype = np.uint8) 
            input_image[0:row ,0:column] = image

            # Step2 : Get detection from YOLO model
            INPUT_WH_YOLO = 640
            blob = cv2.dnn.blobFromImage(input_image,
                                        1/255,
                                        (INPUT_WH_YOLO,INPUT_WH_YOLO),
                                        swapRB=True, 
                                        crop=False)
            self.yolo.setInput(blob)
            preds = self.yolo.forward() 
            detections = preds[0] #(1,25200,5)
            
            # Non maximum supression
            # Step3. Filter detection based on confidence score and probability score
            boxes = []
            confidences = []
            # width and height of the image (input_image)
            image_w, image_h = input_image.shape[:2]
            x_factor = image_w/INPUT_WH_YOLO
            y_factor = image_h/INPUT_WH_YOLO

            #Filter detction as per confidence and probablity scores
            for i in range(len(detections)):
                row = detections[i]
                confidence = row[4] #confidence of detection of license plate

                # if confidence >0.4 then only select the bounding box.
                if confidence>0.4 :
                    class_score = row[5] #probability score of license plate

                    # class_score should be greater than threshold
                    if class_score>0.25 : 
                        cx, cy, w, h = row[0:4] # fetch the bounding box 
                        
                        left = int((cx-0.5*w) * x_factor)
                        top = int((cy-0.5*h) * y_factor)
                        width = int(w*x_factor)
                        height = int(h*y_factor)
                        
                        # construct bounding box from four values
                        # left,  top, width, and height
                        box = np.array([left,top,width,height])
                   
                        confidences.append(confidence)
                        boxes.append(box)

            # Cleaner format.
            boxes_np = np.array(boxes).tolist()
            confidences_np = np.array(confidences).tolist()

            # NMS - Non Max Supression
            index = cv2.dnn.NMSBoxes(bboxes=boxes_np, scores=confidences_np, score_threshold=0.25, nms_threshold=0.45)
            index = np.array(index)
            index = index.flatten()  

            # Step4
            # Draw the bounding box.
            for ind in index:
                # extract bounding box after NMS operation.
                x,y,w,h = boxes_np[ind]
                bb_conf = int(confidences_np[ind]*100)
                #conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
                license_text_bbox, text,roi_thresh = self.extract_text(image,boxes_np[ind]) #OCR
                
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),8)
                #cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,255),-1) 
                #cv2.rectangle(image,(x,y+h),(x+w,y+h+20),(0,0,0),-1)

                #cv2.putText(image, conf_text,(x,y-10), cv2.FONT_HERSHEY_PLAIN, 0.7,(255,255,255),1) 
                cv2.putText(image,text,(x,y+h+27),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                
            return image, boxes_np, index, license_text_bbox, text, roi_thresh
        
        # extracting text
        def extract_text(self,image,bbox):
            """ Function takes image and bbox as an argument and extracts the text from detected region
                Exraction of text is done by OCR techniques using easyocr
            """
            
            # Get roi from bounding box
            # Convert image to grayscale and then use image thresholding for proper text extraction.
            x,y,w,h = bbox
            roi = image[y:y+h, x:x+w]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, roi_thresh = cv2.threshold(roi_gray, 64, 255, cv2.THRESH_BINARY_INV)
            
            reader = easyocr.Reader(['en'], gpu=False)
            output = reader.readtext(roi_thresh)
            # output -[(BBox, text, confidence score)]
            
            for out in output:
                text_bbox, text, text_score = out
                if text_score > 0.4:
                    return text_bbox, text, roi_thresh 
                
            return text_bbox, text, roi_thresh 
        
                        
    except Exception as e:
        raise ObjectDetectionException(e,sys)
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
