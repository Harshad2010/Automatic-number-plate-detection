## Importing libraries
import cv2
import sys
from src.prediction.yolo_prediction_for_logo import YOLO_Pred1
from src.exception import ObjectDetectionException

## Creating class for replacing the brand logo.
class LogoReplacement():   
    try:
        def __init__(self):
            pass

        def replace_logo(self,image, logo_image):
            
            ## Reading the image
            self.img_cv= image
            self.logo_img= logo_image
            result_image = self.img_cv.copy()
            logo_height, logo_width= self.logo_img.shape[:2]

            ## Creating instance of YOLO_Pred class
            yolo = YOLO_Pred1(onnx_model='src/model_training/best.onnx',
                                data_yaml='src/model_training/data.yaml')

            ## Predicting bbox from YOLO
            img_pred = yolo.predictions(self.img_cv)
            yolo_bbox = img_pred[1]

            ## Extracting left, top, width, height from yolo predicted bbox
            x = yolo_bbox[0][0]
            y = yolo_bbox[0][1]
            w = yolo_bbox[0][2]
            h = yolo_bbox[0][3]

            ## Resizing the logo as per the input image
            resized_logo_yolo = cv2.resize(self.logo_img, (w, h))

            ## Detected license plate from input image
            result_image_yolo= img_pred[0]

            ## Putting logo on input image
            result_image_yolo[y:y+h, x:x+w] = resized_logo_yolo
            
            return result_image_yolo
        
    except Exception as e:
        raise ObjectDetectionException(e,sys)