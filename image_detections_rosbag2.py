#!python
import cv2
import copy
import numpy as np
# from models import predict, get_model_init
# from models import 
from models import predict, get_model_init
import os
from canny import canny2

class CrackDetector:
    def __init__(self, debug=False):
        
        self.model, self.class_names=get_model_init()

        folder= 'images'
        #folder = '0_ICUAS23_img'
        results_json = []
        cont=0
        #rm content of folder
        for f in os.listdir('resultados_gs'):
            os.remove(os.path.join('resultados_gs', f))
            
        for f in os.listdir(folder):
            cv_img = cv2.imread(os.path.join(folder, f))
            cracks_paints=copy.deepcopy(cv_img)
            cv2.imshow('cv_img', cv_img)
            self.original_image = copy.deepcopy(cv_img)
            cv_image_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            results=canny2(cv_img, cv_image_gray)
            cv2.imshow('cv_img', cv_img)

            for result in results:
                x, y, alpha, image_cut, pts = result
                if abs(pts[0][0] - pts[1][0] )< 80:
                    continue
                start_x, start_y = x-alpha, y-alpha

                # Detect cracks with the image cut (tile)
                cv2.imshow('image_cut_detect', image_cut)
                total_detections = self.detect_cracks(image_cut)

                for detection in total_detections:
                    original_image = self.paint(detection, cracks_paints, start_x, start_y)
                    cv2.imshow('crack', original_image)
                    bbox_ = detection['bbox']
                    bbox = [float(x) * 448 for x in bbox_]
                    results_json.append([f, bbox])
                    cv2.imwrite('./resultados_gs/' + f, original_image)
                break

            cv2.imshow("Image window", cv_img)
            cv2.imwrite('canny_prueba.png', cv_img)
            # cv2.waitKey(0)
            # cv2.waitKey(0)

        #write in .txt file

        with open('results.txt', 'w') as outfile:
            for line in results_json:
                string_line =line[0] + " " + str(line[1][0]) + " " + str(line[1][1]) + " " + str(line[1][2]) + " " + str(line[1][3]) + '\n'
                outfile.write(string_line)
    
    def detect_cracks(self, image_cut):

        image_detect, total_detections = predict(self.model, self.class_names, image_cut)

        return total_detections

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1) 
        total = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1) + (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the area of intersection rectangle
        return intersection / float(total - intersection)

    def paint(self, detection, img, start_x, start_y):
        msg = detection['msg']
        x1, y1, x2, y2 = detection['bbox']
        x1, y1, x2, y2 = x1 + start_x, y1 + start_y, x2 + start_x, y2 + start_y
        width = img.shape[1]
        height = img.shape[0]
        bbox_thick = int(0.6 * (height + width) / 600)
        rgb=(255,0,0)

        t_size = cv2.getTextSize(msg, 0, 0.7, thickness=bbox_thick // 2)[0]
        c1, c2 = (x1,y1), (x2, y2)
        c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)

        cv2.rectangle(img, (x1,y1), (np.int32(c3[0]), np.int32(c3[1])), rgb, -1)
        img = cv2.putText(img, msg, (c1[0], np.int32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,0), bbox_thick//2,lineType=cv2.LINE_AA)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, bbox_thick)

        return img

    def canny(self, cv_image, cv_image_gray):
        original_image = copy.deepcopy(cv_image)
        original_paint=copy.deepcopy(cv_image)
        _, img = cv2.threshold(cv_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, cv_image_gray)
        redBajo1 = np.array([108, 0, 0], np.uint8)
        redAlto1 = np.array([180, 255, 255], np.uint8)

        redBajo1 = np.array([0, 42, 0])
        redAlto1 = np.array([179, 255, 255])
        
        frameHSV = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        maskRed1 = cv2.inRange(frameHSV, redBajo1, redAlto1)

        cv_blur = cv2.GaussianBlur(maskRed1, (3, 3), 0)
        # cv_blur = cv_image_gray
        # canny = cv2.Canny(cv_blur, 150, 180)
        canny = cv2.Canny(cv_blur, 5, 150)

        kernel = np.ones((5,5),np.uint8)
        dilation = cv2.dilate(canny,kernel,iterations = 1)

        cnt, hierarchy = cv2.findContours(dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(cv_image,cnt,-1,(0,0,255), 2)
        currents_detections=[]
        resutls=[]
        for c in cnt:
            x, y, w, h = cv2.boundingRect(c)
            if x > 200 and w > 70 and h > 70 and abs(w-h) < 10:
                # cv2.drawContours(cv_image,c,-1,(0,255,0), 2)
                
                same = False
                for c in currents_detections:
                    iou = self.calculate_iou(c, [x, y, x + w, y + h])
                    print('iou', iou)
                    if iou > 0.5:
                        same = True
                        continue

                if not same:
                    currents_detections.append([x, y, x + w, y + h])
                    cv2.rectangle(original_paint, (x, y), (x + w, y + h), (36,255,12), 2)
                    #self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
                    cv2.waitKey(0)
                    print("Tile detected")
                    cv2.imshow('original_image', original_paint)
                    print(x, y, w, h)
                    cv_image=original_image[y-0:y+0+h,x-0:x+w+0]
                    resutls.append([x, y, 0, cv_image])

        return resutls


if __name__ == '__main__':
    CrackDetector(debug=False)
