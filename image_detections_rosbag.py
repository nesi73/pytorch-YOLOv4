#!python
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import copy
import numpy as np
from geometry_msgs.msg import Polygon, Point32
# from models import predict, get_model_init
# from models import 
from models import predict, get_model_init


import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge



class CrackDetector:
    def __init__(self, debug=False):
        bag = rosbag.Bag('FBM1_flight3.bag', "r")
        self.model, self.class_names=get_model_init()

        bridge=CvBridge()

        for topic, msg, t in bag.read_messages():
            break
            if 'image' in topic:
                print(topic)

        for topic, msg, t in bag.read_messages(topics=['/camera/color/image_raw/compressed']):
            cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.original_image = copy.deepcopy(cv_img)
            cv_image_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            self.canny(cv_img, cv_image_gray)
            cv2.imshow("Image window", cv_img)

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
        cv_blur = cv2.GaussianBlur(cv_image_gray, (5, 5), 0)
        # cv_blur = cv_image_gray
        # canny = cv2.Canny(cv_blur, 150, 180)
        canny = cv2.Canny(cv_blur, 50, 80)
        kernel = np.ones((5,5),np.uint8)
        dilation = cv2.dilate(canny,kernel,iterations = 1)

        cnt, hierarchy = cv2.findContours(dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(cv_image,cnt,-1,(0,0,255), 2)
        currents_detections=[]
        for c in cnt:
            approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(c)
                ratio = float(w)/h
                if ratio >= 0.9 and ratio <= 1.1:
                    if w > 50 and h > 50 and w < 260 and h < 260:
                        for c in currents_detections:
                            iou = self.calculate_iou(c, [x, y, x + w, y + h])
                            print('iou', iou)
                            if iou > 0.5:
                                continue

                        if True:
                            currents_detections.append([x, y, x + w, y + h])
                            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (36,255,12), 2)
                            #self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
                            cv2.imshow("Image window", cv_image)
                            cv2.waitKey(0)
                            print("Tile detected")
                            pol = Polygon()
                            pol.points.append(Point32(x, y, 0))
                            pol.points.append(Point32(x + w, y, 0))
                            pol.points.append(Point32(x + w, y + h, 0))
                            pol.points.append(Point32(x, y + h, 0))
                            #self.bbox_detections.publish(pol)

                            #offset alpha
                            alpha = 10
                            image_cut = self.original_image[y-alpha:y+alpha+h,x-alpha:x+w+alpha]
                            if image_cut.shape[0] <= 10 or image_cut.shape[1] <= 10:
                                continue
                            ratio = float(image_cut.shape[1])/image_cut.shape[0]
                            if ratio < 0.5:
                                continue
                            # print("Image cut size:", image_cut.shape)

                            start_x, start_y = x-alpha, y-alpha
                            image_detect, total_detections = predict(self.model, self.class_names, image_cut)

                            for detection in total_detections:
                                self.original_image = self.paint(detection, self.original_image, start_x, start_y)

                            cv2.imshow("Image window", self.original_image)
                            if total_detections:
                                cv2.imshow("Image detect", image_detect)
                                cv2.waitKey(0)
                                print("Cracks detected")


if __name__ == '__main__':
    CrackDetector(debug=False)
