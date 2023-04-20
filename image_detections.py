import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import copy
import numpy as np
from geometry_msgs.msg import Polygon, Point32
from models import predict, get_model_init
from sensor_msgs.msg import PointCloud2
import message_filters

class image_node():
    def __init__(self):
        rospy.init_node('image_subscriber', anonymous=True)

        print('loading model...')
        self.model, self.class_names=get_model_init()
        print('model loaded')
        self.image_sub = rospy.Subscriber("/red/camera/color/image_raw", Image, self.image_callback)
        # self.depth_image = rospy.Subscriber("/red/camera/depth/image_raw", Image, self.depth_image_callback)
        self.camera_info_sub = rospy.Subscriber("/red/camera/color/camera_info", CameraInfo, self.camera_info_callback)
        # self.depth_points = rospy.Subscriber("/red/camera/depth_registered/points", PointCloud2, self.depth_points_callback)
        
        # public new topic
        self.image_pub = rospy.Publisher("/image/detections", Image, queue_size=10)
        self.detections_cracks = rospy.Publisher("/image/detections_cracks", Image, queue_size=10)
        self.original_image_detections_cracks = rospy.Publisher("/image/original_image_detections_cracks", Image, queue_size=10)

        #publisher an array ros message
        self.bbox_detections = rospy.Publisher("/bbox/detections", Polygon , queue_size=10)

        self.bridge = CvBridge()

        self.cv_image = None
        self.original_image = None
        self.timer = rospy.Timer(rospy.Duration(0.5), self.timer_callback)
        # image_sub = message_filters.Subscriber("/red/camera/color/image_raw", Image)
        # depth_image = message_filters.Subscriber("/red/camera/depth/image_raw", Image)
        # depth_points = message_filters.Subscriber("/red/camera/depth_registered/points", PointCloud2)


        # ts = message_filters.TimeSynchronizer([image_sub, depth_image, depth_points], 10)
        # ts.registerCallback(self.callback)

        rospy.spin()
    
    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        return max(0, xB - xA + 1) * max(0, yB - yA + 1)

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
        canny = cv2.Canny(cv_blur, 150, 180)
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
                    x, y, w, h = cv2.boundingRect(c)
                    if w > 65 and h > 65 and w < 95 and h < 95:
                        print(w, h)
                        same=False
                        for c in currents_detections:
                            iou = self.calculate_iou(c, [x, y, x + w, y + h])
                            print('iou', iou)
                            if iou > 0.5:
                                same=True
                                break

                        if not same:
                            currents_detections.append([x, y, x + w, y + h])
                            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (36,255,12), 2)
                            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
                            pol = Polygon()
                            pol.points.append(Point32(x, y, 0))
                            pol.points.append(Point32(x + w, y, 0))
                            pol.points.append(Point32(x + w, y + h, 0))
                            pol.points.append(Point32(x, y + h, 0))
                            self.bbox_detections.publish(pol)

                            #offset alpha
                            alpha = 10
                            image_cut = self.original_image[y-alpha:y+alpha+h,x-alpha:x+w+alpha]

                            # cv2.imwrite('image_cut.png', image_cut)

                            start_x, start_y = x-alpha, y-alpha
                            image_detect, total_detections = predict(self.model, self.class_names, image_cut)

                            for detection in total_detections:
                                self.original_image = self.paint(detection, self.original_image, start_x, start_y)

                            self.detections_cracks.publish(self.bridge.cv2_to_imgmsg(image_detect, "bgr8"))
                            self.original_image_detections_cracks.publish(self.bridge.cv2_to_imgmsg(image_detect, "bgr8"))
        #                     cv2.imshow("image_detect", image_detect)

                        
        # cv2.imshow("Image canny", canny)
        # cv2.imshow("Detections", cv_image)
        # cv2.imshow("Detections_cracks", original_image)


    def find_countours(self, cv_image, cv_image_gray):
        cnt, hierarchy = cv2.findContours(cv_image_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cv_image,cnt,-1,(0,0,255), 2)
        cv2.imshow("Image findContours", cv_image)
    
    def image_callback(self, msg):
        if self.original_image is not None:
            cv2.imshow("original_image", self.original_image)

        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.cv_image = copy.deepcopy(cv_image)

        # cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # self.canny(original_image, cv_image_gray)

        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

    def timer_callback(self, event):
        print('timer_callback')
        self.original_image = self.cv_image
        cv_image_gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        print('cv_image_gray')
        self.canny(self.cv_image, cv_image_gray)
        print('finishing timer_callback')
        # cv2.waitKey(3)

    def camera_info_callback(self, msg):
        # print(msg)
        pass

    def depth_points_callback(self, msg):
        pass

    def depth_image_callback(self, msg):
        pass
        # depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # depth_array = np.array(depth_image, dtype=np.float32)   
        # np.save("depth_img.npy", depth_array)
        # rospy.loginfo(depth_array)
        # #To save image as png
        # # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # cv2.imshow("Depth Image window", depth_image)
        # cv2.waitKey(3)

    # def callback(self, ros_raw_image, ros_depth_image, depth_points):
    #     self.image_callback(ros_raw_image)
    #     self.depth_image_callback(ros_depth_image)
    #     self.depth_points_callback(depth_points)
        

if __name__ == '__main__':
    image_node()