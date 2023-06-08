"""
image_detections_inspection_tf.py
"""

import copy
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from geometry_msgs.msg import Polygon, Point32
# from models import predict, get_model_init
from std_msgs.msg import Bool
import tf
from transforms3d import quaternions
from config_ICUAS import cfg
from models2 import predict, get_model_init


class CrackDetector():
    """Crack Detector Node"""

    # Subscribers
    RUN_NODE_TOPIC = "/crack_detector/run"
    COLOR_CAM_IMAGE_TOPIC = "/red/camera/color/image_raw/compressed"
    COLOR_CAM_INFO_TOPIC = "/red/camera/color/camera_info"
    # Publishers
    TILE_DETECTION_TOPIC = "/red/crack_detector/tiles/image"
    CRACK_DETECTION_TOPIC = "/red/crack_detector/crack/image"
    CRACK_IN_ORIGINAL_IMAGE_TOPIC = "/red/crack_detector/crack/original_image"
    TILE_BBOX_DETECTION_TOPIC = "/red/crack_detector/tiles/bbox"

    def __init__(self):
        rospy.init_node('crack_detector', anonymous=True)

        self.model, self.class_names = get_model_init()
        self.run_flag = True  # TEMPORAL
        self.camera_matrix, self.dist_coeffs = None, None
        self.drone_pose = {'position': None, 'orientation': None}
        self.detections_pose = []
        self.original_image = None

        # tf
        self.br = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()

        self.flag_run_sub = rospy.Subscriber(
            self.RUN_NODE_TOPIC, Bool, self.callback_flag_run)
        self.image_sub = rospy.Subscriber(
            self.COLOR_CAM_IMAGE_TOPIC, Image, self.image_callback)
        self.camera_info_sub = rospy.Subscriber(
            self.COLOR_CAM_INFO_TOPIC, CameraInfo, self.camera_info_callback)

        # public new topic
        self.image_pub = rospy.Publisher(
            self.TILE_DETECTION_TOPIC, Image, queue_size=10)
        self.detections_cracks = rospy.Publisher(
            self.CRACK_DETECTION_TOPIC, Image, queue_size=10)
        self.original_image_detections_cracks = rospy.Publisher(
            self.CRACK_IN_ORIGINAL_IMAGE_TOPIC, Image, queue_size=10)

        # publisher an array ros message
        self.bbox_detections = rospy.Publisher(
            self.TILE_BBOX_DETECTION_TOPIC, Polygon, queue_size=10)

        self.bridge = CvBridge()
        print('INIT')

    def callback_flag_run(self, msg):
        """Flag to enable crack detections or not"""
        print("FLAG RUN:", msg.data)
        self.run_flag = msg.data

    def pose_callback(self, msg):
        """Position and orientation update"""
        self.drone_pose['position'] = msg.pose.position
        self.drone_pose['orientation'] = msg.pose.orientation

    def points_sort(self, points):
        """Get upper right and lower left of list of points"""
        points = np.array(points).reshape((4, 2)).astype('float32')
        points = points[np.argsort(points[:, 0])]
        points_x = points[:2, :]
        points_y = points[2:, :]

        points_x = points_x[np.argsort(points_x[:, 1])]
        points_y = points_y[np.argsort(points_y[:, 1])]
        return [points_x[0], points_x[1], points_y[1], points_y[0]]

    def image_callback(self, msg):
        """Front camera color image callback"""

        print('holaa')
        if not self.run_flag:
            return

        # cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.original_image = copy.deepcopy(cv_image)
        cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        tiles_detected = self.detect_tiles(cv_image, cv_image_gray)
        time = msg.header.stamp
        frame = msg.header.frame_id
        # frame = "red/camera"

        for tile in tiles_detected:
            x, y, alpha, image_cut, translation, rotation = tile
            start_x, start_y = x-alpha, y-alpha

            # Detect cracks with the image cut (tile)
            crack_detections = self.detect_cracks(image_cut)

            # Skip to next tile if no cracks detected
            if len(crack_detections) <= 0:
                continue

            # self.R, self.translation calculated in calculate_3D_points with solvePnP

            quat = quaternions.mat2quat(rotation)
            self.br.sendTransform((translation[0][0], translation[1][0], translation[2][0]),
                                  quat,
                                  time,
                                  "object",
                                  frame)

            try:
                (trans, rot) = self.listener.lookupTransform(
                    'world', 'object', rospy.Time(0))

                # Check if already detected
                exist = False
                for detection in self.detections_pose:
                    # Is detection is close enough to previous detection?
                    if abs(detection['position'][0] - trans[0]) < cfg.radius and \
                            abs(detection['position'][1] - trans[1]) < cfg.radius and \
                            abs(detection['position'][2] - trans[2]) < cfg.radius:
                        exist = True
                        detection['detected'] += 1  # increase counter

                        # Only if detected n times
                        if detection['detected'] == cfg.times_detected:
                            print('DETECTED ', cfg.times_detected, ' TIMES')

                            self.run_flag = False
                            for crack in crack_detections:
                                original_image = self.paint(
                                    crack, cv_image, start_x, start_y)
                                self.detections_cracks.publish(
                                    self.bridge.cv2_to_imgmsg(original_image, "bgr8"))
                                self.original_image_detections_cracks.publish(
                                    self.bridge.cv2_to_imgmsg(original_image, "bgr8"))

                                if cfg.debug:
                                    cv2.imshow("CRACK", original_image)
                        break

                # The position of the new detection is not inside the
                # radious of any previous detection
                if not exist:
                    self.detections_pose.append(
                        {'position': trans, 'detected': 1})

            # GOD HELP ME
            except Exception as e:
                print(e)
                continue

        if cfg.debug:
            cv2.imshow("Image window", cv_image)
            cv2.waitKey(3)

    def camera_info_callback(self, msg):
        """Front color camera info callback"""
        if self.run_flag:
            self.camera_matrix = np.array(msg.K).reshape((3, 3))
            self.dist_coeffs = np.array(msg.D)

    def calculate_iou(self, boxA, boxB):
        """Calculate intersection over union"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        total = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1) + \
            (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the area of intersection rectangle
        return intersection / float(total - intersection)

    def paint(self, detection, img, start_x, start_y):
        """Paint detection on an image"""
        msg = detection['msg']
        x1, y1, x2, y2 = detection['bbox']
        x1, y1, x2, y2 = x1 + start_x, y1 + start_y, x2 + start_x, y2 + start_y
        width = img.shape[1]
        height = img.shape[0]
        bbox_thick = int(0.6 * (height + width) / 600)
        rgb = (255, 0, 0)

        t_size = cv2.getTextSize(msg, 0, 0.7, thickness=bbox_thick // 2)[0]
        c1, c2 = (x1, y1), (x2, y2)
        c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)

        cv2.rectangle(img, (x1, y1), (np.int32(
            c3[0]), np.int32(c3[1])), rgb, -1)
        img = cv2.putText(img, msg, (c1[0], np.int32(
            c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, bbox_thick)

        return img

    def detect_tiles3(self, cv_image, cv_image_gray):

        original_image = copy.deepcopy(cv_image)
        _, img = cv2.threshold(cv_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, cv_image_gray)

        redBajo1 = np.array([0, 42, 0])
        redAlto1 = np.array([179, 255, 255])

        frameHSV = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        maskRed1 = cv2.inRange(frameHSV, redBajo1, redAlto1)
        cv_blur = cv2.GaussianBlur(maskRed1, (3, 3), 0)

        canny = cv2.Canny(cv_blur, 5, 150)

        kernel = np.ones((5,5),np.uint8)

        dilation = cv2.dilate(canny,kernel,iterations = 1)

        cnt, hierarchy = cv2.findContours(dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(cv_image,cnt,-1,(0,0,255), 2)

        currents_detections=[]
        resutls=[]

        for c in cnt:

            x, y, w, h = cv2.boundingRect(c)

            if x > 200 and w > 70 and h > 70:
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (36,255,12), 2)
                cv2.imwrite('canny.png', cv_image)

                cv2.imshow('canny_tile', cv_image)

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
                    print("Tile detected")
                    pol = Polygon()
                    pol.points.append(Point32(x, y, 0))
                    pol.points.append(Point32(x + w, y, 0))
                    pol.points.append(Point32(x + w, y + h, 0))
                    pol.points.append(Point32(x, y + h, 0))
                    #self.bbox_detections.publish(pol)
                    cv_image=original_image[y-0:y+0+h,x-0:x+w+0]

                    x1, y1, x2, y2 = x-cfg.margin_tile, y-cfg.margin_tile, x + \
                            w+cfg.margin_tile, y+cfg.margin_tile+h
                    points_sort = self.points_sort([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
                    rotation, translation = self.calculate_3D_points(
                        [points_sort])

                    resutls.append([x, y, 0, cv_image, translation, rotation])

        return resutls

    def detect_tiles2(self, cv_image, cv_image_gray):
        """Detect tile in image

        Return:
        - List with [(top left corner of tile, tile margin, image cut)]
        """
        cv_blur = cv2.GaussianBlur(cv_image_gray, (5, 5), 0)
        canny = cv2.Canny(cv_blur, 50, 80)
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(canny, kernel, iterations=1)

        contours, __hierarchy = cv2.findContours(
            dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        currents_detections, results = [], []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)

            # Only 4 edges polygons
            if len(approx) != 4:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h

            # Ratio to squariness of the polygon
            if ratio >= cfg.ratio_tile[0] and ratio <= cfg.ratio_tile[1]:
                x, y, w, h = cv2.boundingRect(cnt)

                # Check size of the tile in pixels
                if w > cfg.size_tile[0] and h > cfg.size_tile[0] and w < cfg.size_tile[1] and h < cfg.size_tile[1]:

                    same = False
                    for cnt in currents_detections:
                        iou = self.calculate_iou(cnt, [x, y, x + w, y + h])
                        if iou > cfg.iou_tiles:
                            same = True
                            break

                    # Skip if tiles are the same
                    if same:
                        continue

                    # If tile with the margin is out of the image, continue
                    if x-cfg.margin_tile < 0 or y-cfg.margin_tile < 0 or x+w+cfg.margin_tile > cv_image.shape[1] or y+h+cfg.margin_tile > cv_image.shape[0]:
                        continue

                    currents_detections.append([x, y, x + w, y + h])

                    pol = Polygon()
                    pol.points.append(Point32(x, y, 0))
                    pol.points.append(Point32(x + w, y, 0))
                    pol.points.append(Point32(x + w, y + h, 0))
                    pol.points.append(Point32(x, y + h, 0))

                    # Publish bbox detections and image
                    self.bbox_detections.publish(pol)
                    self.image_pub.publish(
                        self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

                    # CALCULATE 3D POINTS OF THE TILE WITH SOLVEPNP.
                    # Two methods:
                    # 1) Using bbox title
                    # 2) Using specific corners of the tile bbox title with boundingRect

                    #######
                    # DEPRECATED
                    #######
                    # x1, y1, x2, y2 = x-cfg.margin_tile, y-cfg.margin_tile, x + \
                    #     w+cfg.margin_tile, y+cfg.margin_tile+h
                    # rotation, translation = self.calculate_3D_points(
                    #     [[x1, y1], [x1, y2], [x2, y2], [x2, y1]])

                    # countour points approxPolyDP
                    contour_points = cv2.approxPolyDP(
                        cnt, 0.01*cv2.arcLength(cnt, True), True)
                    points_sort = self.points_sort(contour_points)
                    rotation, translation = self.calculate_3D_points(
                        [points_sort])
                    # -----------------------------------------------------

                    # Return the top left corner of the tile, the margin and
                    # the image cut (the first are for draw the tile in the original image)
                    image_cut = self.original_image[y-cfg.margin_tile:y +
                                                    cfg.margin_tile+h, x-cfg.margin_tile:x+w+cfg.margin_tile]
                    results.append(
                        [x, y, cfg.margin_tile, image_cut, translation, rotation])

                    if cfg.debug:
                        cv2.rectangle(cv_image, (x, y),
                                      (x + w, y + h), (36, 255, 12), 2)
                        cv2.circle(
                            cv_image, (contour_points[0][0][0], contour_points[0][0][1]), 5, (0, 0, 255), -1)
                        cv2.circle(
                            cv_image, (contour_points[1][0][0], contour_points[1][0][1]), 5, (0, 255, 0), -1)
                        cv2.circle(
                            cv_image, (contour_points[2][0][0], contour_points[2][0][1]), 5, (255, 0, 255), -1)
                        cv2.circle(
                            cv_image, (contour_points[3][0][0], contour_points[3][0][1]), 5, (255, 0, 0), -1)

        return results

    def detect_tiles(self, cv_image, cv_image_gray):

        redBajo1 = np.array([0, 0, 50])
        redAlto1 = np.array([50, 60, 255])

        fondoBajo1 = np.array([50, 30, 0])
        fondoAlto1 = np.array([255, 100, 255])
        
        frameHSV = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        cv2.imwrite('canny.png', frameHSV)
        maskRed1 = cv2.inRange(frameHSV, redBajo1, redAlto1)
        cv2.imwrite('canny.png', maskRed1)

        fondoRed1 = cv2.inRange(frameHSV, fondoBajo1, fondoAlto1)

        kernel = np.ones((3,3),np.uint8)
        dilate = cv2.dilate(fondoRed1,kernel,iterations = 1)
        
        #change uint16 to uint8
        cv_blur = cv2.GaussianBlur(dilate, (7, 7), 9)
        mediana = cv2.medianBlur(cv_blur, 5)
        
        canny = cv2.Canny(mediana, 5, 150)
        dilate_canny = cv2.dilate(canny,kernel,iterations = 1)

        cnt, hierarchy = cv2.findContours(dilate_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        currents_detections=[]
        resutls=[]
        for c in cnt:
            x, y, w, h = cv2.boundingRect(c)
            if w < 50:
                continue
            if w > 50 and h > 50 and w < 260 and h < 260:

                cv2.imwrite('canny.png', cv_image)
                alpha = 15
                if x- alpha > 0 and y - alpha > 0 and x + alpha + w < cv_image.shape[1] and y + alpha + h < cv_image.shape[0]:
                    cut_image_erode = dilate[y - alpha:y + alpha + h, x - alpha:x + alpha + w]
                else:
                    continue
                #porcentaje de blancos
                num_ones = np.sum(cut_image_erode == 255)

                if num_ones/(cut_image_erode.shape[0]*cut_image_erode.shape[1]) < 0.23 or num_ones/(cut_image_erode.shape[0]*cut_image_erode.shape[1]) > 0.90:
                    continue
                
                if cut_image_erode.shape[0] < 2 or cut_image_erode.shape[1] < 2:
                    continue
                cut_image_erode = cv2.cvtColor(cut_image_erode, cv2.COLOR_GRAY2BGR)
                gaussian = cv2.GaussianBlur(cut_image_erode, (5, 5), 0)
                canny_cut_image_erode = cv2.Canny(gaussian, 125, 150)
                dilate_ = cv2.dilate(canny_cut_image_erode,kernel,iterations = 1)
                cnt_erode,_= cv2.findContours(canny_cut_image_erode, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                cont_cnts=0
                currents_detections=[]
                for c_erode in cnt_erode:
                    x_, y_, w_, h_ = cv2.boundingRect(c_erode)
                    if w_ < 7 or h_ < 7:
                        continue
                    same = False
                    for d in currents_detections:
                        iou = self.calculate_iou(d, [x_, y_, x_+w_, y_+h_])
                        if iou > 0.8:
                            same=True
                            break
                    if same:
                        continue
                    currents_detections.append([x_,y_,x_+w_, y_+h_])
                    cont_cnts+=1

                if cont_cnts == 1:
                    if x_ < 5 or y_ < 5 or x_ + w_ > cut_image_erode.shape[1] - 5 or y_ + h_ > cut_image_erode.shape[0] - 5:
                        continue 
                    cv_image_final_cut=cv_image[y:y+h, x:x+w]
                    cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0,255,0), 2)
                    
                    points_sorted=self.points_sort([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
                    rotation, translation = self.calculate_3D_points(
                        points_sorted)
                    resutls.append([x, y, 0, cv_image_final_cut, translation, rotation]) 

        return resutls
    
    def calculate_3D_points(self, image_points):
        """Calculate PnP"""
        object_points = np.array(cfg.object_points).reshape(
            (4, 3)).astype('float32')
        image_points = np.array(image_points).reshape((4, 2)).astype('float32')

        rtval, rvec, tvec = cv2.solvePnP(
            object_points, image_points, self.camera_matrix, self.dist_coeffs)
        R = cv2.Rodrigues(rvec)[0]

        return R, tvec

    def detect_cracks(self, image_cut):
        """Detect cracks"""
        image_detect, total_detections = predict(
            self.model, self.class_names, image_cut)

        return total_detections


if __name__ == '__main__':
    CrackDetector()
    rospy.spin()
