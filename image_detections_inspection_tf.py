import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import copy
import numpy as np
from geometry_msgs.msg import Polygon, Point32
#from models import predict, get_model_init
from sensor_msgs.msg import PointCloud2
import message_filters
from std_msgs.msg import Bool, Int32
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tf
from transforms3d import quaternions
from config_ICUAS import cfg
from models2 import predict, get_model_init

class image_node():
    def __init__(self):
        rospy.init_node('image_subscriber', anonymous=True)

        self.model, self.class_names=get_model_init()
        self.run_flag = False
        self.camera_matrix, self.dist_coeffs = None, None
        self.translation, self.rotation = None, None
        self.drone_pose = {'position': None, 'orientation': None}
        self.detections_pose = []
        self.original_image = None

        # tf
        self.br = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()

        self.flag_run_sub = rospy.Subscriber("/flag/run", Bool, self.callback_flag_run)
        self.image_sub = rospy.Subscriber("/red/camera/color/image_raw", Image, self.image_callback)
        self.camera_info_sub = rospy.Subscriber("/red/camera/color/camera_info", CameraInfo, self.camera_info_callback)
        
        # public new topic
        self.image_pub = rospy.Publisher("/image/detections", Image, queue_size=10)
        self.detections_cracks = rospy.Publisher("/image/detections_cracks", Image, queue_size=10)
        self.original_image_detections_cracks = rospy.Publisher("/image/original_image_detections_cracks", Image, queue_size=10)

        #publisher an array ros message
        self.bbox_detections = rospy.Publisher("/bbox/detections", Polygon , queue_size=10)

        self.bridge = CvBridge()
        print('INIT')
    
    def callback_flag_run(self, msg):
        print(msg.data)
        self.run_flag = msg.data
    
    def pose_callback(self, msg):
        if self.run_flag:
            self.drone_pose['position'] = msg.pose.position
            self.drone_pose['orientation'] = msg.pose.orientation

    def points_sort(self, points):
        points = np.array(points).reshape((4,2)).astype('float32')
        points = points[np.argsort(points[:, 0])]
        points_x = points[:2, :]
        points_y = points[2:, :]

        points_x = points_x[np.argsort(points_x[:, 1])]
        points_y = points_y[np.argsort(points_y[:, 1])]
        return [points_x[0], points_x[1], points_y[1], points_y[0]]

    def image_callback(self, msg):
        if self.run_flag:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.original_image = copy.deepcopy(cv_image)
            cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            results = self.detect_tiles(cv_image, cv_image_gray)
            time = msg.header.stamp

            for result in results:
                x, y, alpha, image_cut = result
                start_x, start_y = x-alpha, y-alpha

                # Detect cracks with the image cut (tile)
                total_detections = self.detect_cracks(image_cut)
                
                # If some crack is detected
                if len(total_detections) > 0:

                    # self.R, self.translation calculated in calculate_3D_points with solvePnP
                    quat = quaternions.mat2quat(self.rotation)
                    self.br.sendTransform((self.translation[0][0], self.translation[1][0], self.translation[2][0]),
                            quat,
                            time,
                            "object",
                            "red/camera")

                    try:
                        (trans,rot) = self.listener.lookupTransform('world', 'object', rospy.Time(0))
                        
                        exist=False
                        for detection in self.detections_pose:
                            if abs(detection['position'][0] - trans[0]) < cfg.radius and abs(detection['position'][1] - trans[1]) < cfg.radius and abs(detection['position'][2] - trans[2]) < cfg.radius:
                                exist=True
                                detection['detected'] += 1

                                # Only if detected n times
                                if detection['detected'] == cfg.times_detected:
                                    print('DETECTED ', cfg.times_detected,' TIMES')
                                    
                                    self.run_flag = False
                                    for detection in total_detections:
                                        original_image = self.paint(detection, cv_image, start_x, start_y)
                                        self.detections_cracks.publish(self.bridge.cv2_to_imgmsg(original_image, "bgr8"))
                                        self.original_image_detections_cracks.publish(self.bridge.cv2_to_imgmsg(original_image, "bgr8"))
                                        
                                        if cfg.debug:
                                            cv2.imshow("CRACK", original_image)
                                break
                        
                        # The position of the new detection is not inside the radius of any previous detection
                        if not exist:
                            self.detections_pose.append({'position':trans, 'detected': 1})

                    except Exception as e:
                        print(e)
                        continue

            if cfg.debug:
                cv2.imshow("Image window", cv_image)
                cv2.waitKey(3)

    def camera_info_callback(self, msg):
        if self.run_flag:
            self.camera_matrix = np.array(msg.K).reshape((3, 3))
            self.dist_coeffs = np.array(msg.D)

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

    def detect_tiles(self, cv_image, cv_image_gray):
        cv_blur = cv2.GaussianBlur(cv_image_gray, (5, 5), 0)
        canny = cv2.Canny(cv_blur, 50, 80)
        kernel = np.ones((5,5),np.uint8)
        dilation = cv2.dilate(canny,kernel,iterations = 1)

        cnt, hierarchy = cv2.findContours(dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        currents_detections, results = [], []
        for c in cnt:
            approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(c)

                ratio = float(w)/h
                
                if ratio >= cfg.ratio_tile[0] and ratio <= cfg.ratio_tile[1]:
                    x, y, w, h = cv2.boundingRect(c)

                    if w > cfg.size_tile[0] and h > cfg.size_tile[0] and w < cfg.size_tile[1] and h < cfg.size_tile[1]:
                        
                        same=False
                        for c in currents_detections:
                            iou = self.calculate_iou(c, [x, y, x + w, y + h])
                            if iou > cfg.iou_tiles:
                                same=True
                                break

                        if not same:
                            
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
                            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
                            
                            # CALCULATE 3D POINTS OF THE TILE WITH SOLVEPNP. Two ways: First with the bbox title and second with the specific corners of the tile
                            # bbox title with boundingRect
                            x1, y1, x2, y2 = x-cfg.margin_tile, y-cfg.margin_tile, x+w+cfg.margin_tile, y+cfg.margin_tile+h
                            self.rotation, self.translation = self.calculate_3D_points([[x1, y1],[x1, y2],[x2, y2],[x2, y1]])

                            # countour points approxPolyDP
                            pt_cnt = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)
                            points_sort = self.points_sort(pt_cnt)
                            self.rotation, self.translation = self.calculate_3D_points([points_sort])
                            # -----------------------------------------------------

                            # Return the top left corner of the tile, the margin and the image cut (the first are for draw the tile in the original image)
                            image_cut = self.original_image[y-cfg.margin_tile:y+cfg.margin_tile+h,x-cfg.margin_tile:x+w+cfg.margin_tile]
                            results.append([x, y, cfg.margin_tile, image_cut])

                            if cfg.debug:
                                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (36,255,12), 2)
                                cv2.circle(cv_image, (pt_cnt[0][0][0], pt_cnt[0][0][1]), 5, (0, 0, 255), -1)
                                cv2.circle(cv_image, (pt_cnt[1][0][0], pt_cnt[1][0][1]), 5, (0, 255,0), -1)
                                cv2.circle(cv_image, (pt_cnt[2][0][0], pt_cnt[2][0][1]), 5, (255, 0, 255), -1)
                                cv2.circle(cv_image, (pt_cnt[3][0][0], pt_cnt[3][0][1]), 5, (255, 0, 0), -1)



        return results
      
    def calculate_3D_points(self, image_points):
        object_points = np.array(cfg.object_points).reshape((4,3)).astype('float32')
        image_points = np.array(image_points).reshape((4,2)).astype('float32')

        rtval, rvec, tvec = cv2.solvePnP(object_points, image_points, self.camera_matrix, self.dist_coeffs)
        R = cv2.Rodrigues(rvec)[0]

        return R, tvec

    def detect_cracks(self, image_cut):

        image_detect, total_detections = predict(self.model, self.class_names, image_cut)

        return total_detections

if __name__ == '__main__':
    image_node()
    rospy.spin()
