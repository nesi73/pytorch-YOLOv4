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
from std_msgs.msg import Bool, Int32
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tf
from transforms3d import quaternions

class image_node():
    def __init__(self):
        rospy.init_node('image_subscriber', anonymous=True)

        self.model, self.class_names=get_model_init()
        self.run_flag = False
        self.camera_matrix, self.dist_coeffs, self.R, self.t = None, None, None, None
        self.translation, self.rotation = None, None
        self.drone_pose = {'position': None, 'orientation': None}
        self.detections_pose = []

        # tf
        self.br = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()

        #plot 3D points
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')

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
    
    def callback_flag_run(self, msg):
        self.run_flag = msg.data
    
    def pose_callback(self, msg):
        position = msg.pose.position
        orientation = msg.pose.orientation
        self.drone_pose['position'] = position
        self.drone_pose['orientation'] = orientation

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
            cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            results = self.detect_tiles(cv_image, cv_image_gray)
            time = msg.header.stamp

            for result in results:
                x, y, alpha, image_cut = result
                start_x, start_y = x-alpha, y-alpha
                total_detections = self.detect_cracks(image_cut)
                
                if len(total_detections) > 0:
                    quat = quaternions.mat2quat(self.R)
                    self.br.sendTransform((self.translation[0][0], self.translation[1][0], self.translation[2][0]),
                            quat,
                            time,
                            "object",
                            "red/camera")

                    try:
                        (trans,rot) = self.listener.lookupTransform('world', 'object', rospy.Time(0))
                        print(trans)
                        exist=False
                        alpha = 2 #Test ratio for distance between detections in meters
                        times_detected= 3 #Number of times that the same detection is detected
                        for detection in self.detections_pose:
                            if abs(detection['position'][0] - trans[0]) < alpha and abs(detection['position'][1] - trans[1]) < alpha and abs(detection['position'][2] - trans[2]) < alpha:
                                exist=True
                                detection['detected'] += 1

                                # Only if detected n times
                                if detection['detected'] == times_detected:
                                    print('DETECTED ', times_detected,' TIMES')
                                    self.run_flag = False
                                    for detection in total_detections:
                                        original_image = self.paint(detection, cv_image, start_x, start_y)

                                        cv2.imshow("CRACK", original_image)

                                        self.detections_cracks.publish(self.bridge.cv2_to_imgmsg(original_image, "bgr8"))
                                        self.original_image_detections_cracks.publish(self.bridge.cv2_to_imgmsg(original_image, "bgr8"))
                                break
                        
                        if not exist:
                            self.detections_pose.append({'position':trans, 'detected': 1})
                            print('new detection', self.detections_pose)

                    except Exception as e:
                        print(e)
                        continue


            cv2.imshow("Image window", cv_image)
            cv2.waitKey(3)

    def camera_info_callback(self, msg):
        if self.run_flag:
            self.camera_matrix = np.array(msg.K).reshape((3, 3))
            self.dist_coeffs = np.array(msg.D)
            self.R = np.array(msg.R).reshape((3, 3))
            P = np.array(msg.P).reshape((3, 4))
            self.t = P[:, 3]

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

    def detect_tiles(self, cv_image, cv_image_gray, alpha=10):
        original_image = copy.deepcopy(cv_image)
        cv_blur = cv2.GaussianBlur(cv_image_gray, (5, 5), 0)
        canny = cv2.Canny(cv_blur, 150, 180)
        kernel = np.ones((5,5),np.uint8)
        dilation = cv2.dilate(canny,kernel,iterations = 1)

        cnt, hierarchy = cv2.findContours(dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cv_image,cnt,-1,(0,0,255), 2)

        currents_detections=[]
        results=[]
        for c in cnt:
            approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(c)

                ratio = float(w)/h
                
                if ratio >= 0.9 and ratio <= 1.1:
                    x, y, w, h = cv2.boundingRect(c)

                    #TODO: and w < 95 and h < 95
                    if (w > 65 and h > 65):
                        same=False
                        for c in currents_detections:
                            iou = self.calculate_iou(c, [x, y, x + w, y + h])
                            if iou > 0.5:
                                same=True
                                break

                        if not same:

                            if x-alpha < 0 or y-alpha < 0 or x+w+alpha > cv_image.shape[1] or y+h+alpha > cv_image.shape[0]:
                                continue

                            currents_detections.append([x, y, x + w, y + h])
                            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (36,255,12), 2)
                            # cv2.circle(cv_image, (x, y), 5, (0, 0, 255), -1)
                            # cv2.circle(cv_image, (x + w, y), 5, (0, 0, 255), -1)
                            # cv2.circle(cv_image, (x + w, y + h), 5, (0, 0, 255), -1)
                            # cv2.circle(cv_image, (x, y + h), 5, (0, 0, 255), -1)
                            pt_cnt = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)
                            cv2.circle(cv_image, (pt_cnt[0][0][0], pt_cnt[0][0][1]), 5, (0, 0, 255), -1)
                            cv2.circle(cv_image, (pt_cnt[1][0][0], pt_cnt[1][0][1]), 5, (0, 255,0), -1)
                            cv2.circle(cv_image, (pt_cnt[2][0][0], pt_cnt[2][0][1]), 5, (255, 0, 255), -1)
                            cv2.circle(cv_image, (pt_cnt[3][0][0], pt_cnt[3][0][1]), 5, (255, 0, 0), -1)

                            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
                            # cv2.imshow("Image window1", cv_image)
                            # cv2.waitKey(3)
                            pol = Polygon()
                            pol.points.append(Point32(x, y, 0))
                            pol.points.append(Point32(x + w, y, 0))
                            pol.points.append(Point32(x + w, y + h, 0))
                            pol.points.append(Point32(x, y + h, 0))
                            self.bbox_detections.publish(pol)

                            #offset alpha
                            image_cut = original_image[y-alpha:y+alpha+h,x-alpha:x+w+alpha]
                            results.append([x, y, alpha, image_cut])

                            x1, y1, x2, y2 = x-alpha, y-alpha, x+w+alpha, y+alpha+h
                            # bbox title with boundingRect
                            self.translation, self.rotation = self.calculate_3D_points([[x1, y1],[x1, y2],[x2, y2],[x2, y1]])

                            # countour points approxPolyDP
                            points_sort = self.points_sort(pt_cnt)
                            self.translation, self.rotation = self.calculate_3D_points([points_sort])

                            # self.plot_3D_points(world_points)
                            # # plt.show()

                            # #save plt
                            # plt.savefig('test.png')
                            # a=0


        return results
    
    def plot_3D_points(self, points):
        self.ax.scatter(points[:,0], points[:,1], points[:,2], c='r', marker='o')
        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')
      
    def calculate_3D_points(self, imagePoints):
        objectPoints = [[0,0,0], [0,1,0], [1,1,0], [1,0,0]]
        objectPoints = np.array(objectPoints).reshape((4,3)).astype('float32')
        imagePoints = np.array(imagePoints).reshape((4,2)).astype('float32')

        rtval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, self.camera_matrix, self.dist_coeffs)
        R = cv2.Rodrigues(rvec)[0]

        transfP2C = np.concatenate((R, tvec), axis=1)
        transfP2C = np.concatenate((transfP2C, np.array([[0,0,0,1]])), axis=0)

        camera_points = transfP2C @ objectPoints
        
        # print('self.R', self.R)
        # print('self.t', self.t)
        # transfC2W = np.concatenate((self.R, self.t.reshape((3,1))), axis=1)
        # transfC2W = np.concatenate((transfC2W, np.array([[0,0,0,1]])), axis=0)

        # world_points = transfC2W @ camera_points

        return R, tvec

    def detect_cracks(self, image_cut):

        image_detect, total_detections = predict(self.model, self.class_names, image_cut, conf_thres=0.7)

        return total_detections

if __name__ == '__main__':
    image_node()
    rospy.spin()
