import cv2
import numpy as np
import copy 

def calculate_3D_points(self, image_points):
    """Calculate PnP"""
    object_points = np.array([[0,0,0], [0,1,0], [1,1,0], [1,0,0]]).reshape(
        (4, 3)).astype('float32')
    image_points = np.array(image_points).reshape((4, 2)).astype('float32')

    rtval, rvec, tvec = cv2.solvePnP(
        object_points, image_points, self.camera_matrix, self.dist_coeffs)
    R = cv2.Rodrigues(rvec)[0]

    return R, tvec

def points_sort(points):
    """Get upper right and lower left of list of points"""
    points = np.array(points).reshape((4, 2)).astype('float32')
    points = points[np.argsort(points[:, 0])]
    points_x = points[:2, :]
    points_y = points[2:, :]

    points_x = points_x[np.argsort(points_x[:, 1])]
    points_y = points_y[np.argsort(points_y[:, 1])]
    return [points_x[0], points_x[1], points_y[1], points_y[0]]

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1) 
    total = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1) + (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the area of intersection rectangle
    return intersection / float(total - intersection)

def canny2(cv_image, cv_image_gray):
    post_proc = copy.deepcopy(cv_image)
    redBajo1 = np.array([0, 0, 50])
    redAlto1 = np.array([50, 60, 255])

    fondoBajo1 = np.array([50, 30, 0])
    fondoAlto1 = np.array([255, 200, 255])
    
    frameHSV = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    cv2.imwrite('canny.png', frameHSV)
    maskRed1 = cv2.inRange(frameHSV, redBajo1, redAlto1)
    cv2.imwrite('canny.png', maskRed1)

    fondoRed1 = cv2.inRange(frameHSV, fondoBajo1, fondoAlto1)

    cv2.imwrite('canny.png', fondoRed1)
    #not
    #maskRed1 = cv2.bitwise_not(fondoRed1)
    #or
    kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(fondoRed1,kernel,iterations = 3)
    cv2.imwrite('canny.png', dilate)
    
    #change uint16 to uint8
    cv_blur = cv2.GaussianBlur(dilate, (7, 7), 9)
    mediana = cv2.medianBlur(cv_blur, 5)
    cv2.imwrite('canny.png', mediana)
    cv2.imshow('canny', dilate)
    # cv_blur = cv_image_gray
    # canny = cv2.Canny(cv_blur, 150, 180)
    canny = cv2.Canny(mediana, 5, 150)
    cv2.imwrite('canny.png', canny)
    dilate_canny = cv2.dilate(canny,kernel,iterations = 1)

    cnt, hierarchy = cv2.findContours(dilate_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(cv_image,cnt,-1,(0,0,255), 2)

    black_image = np.zeros((cv_image.shape[0], cv_image.shape[1], 3), np.uint8)

    cv2.imwrite('canny.png', cv_image)
    # cv2.drawContours(cv_image,cnt,-1,(0,0,255), 2)
    cv2.imwrite('canny.png', cv_image)

    currents_detections=[]
    resutls=[]
    for c in cnt:
        x, y, w, h = cv2.boundingRect(c)
        if w < 50:
            continue
        #cv2.drawContours(cv_image,c,-1,(255,0,0), 2)
        cv2.imwrite('canny.png', cv_image)
        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (255,255,0), 6)

        cv2.imwrite('canny.png', cv_image)
        if w > 30 and h > 30 and w < 400 and h < 400:
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0,0,255), 2)

            cv2.imwrite('canny.png', cv_image)
            alpha = 0
            if x- alpha > 0 and y - alpha > 0:
                cut_image_erode = fondoRed1[y - alpha:y + alpha + h, x - alpha:x + alpha + w]
            else:
                continue
            #porcentaje de blancos
            num_ones = np.sum(cut_image_erode == 255)
            #si no hay blanco en la imagen
            #angulo bbox
            angulo = cv2.minAreaRect(c)
            #print(angulo)
            #print(num_ones/(cut_image_erode.shape[0]*cut_image_erode.shape[1]))
            if num_ones/(cut_image_erode.shape[0]*cut_image_erode.shape[1]) < 0.23 or num_ones/(cut_image_erode.shape[0]*cut_image_erode.shape[1]) > 0.90:
                continue
            
            if cut_image_erode.shape[0] < 2 or cut_image_erode.shape[1] < 2:
                continue
            cut_image_erode = cv2.cvtColor(cut_image_erode, cv2.COLOR_GRAY2BGR)
            gaussian = cv2.GaussianBlur(cut_image_erode, (5, 5), 0)
            cv2.imwrite('canny.png', gaussian)
            canny_cut_image_erode = cv2.Canny(gaussian, 125, 150)
            cv2.imwrite('canny.png', canny_cut_image_erode)
            dilate_ = cv2.dilate(canny_cut_image_erode,kernel,iterations = 1)
            cv2.imwrite('canny.png', dilate_)
            cnt_erode,_= cv2.findContours(canny_cut_image_erode, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            #draw contours
            #cv2.drawContours(canny_cut_image_erode,cnt_erode,-1,(0,0,255), 2)
            cv2.imwrite('canny.png', canny_cut_image_erode)

            cont_cnts=0
            for c_erode in cnt_erode:
                area = cv2.contourArea(c_erode)
                x_, y_, w_, h_ = cv2.boundingRect(c_erode)
                # cv2.rectangle(dilate_, (x_, y_), (x_ + w_, y_ + h_), (255,0,0), 2)
                cv2.imwrite('canny.png', dilate_)
                if w_ < 7 or h_ < 7:
                    continue
                same = False
                for d in currents_detections:
                    iou = calculate_iou(d, [x_, y_, x_+w_, y_+h_])
                    if iou > 0.8:
                        same=True
                        break
                if same:
                    continue

                if x_ == 0 and y_ == 0:
                    continue
                currents_detections.append([x_,y_,x_+w_, y_+h_])
                cont_cnts+=1

            if cont_cnts == 1:
                if y < 30 and h < 100:
                    continue
                cv2.rectangle(black_image, (x, y), (x + w, y + h), (0,255,0), 2)
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0,255,0), 2)
                cv2.imwrite('canny.png', cv_image)
                alpha_cut=0
                cv_image_final_cut=post_proc[y-alpha_cut:y+alpha_cut+h, x-alpha_cut:x+alpha_cut+w]
                resutls.append([x+alpha, y+alpha, 0, cv_image_final_cut, [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]])
            # cv2.imshow('cv_image', cv_image) 

    return resutls

cv_image=cv2.imread('canny_prueba.png')
cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
resutls=canny2(cv_image, cv_image_gray)