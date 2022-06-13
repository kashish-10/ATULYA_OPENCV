import numpy as np
import cv2
import cv2.aruco as aruco
import imutils
import math
aruco1=(cv2.imread("Ha.jpg"))
aruco2=(cv2.imread("HaHa.jpg"))
aruco3= (cv2.imread("LMAO.jpg"))
aruco4 = (cv2.imread("XD.jpg"))

# aruco = ['Ha.jpg','HaHa.jpg','LMAO.jpg','XD.jpg']
# arucodict = {}


def findAruco(img):                #function for finding corners and ids of arucos
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_5X5_250')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    (corners,ids,rejected) = aruco.detectMarkers(gray,arucoDict,parameters = arucoParam)
    return (corners,ids,rejected)

def aruco_Coords(img):      #function for finding the four co-ordinates of arucos
    (c,i,r) = findAruco(img)
    if len(c)>0:
        i=i.flatten()
        for (markercorner,markerid) in zip(c,i):
            corner = markercorner.reshape((4,2))
            (topleft,topright,bottomright,bottomleft) = corner
            bottomleft = (int(bottomleft[0]),int(bottomleft[1]))
            bottomright = (int(bottomright[0]),int(bottomright[1]))
            m = ((bottomright[1]-bottomleft[1])/(bottomright[0]-bottomleft[0]))
            fi = math.atan(m)
            a = fi * 180/math.pi
            # print(np.shape(img))
            img = imutils.rotate_bound(img,-a) #rotating the arucos
            (c, i, r) = findAruco(img)
            if len(c) > 0:
                i = i.flatten()
                for (markercorner, markerid) in zip(c, i):
                    corner = markercorner.reshape((4, 2))
                    (topleft, topright, bottomright, bottomleft) = corner


                    topleft = (int(topleft[0]), int(topleft[1]))
                    topright = (int(topright[0]), int(topright[1]))
                    bottomright = (int(bottomright[0]), int(bottomright[1]))
                    bottomleft = (int(bottomleft[0]), int(bottomleft[1]))
                    img = img[topleft[1]:bottomright[1],topleft[0]:bottomright[0]]   #cropping the arucos extra white space

    return img

# def findColor(value):
#     if




aruco_Coords(aruco1)
cv2.waitKey(0)
aruco_Coords(aruco2)
cv2.waitKey(0)
aruco_Coords(aruco3)
cv2.waitKey(0)
aruco_Coords(aruco4)
cv2.waitKey(0)


img = cv2.imread("CVtask.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret , thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
cont,heir = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for cnt in cont:
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    x, y, w, h = cv2.boundingRect(approx)

    if len(approx) == 4:
        if float(w)/h >= 0.95 and float(w)/h <= 1.05:
            cv2.drawContours(img, [approx], 0, (0, 0, 0), -1)
            slope = (approx[1][0][1]-approx[2][0][1])/(approx[1][0][0]-approx[2][0][0])
            slope = 180/math.pi*math.atan(slope)
            dx = math.sqrt((approx[1][0][1]-approx[2][0][1])**2 + (approx[1][0][0]-approx[2][0][0])**2)
            dx = int(dx)
            dy = math.sqrt((approx[2][0][1] - approx[3][0][1]) ** 2 + (approx[2][0][0] - approx[3][0][0]) ** 2)
            dy = int(dy)
            #print(dx,dy)
            blank = np.zeros(np.shape(img[y:y+h , x:x+w]) , np.uint8)

            aru = cv2.resize(aruco_Coords(aruco3),(dx,dy))
            blank[int((h-dy)/2):int((h+dy)/2),int((w-dx)/2):int((w+dx)/2)] = blank[int((h-dy)/2):int((h+dy)/2),int((w-dx)/2):int((w+dx)/2)] + aru
            blank = imutils.rotate(blank,-slope)
            cv2.imshow("image", blank)
            cv2.waitKey(0)
            img[y:y+h,x:x+w] = img[y:y+h,x:x+w] + blank

            # for i in aruco:
            #     x = cv2.imread(i)
            #     (c,i,r) = findAruco(x)
            #     arucodict[i] = ids
            # print(arucodict)


            cv2.namedWindow("Detected_Squares",cv2.WINDOW_NORMAL)
            cv2.imshow("Detected_Squares",img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()