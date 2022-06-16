import numpy as np
import cv2
import cv2.aruco as aruco
import imutils
import math
aruco1 = (cv2.imread("Ha.jpg"))
aruco2 = (cv2.imread("HaHa.jpg"))
aruco3 = (cv2.imread("LMAO.jpg"))
aruco4 = (cv2.imread("XD.jpg"))

aruco_list = (aruco1,aruco2,aruco3,aruco4)  #list to store aruco images


def findAruco(img):                #function for finding corners and ids of arucos
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_5X5_250')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    (corners,ids,rejected) = aruco.detectMarkers(gray,arucoDict,parameters = arucoParam)
    return (corners,ids,rejected)

id_list = []     #list to store store respective aruco ids
for i in aruco_list:
    id_list.append((findAruco(i))[1][0][0])
# print(id_list)
def color(color,ll,ul):     #function to detect color of squares
    if color[0] in range(ll[0],ul[0]+1):
        if color[1] in range(ll[1],ul[1]+1):
            if color[2] in range(ll[2], ul[2] + 1):
                return True
    else:
        return False

def aruco_Coords(img,aru_length,aru_height,bound_height,bound_length,angle):
    (c,i,r) = findAruco(img)
    if len(c)>0:
        i=i.flatten()
        for (markercorner,markerid) in zip(c,i):
            corner = markercorner.reshape((4,2))
            (topleft,topright,bottomright,bottomleft) = corner
            bottomleft = (int(bottomleft[0]),int(bottomleft[1]))
            bottomright = (int(bottomright[0]),int(bottomright[1]))
            m = ((bottomright[1]-bottomleft[1])/(bottomright[0]-bottomleft[0]))           #finding slope
            fi = math.atan(m)
            a = fi * 180/math.pi                             #finding inclined angle of arucos
            # print(np.shape(img))
            img = imutils.rotate_bound(img,-a)               #rotating the arucos
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
            blank=np.zeros((int(bound_height),int(bound_length),3))                  #creating a blank
            s=np.shape(blank[int((int(bound_height)-int(aru_height))/2):int((int(bound_length)+int(aru_length))/2),int((int(bound_length)-int(aru_length))/2):int((int(bound_length)+int(aru_length))/2)])
            #print(s)
            img=cv2.resize(img,(s[1],s[0]))       #resizing the image
            blank[int((int(bound_height)-int(aru_height))/2):int((int(bound_length)+int(aru_length))/2),int((int(bound_length)-int(aru_length))/2):int((int(bound_length)+int(aru_length))/2)]=img
            img=imutils.rotate(blank,angle)       #rotating the blank
    return img

img = cv2.imread("CVtask.jpg")
def final_func(img,id_list):    #function for detecting squares,its contours and imposing the aruco on the respective squares
        img = cv2.imread("CVtask.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret , thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        cont,heir = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for cnt in cont:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)

            if len(approx) == 4:
                if float(w)/h >= 0.95 and float(w)/h <= 1.05:

                    slope = (approx[1][0][1]-approx[2][0][1])/(approx[1][0][0]-approx[2][0][0])
                    angle = 180/math.pi*math.atan(slope)
                    dx = math.sqrt((approx[1][0][1]-approx[2][0][1])**2 + (approx[1][0][0]-approx[2][0][0])**2)
                    dx = int(dx)
                    dy = math.sqrt((approx[2][0][1] - approx[3][0][1]) ** 2 + (approx[2][0][0] - approx[3][0][0]) ** 2)
                    dy = int(dy)

                    if color(img[int(y+(h/2)) , int(x+(w/2))],(0,128,0),(152,255,154)): #green
                        ind = id_list.index(1)
                        aruco_img = aruco_Coords(aruco_list[ind],dx,dy,h,w,-angle)
                        print(1)
                    elif color(img[int(y+(h/2)),int(x+(w/2))],(0,100,200),(153,204,255)): #orange
                        ind = id_list.index(2)
                        aruco_img = aruco_Coords(aruco_list[ind],dx,dy,h,w,-angle)
                        print(2)
                    elif color(img[int(y+(h/2)),int(x+(w/2))],(0,0,0),(20,20,20)): #black
                        ind = id_list.index(3)
                        aruco_img = aruco_Coords(aruco_list[ind],dx,dy,h,w,-angle)
                        print(3)
                    elif color(img[int(y+(h/2)),int(x+(w/2))],(200,200,200),(250,250,250)): #pink-peach
                        ind = id_list.index(4)
                        aruco_img = aruco_Coords(aruco_list[ind],dx,dy,h,w,-angle)
                        print(4)
                    cv2.drawContours(img,[approx],-1,(0,0,0),-1)
                    img[y:y+h,x:x+w] = img[y:y+h,x:x+w]+ aruco_img
        return img

final_func(img,id_list)

cv2.namedWindow("FINAL_IMAGE",cv2.WINDOW_NORMAL)
cv2.imshow("FINAL_IMAGE",final_func(img,id_list))
cv2.waitKey(0)
cv2.destroyAllWindows()