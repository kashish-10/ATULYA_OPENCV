# ATULYA_OPENCV

PROBLEM STATEMENT - https://docs.google.com/document/d/16pL5th0OLdQVD4YZw9joZ5izpAZ75c9rf4CseUSPSwA/edit

#PSEUDOCODE 

Required libraries namely numpy , cv2, aruco , imutils , math were imported.

A function named findAruco was then created to find the ids of the arucos.

Another function named aruco_Coords was then created to find the co-ordinates of the corners of the four aruco markers.

Another function named color was created to find the color of the different squares.

The inclination angle of the arucos was then determined using the coordinates obtained above.

The arucos were then made straight by rotating them by those angles.

The rotated arucos were then cropped to remove the extra white spaces around them.

The CVtask image was then converted to grayscale , thresholded and contours were found.

Comtours were then drawn around the squares which were determined using the ratio of their sides.

Their side lengths were then determined using the result obtained from the function approxPolyDP.

Then a blank image of same shape as the squares was cretaed.

The cropped and rotated arucos were then imposed over the blank images to match the shape of the respective sqaures.

Finally the new arucos were imposed over the respective squares.
