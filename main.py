import numpy as np
import cv2, time

img = cv2.imread('scanned-form.jpg')
img = cv2.resize(img, (int(img.shape[1]*.75), int(img.shape[0]*.75)))
clone = img.copy()
final = img.copy()
cv2.namedWindow("image")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (21, 21), 0)
gray = cv2.medianBlur(gray, 33)
gray_edge = cv2.Canny(gray,33,120)

contours, heirarchy = cv2.findContours(gray_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def distance(pt1, pt2):
    pt2 = tuple(pt2)
    if (np.array(pt1) < 0).any() | (np.array(pt2) < 0).any():
        return np.inf
    (x1, y1), (x2, y2) = pt1, pt2
    dist = np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    return dist

def nearest_point(x,y,kp):
    min_d = np.inf
    for ix, p in enumerate(kp):
        p = tuple(p)[0]
        if (distance((x,y),p)) < min_d:
            min_d = distance((x,y),p)
            min_ix = ix
    return min_ix

p = np.array([tuple(c[0][0]) for c in contours])
kp = order_points(p).astype(np.int32)
kp = kp.reshape(-1,1,2)

btn_down = False
outpoints = []

def mouse_handler(event, x, y, flags, data):

    global btn_down, outpoints, clone, final, kp

    if event == cv2.EVENT_LBUTTONDOWN:
        if btn_down == False:
            btn_down = True

    elif event == cv2.EVENT_MOUSEMOVE and btn_down:

        for i in range(4):

            cv2.circle(clone, tuple(kp[i][0]), 20, (255, 88, 50), 5)
 
            cv2.putText(clone, "{}".format(i+1), 
                        tuple((kp[i][0][0]+20, kp[i][0][1]-20)), 
                        FONTFACE, FONTSCALE, (255, 88, 50), FONTTHICKNESS, cv2.LINE_AA)

            cv2.imshow("image", cv2.polylines(clone, [kp], True, (0,0,255), 8)[...,::-1])

        clone = final.copy()

        for i in range(4):

            cv2.circle(clone, tuple(kp[i][0]), 20, (255, 88, 50), 5)
            cv2.putText(clone, "DRAG CIRCLES WITH MOUSE TO ADJUST", 
                        (100, 80), FONTFACE, FONTSCALE , FONTCOLOR, FONTTHICKNESS, cv2.LINE_AA)
            cv2.putText(clone, "{}".format(i+1), 
                        tuple((kp[i][0][0]+20, kp[i][0][1]-20)), 
                        FONTFACE, FONTSCALE, (255, 88, 50), FONTTHICKNESS, cv2.LINE_AA)

            cv2.imshow("image", cv2.polylines(clone, [kp], True, (0,0,255), 8)[...,::-1])

        kp[nearest_point(x,y,kp)] = ((x,y))
        cv2.circle(clone, (x, y), 3, (50, 88, 50),3)
        cv2.imshow("image", clone)

    elif event == cv2.EVENT_LBUTTONUP:

        btn_down = False

        cv2.putText(clone, "DRAG CIRCLES WITH MOUSE TO ADJUST", 
                    (100, 80), FONTFACE, FONTSCALE , FONTCOLOR, FONTTHICKNESS, cv2.LINE_AA)

        cv2.putText(clone, "TYPE q TO EXIT AND SAVE", 
                        (100, 120), FONTFACE, FONTSCALE , FONTCOLOR, FONTTHICKNESS, cv2.LINE_AA)
        
        for i in range(4):

            cv2.circle(clone, tuple(kp[i][0]), 20, (255, 88, 50), 5)
    
            cv2.putText(clone, "{}".format(i+1), 
                        tuple((kp[i][0][0]+20, kp[i][0][1]-20)), 
                        FONTFACE, FONTSCALE, (255, 88, 50), FONTTHICKNESS, cv2.LINE_AA)

            cv2.imshow("image", cv2.polylines(clone, [kp], True, (0,0,255), 8)[...,::-1])

cv2.setMouseCallback("image", mouse_handler)

while True:

    FONTFACE = cv2.FONT_HERSHEY_COMPLEX
    FONTSCALE = .9
    FONTCOLOR = (255, 88, 50)
    FONTTHICKNESS = 2

    cv2.putText(clone, "DRAG CIRCLES WITH MOUSE TO ADJUST", 
                (100, 80), FONTFACE, FONTSCALE , FONTCOLOR, FONTTHICKNESS, cv2.LINE_AA)
    cv2.putText(clone, "TYPE q TO EXIT AND SAVE", 
                    (100, 120), FONTFACE, FONTSCALE , FONTCOLOR, FONTTHICKNESS, cv2.LINE_AA)

    for i in range(4):

        cv2.circle(clone, tuple(kp[i][0]), 20, (255, 88, 50), 5)

        cv2.putText(clone, "{}".format(i+1), 
                    tuple((kp[i][0][0]+20, kp[i][0][1]-20)), 
                    FONTFACE, FONTSCALE, (255, 88, 50), FONTTHICKNESS, cv2.LINE_AA)

        cv2.imshow("image", cv2.polylines(clone, [kp], True, (0,0,255), 10)[...,::-1])

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

outpoints = kp
print('\nFINAL KEYPOINTS\n', outpoints)
pts_src = np.array(outpoints, dtype=float)
pts_dst = np.array([[0,0], [500, 0], [500,667],[0,667]], dtype=float)
h, status = cv2.findHomography(pts_src, pts_dst)
im_out = cv2.warpPerspective(img, h, (500, 667))
cv2.imshow('scanned final image', im_out)
cv2.waitKey(0) & 0xFF
cv2.imwrite('scanned_final.jpg', im_out)
cv2.destroyAllWindows()
