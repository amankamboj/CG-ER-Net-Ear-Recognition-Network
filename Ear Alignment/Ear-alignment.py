import cv2
import numpy as np


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    print('final angle', angle)
    (h, w) = image.shape[:2]
    print('w,h=',w,h)
    (cX, cY) = (w // 2, h // 2)
    x_c=int(cX)
    y_c=int(cY)
    print('final image',x_c, y_c)


    org = cv2.circle(image, (x_c,y_c), radius=6, color=(255, 255, 255), thickness=-6)
    cv2.imwrite('rotatedimage_final.jpg',org)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

img_org = cv2.imread('151.jpg')
img = cv2.imread('151mask.jpg',0)
ret,thresh = cv2.threshold(img,127,255,0)
contours, hierarchy = cv2.findContours(thresh,1,2)
#print((np.array(img1)).shape)
#ret,thresh = cv2.threshold(img1,127,255,0)
#im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
#cnt = cnt[1]
org=img.copy()
print('All contour',contours)
# Find the index of the largest contour
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt=contours[max_index] 


print('final contour',cnt)
x,y,w,h = cv2.boundingRect(cnt)
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imwrite('minrectange_mask.jpg',img)

rect = cv2.minAreaRect(cnt)
print(rect)

box = cv2.boxPoints(rect)
#cv2.circle(img,, 63, (0,0,255), -1)
#print(box)
print(box)
box = np.int0(box)
img=cv2.drawContours(img,[box],0,(255,0,0),5)
#cv2.imwrite('minrectange2_mask.jpg',img)
angle=rect[2]
print('caluated angle',angle)
#dst1 = rotate_bound(img_org,(angle))


if angle<0:
    print('caluated angle',angle)
    #dst1 = rotate_image(img_original,x_c,y_c,(angle+90))
    #angle=np.degrees(np.arctan(angle))
    dst1 = rotate_bound(img_org,(angle+90))
if angle>0:
    print('caluated angle',angle)
    #angle=np.degrees(np.arctan(angle))
    
    dst1 = rotate_bound(img_org,(angle-90))
#final=rotate_bound(org,angle)


[x1,y1]=box[0]
[x2,y2]=box[1]
[x3,y3]=box[2]
[x4,y4]=box[3]
#angle=(y2-y1)/(x2-x1+1.01)
angle=rect[2]
y_c, x_c = img.shape[:2]

x_c=int(x_c/2)
y_c=int(y_c/2)
print(y_c, x_c)

print('angle',angle)

org = cv2.circle(org, (x1,y1), radius=10, color=(255, 255, 255), thickness=-10)
org = cv2.circle(org, (x2,y2), radius=10, color=(255, 255, 255), thickness=-10)
org = cv2.circle(org, (x3,y3), radius=10, color=(255, 255, 255), thickness=-10)
org = cv2.circle(org, (x4,y4), radius=10, color=(255, 255, 255), thickness=-10)

org = cv2.circle(org, (x_c,y_c), radius=20, color=(0, 0, 255), thickness=-20)
org=cv2.drawContours(org,[box],0,(255,0,0),5)
img_org = cv2.circle(img_org, (x_c,y_c), radius=10, color=(0, 0, 255), thickness=-10)
cv2.imwrite('point_and_rectangle_mask.jpg',org)
#cv2.imwrite('point_and_rectangle_original.jpg',img_org)
cv2.imwrite('rotatedimage_mask.jpg',dst1)



