"""
@author: Pradnya Mundargi
"""
import numpy as np
import cv2
#import matplotlib.pyplot as plt


def roi(image):
    pts=np.array([[70,540],[900,540],[480,310]])
    mask=np.zeros_like(image)
    cv2.fillConvexPoly(mask,pts,1)
    #cv2.fillPoly(mask, pts, 255)
    final_img=cv2.bitwise_and(image, image, mask=mask)
    return final_img

def count_blobs(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)


cap=cv2.VideoCapture(r'C:\Users\mprad\OneDrive\Desktop\Spring_2022\Perception\Project2\whiteline.mp4')
if (cap.isOpened()== False):
  print("Error opening video file")


img=[]
while(cap.isOpened()):
  ret, frame = cap.read()
  gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  crop=roi(gray)
  _,thresh1 = cv2.threshold(crop,200,255,cv2.THRESH_BINARY)
  #check_left=frame[:,:480]
  lines = cv2.HoughLinesP(thresh1, 2, np.pi/180, 1, minLineLength=100, maxLineGap=80)
  for line in lines:
    x1, y1, x2, y2 = line[0]
    
    check_left=thresh1[:,:480]
    check_right=thresh1[:,480:]
    if count_blobs(check_left)>= count_blobs(check_right):
        if x1<480 and x2<480:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        else:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    else:
        if x1>480 and x2>480:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        else:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3) 

  if ret == True:
    cv2.imshow('Gray',frame)
    img.append(frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
      break
  else:
      pass
  
cap.release()
cv2.destroyAllWindows()
# Animation file
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 10, (960,540))
for i in range(len(img)):
    pic = np.array(img[i])
    out.write(pic)

out.release() 

