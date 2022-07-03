"""
@author: Pradnya Mundargi
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


def roi(image):
    pts=np.array([[0,720],[1230,720],[690,415]])
    mask=np.zeros_like(image)
    cv2.fillConvexPoly(mask,pts,1)
    #cv2.fillPoly(mask, pts, 255)
    final_img=cv2.bitwise_and(image, image, mask=mask)
    return final_img

def warp(image):
    src=np.float32([(0.45,0.65),(0.15,1),(0.60,0.65),(1,1)])
    dst=np.float32([(0,0), (0,1), (1,0),(1,1)])
    dst_img_size=(image.shape[1],image.shape[0])
    img_size = np.float32([(image.shape[1],image.shape[0])])
    dst = dst * np.float32(dst_img_size)
    src = src* img_size
    W = cv2.getPerspectiveTransform(src, dst)
    warp_img = cv2.warpPerspective(image, W, dst_img_size)
    return warp_img


left1=[]
left2=[]
left3=[]
right1=[]
right2=[]
right3=[]

def sliding_window(image, limit=100, win_num=4):
    global left1, left2, left3,right1, right2, right3 
    check_count=0
    final = np.dstack((image, image, image))*255

    # Histogram peaks for left and right lanes 
    histogram = np.sum(image[image.shape[0]//2:,:], axis=0)
    left_half = np.argmax(histogram[:int(histogram.shape[0]/2)])
    right_half = np.argmax(histogram[int(histogram.shape[0]/2):]) + int(histogram.shape[0]/2)
    
    # All the x and y positions with nonzero pixels values
    dummy = image.nonzero()
    x_index = np.array(dummy[1])
    y_index= np.array(dummy[0])
    left_curr = left_half
    right_curr = right_half
    left_indices = []
    right_indices = []
    window_height = np.int(image.shape[0]/win_num)
    
    # Iterate through each window
    for win in range(win_num):
        left_x_bottom = left_curr - limit
        right_x_bottom = right_curr - limit
        left_x_top = left_curr + limit
        right_x_top = right_curr + limit
        y_top = image.shape[0] - win*window_height
        y_bottom = image.shape[0] - (win+1)*window_height
        check_count=1
        # Box visualization 
        if check_count==1:
            cv2.rectangle(final,(right_x_bottom,y_bottom),(right_x_top,y_top),
            (0,255,255), 3) 
            cv2.rectangle(final,(left_x_bottom,y_bottom),(left_x_top,y_top),
            (0,255,255), 3) 
            
        keep_left = ((x_index < left_x_top) & (x_index >= left_x_bottom) & (y_index < y_top) & (y_index >= y_bottom)).nonzero()[0]
        keep_right = ((x_index < right_x_top) & (x_index >= right_x_bottom) & (y_index >= y_bottom) & (y_index < y_top)).nonzero()[0]
        left_indices.append(keep_left)
        right_indices.append(keep_right)
      
    right_indices = np.concatenate(right_indices)
    left_indices = np.concatenate(left_indices)
    
    # Extract left and right line pixel positions
    right_x = x_index[right_indices]
    right_y = y_index[right_indices] 
    left_x = x_index[left_indices]
    left_y = y_index[left_indices] 
    
    
    # Fit polynomial
    right_lane_eq = np.zeros(3) # 3 coefficient array
    left_lane_eq= np.zeros(3)
    right_fit = np.polyfit(right_y, right_x, 2)
    left_fit = np.polyfit(left_y, left_x, 2)
    
    right1.append(right_fit[0])
    right2.append(right_fit[1])
    right3.append(right_fit[2])
    left1.append(left_fit[0])
    left2.append(left_fit[1])
    left3.append(left_fit[2])
    
    right_lane_eq[0] = np.mean(right1[-10:])
    right_lane_eq[1] = np.mean(right2[-10:])
    right_lane_eq[2] = np.mean(right3[-10:])
    left_lane_eq[0] = np.mean(left1[-10:])
    left_lane_eq[1] = np.mean(left2[-10:])
    left_lane_eq[2] = np.mean(left3[-10:])
    
    
    # Generate x and y values for plotting
    curve_plot = np.linspace(0, image.shape[0]-1, image.shape[0] )
    right_curve = right_lane_eq[0]*curve_plot**2 + right_lane_eq[1]*curve_plot + right_lane_eq[2]
    left_curve = left_lane_eq[0]*curve_plot**2 + left_lane_eq[1]*curve_plot + left_lane_eq[2]
    final[y_index[right_indices], x_index[right_indices]] = [0, 0, 255]
    final[y_index[left_indices], x_index[left_indices]] = [0, 255, 0]
    return final, (left_curve, right_curve), (left_lane_eq, right_lane_eq), curve_plot



def radius_of_curvature(image, left_curve, right_curve):
    curve_plot = np.linspace(0, image.shape[0]-1, image.shape[0])
    max_ = np.max(curve_plot)
    new_right_curve = np.polyfit(curve_plot, right_curve, 2)
    new_left_curve = np.polyfit(curve_plot, left_curve, 2)
    
    #Radius of curvature
    right_c_rad = ((1 + (2*new_right_curve[0]*max_+new_right_curve[1])**2)**1.5) / np.absolute(2*new_right_curve[0])
    left_c_rad = ((1 + (2*new_left_curve[0]*max_ + new_left_curve[1])**2)**1.5) / np.absolute(2*new_left_curve[0])
    avg_rad= (right_c_rad + left_c_rad)/2
    print('Left ROC:',left_c_rad)
    print('Right ROC:',right_c_rad)
    print('Average ROC:',avg_rad)
    return (left_c_rad, right_c_rad, avg_rad)

def inv_warp(image):
    src=np.float32([(0,0), (0,1), (1,0),(1,1)])
    dst=np.float32([(0.45,0.65),(0.15,1),(0.60,0.65),(1,1)])
    dst_img_size=(image.shape[1],image.shape[0])
    img_size = np.float32([(image.shape[1],image.shape[0])])
    dst = dst * np.float32(dst_img_size)
    src = src* img_size
    W = cv2.getPerspectiveTransform(src, dst)
    inv_warp_img = cv2.warpPerspective(image, W, dst_img_size)
    return inv_warp_img

def final_plots(image, left_fit, right_fit):
    curve_plot = np.linspace(0, image.shape[0]-1, image.shape[0])
    color_img = np.zeros_like(image)
    
    left = np.array([np.transpose(np.vstack([left_fit, curve_plot]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, curve_plot])))])
    points = np.hstack((left, right))
    
    cv2.fillPoly(color_img, np.int_(points), (200,0,0))
    inv_perspective = inv_warp(color_img)
    inv_perspective = cv2.addWeighted(image, 1, inv_perspective, 0.7, 0)
    return inv_perspective



cap=cv2.VideoCapture(r'C:\Users\mprad\OneDrive\Desktop\Spring_2022\Perception\Project2\challenge.mp4')
if (cap.isOpened()== False):
  print("Error opening video file")


img=[]
w_img=[]
final_img=[]
window_track=[]
while(cap.isOpened()):
  ret, frame = cap.read()
  gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  crop=roi(gray)
  _,thresh1 = cv2.threshold(crop,200,255,cv2.THRESH_BINARY)
  
  w=warp(thresh1)
  out_img, curves, lanes, ploty=sliding_window(w)
  curverad=radius_of_curvature(frame, curves[0],curves[1])
  img_ = final_plots(frame, curves[0], curves[1])
  
  if ret == True:
    cv2.imshow('Gray',img_)
    img.append(frame)
    w_img.append(w)
    window_track.append(out_img)
    final_img.append(img_)
    
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
      break
  else:
      pass
  
cap.release()
cv2.destroyAllWindows()

# Animation file
out1 = cv2.VideoWriter('Input_frame.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 10, (1280,720))
for i in range(len(img)):
    pic = np.array(img[i])
    out1.write(pic)

out1.release()

out2 = cv2.VideoWriter('warp.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 10, (1280,720))
for i in range(len(w_img)):
    pic1 =cv2.cvtColor(w_img[i],cv2.COLOR_GRAY2RGB)
    pic1 = np.array(pic1)
    out2.write(pic1)

out2.release()

out3 = cv2.VideoWriter('Curve_Track.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 10, (1280,720))
for i in range(len(window_track)):
    pic2 = np.array(window_track[i])
    out3.write(pic2)
out3.release() 


out4 = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 10, (1280,720))
for i in range(len(final_img)):
    pic3 = np.array(final_img[i])
    out4.write(pic3)
    
out4.release() 