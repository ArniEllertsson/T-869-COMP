import cv2 
import numpy as np
import time
import sys
import random 
import math

def find_line_model(points):
    #have to add epsilon for horizontal and vertical lines
    m = (points[1,1] - points[0,1]) / (points[1,0] - points[0,0] + sys.float_info.epsilon)  # slope (gradient) of the line
    c = points[1,1] - m * points[1,0]                                     # y-intercept of the line
 
    return m, c

def find_intercept_point(m, c, testpoint):
 
    x0,y0 = testpoint
    # intersection point with the model
    x = (x0 + m*y0 - m*c)/(1 + m**2)
    y = (m*x0 + (m**2)*y0 - (m**2)*c)/(1 + m**2) + c
 
    return x, y

def sample_count(prob, s):
    return np.log(1-prob)/(np.log(1-(1-np.power(np.exp(1)),s)))

def ransac(points,iter,n_samples,threshold,inline_ratio):
    ratio = 0.
    test_points = random.sample(points, n_samples)
    for i in range(iter):
        rng = np.random.default_rng()
        rand = rng.integers((len(points) - 2), size = 2)
        
        line_points = np.zeros((2,2))
        line_points[0,0]= points[rand[0]][0]
        line_points[0,1]= points[rand[0]][1]
        line_points[1,0]= points[rand[1]][0]
        line_points[1,1]= points[rand[1]][1]
                 
        m, c = find_line_model(line_points)
        
        x_list = []
        y_list = []
        num = 0
 
        # find orthogonal lines to the model for all testing points
        for test_point in test_points:

            x0 = test_point[0]
            y0 = test_point[1]
    
            # find an intercept point of the model with a normal from point (x0,y0)
            x1, y1 = find_intercept_point(m, c,test_point)
    
            # distance from point to the model
            dist = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    
            # check whether it's an inlier or not
            if dist < threshold:
                x_list.append(x0)
                y_list.append(y0)
                num += 1

            # in case a new model is better - cache it
            if num/float(n_samples) > ratio:
                ratio = num/float(n_samples)
                bestPoint1 = points[rand[0]]
                bestPoint2 = points[rand[1]]

            # we are done in case we have enough inliers
            if num > n_samples*inline_ratio:
                print("done")
                break
        
        
    return bestPoint1,bestPoint2
    

if __name__ == '__main__':
    theta = 3 #threshold
    inline_ratio = 0.80
    n_samples = 500
    itterations = 200
    cap = cv2.VideoCapture(0)
    while(1):
        startT = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            print("failed to grab frame")
            break 
        
        edge = cv2.Canny(frame,100, 200,None,3)
        #----------------------------------
        edgePixels = edge.tolist()
        
        mylist = []
        for x,ylist in enumerate(edgePixels):
            for y,val in enumerate(ylist):
                if val > 0:
                    mylist.append((x,y))
        #----------------------------------
        #task 4
        if (len(mylist)+2)>n_samples:
            p1,p2 = ransac(mylist,itterations,n_samples,theta,inline_ratio) 
            p1 = tuple(reversed(p1))
            p2 = tuple(reversed(p2))  
            cv2.line(frame,p1,p2,(255,0,0),2)   
        stopT = time.perf_counter()
        fps = 1/(stopT - startT)
       
        fps = str(int(fps))
        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 1, cv2.LINE_AA)
        
        cv2.imshow('image',frame)
        cv2.imshow('edge',edge)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       
    cv2.destroyAllWindows()       
    cap.release()        
        