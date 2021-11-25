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
    bestPoints = []
    points_copy = points
    for i in range(4): #get four lines 
        test_points = random.sample(points_copy, n_samples)
        bestInnerP = []
        ratio = 0.
        for i in range(iter):
            rng = np.random.default_rng()
            rand = rng.integers(0,len(points_copy) , size = 2)
            
            line_points = np.zeros((2,2))
            line_points[0,0]= points_copy[rand[0]][0]
            line_points[0,1]= points_copy[rand[0]][1]
            line_points[1,0]= points_copy[rand[1]][0]
            line_points[1,1]= points_copy[rand[1]][1]
                    
            m, c = find_line_model(line_points)
            
            inliners_list = []
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
                    inliners_list.append((x0,y0)) #add inliners to a list
                    num += 1

                # in case a new model is better - cache it
                if num/float(n_samples) > ratio:
                    ratio = num/float(n_samples)
                    bestP1 =  points_copy[rand[0]]
                    bestP2 =  points_copy[rand[1]]
                    bestInnerP = inliners_list
                    

                # we are done in case we have enough inliers
                if num > n_samples*inline_ratio:
                    print("done")
                    break
        #add the best line points to the list
        bestPoints.append(bestP1)
        bestPoints.append( bestP2)
        for i in range(len(bestInnerP)): 
            #delete all inner poins 
            points_copy.remove(bestInnerP[i])
           
            
        
    return bestPoints

def line_intersection(p1,p2,p3,p4):
  
    d = (p1[0]-p2[0])*(p3[1]-p4[1]) - (p3[0]-p4[0])*(p1[1]-p2[1])
    if d == 0:
        return -1,-1
    
    x = (p1[0]*p2[1]-p1[1]*p2[0])*(p3[0]-p4[0]) - (p1[0]-p2[0])*(p3[0]*p4[1]-p3[1]*p4[0])
    x = x/d
    y = (p1[0]*p2[1]-p1[1]*p2[0])*(p3[1]-p4[1]) - (p1[1]-p2[1])*(p3[0]*p4[1]-p3[1]*p4[0])
    y = y/d
    
    #if the mach is outside the image 
    if (x > 480) | (y>640):
        return -1,-1
    return x, y

if __name__ == '__main__':
    theta = 3 #threshold
    inline_ratio = 0.80
    itterations = 100
    cap = cv2.VideoCapture(0)
    while(1):
        startT = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            print("failed to grab frame")
            break 
        scale_factor = 0.5
        #frame =cv2.resize(frame, None, fx= scale_factor, fy= scale_factor
        #                 , interpolation= cv2.INTER_LINEAR)
        edge = cv2.Canny(frame,100, 200,None,3)
        
        lines = cv2.HoughLines(edge, 1, np.pi / 180,100)
        numb = 0
        points = []
        if lines is not None:
            if len(lines) > 4:
                for i in range(4):
                    rho = lines[i][0][0]
                    theta = lines[i][0][1]
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                    points.append(pt1)
                    points.append(pt2)
                    cv2.line(frame, pt1, pt2, (0,0,255), 2)

                numb = 0
                intersection_points = []
                #ind intersection line 1 with all
                for i in range(2,len(points),2):
                    x,y = line_intersection(points[0],points[1],points[i],points[i+1])      
                    if x  > 1:
                        intersection_points.append((int(x),int(y)))
                        numb +=1
                        cv2.circle(frame, (int(x),int(y)), 10, (255, 0, 0), 2) 
                  
                    
                #compare line 2 with 3 and 4
                for i in range(4,len(points),2):
                    x,y = line_intersection(points[2],points[3],points[i],points[i+1])      
                    if x  > 1:
                        intersection_points.append((int(x),int(y)))
                        numb +=1
                        cv2.circle(frame, (int(x),int(y)), 10, (255, 0, 0), 2) 
    
                #find intersection with line 3 and 4 
                x,y = line_intersection(points[4],points[5],points[6],points[7])      
                if x  > 1:
                    intersection_points.append((int(x),int(y)))
                    numb +=1
                    cv2.circle(frame, (int(x),int(y)), 10, (255, 0, 0), 2)               
   
                        
            

        #----------------------------------
        '''

        #for ransac lines
        edgePixels = edge.tolist()
        
        mylist = []
        for x,ylist in enumerate(edgePixels):
            for y,val in enumerate(ylist):
                if val > 0:
                    mylist.append((x,y))
        #----------------------------------
         
        n_samples = int(len(mylist)*0.3)
        points = ransac(mylist,itterations,n_samples,theta,inline_ratio)
          
        cv2.line(frame,tuple(reversed(points[0])),tuple(reversed(points[1])),(0,255,0),2)
        cv2.line(frame,tuple(reversed(points[2])),tuple(reversed(points[3])),(255,0,0),2)
        cv2.line(frame,tuple(reversed(points[4])),tuple(reversed(points[5])),(0,0,255),2)
        cv2.line(frame,tuple(reversed(points[6])),tuple(reversed(points[7])),(0,255,255),2)           
        '''
        stopT = time.perf_counter()
        
        fps = 1/(stopT - startT)
       
        fps = str(int(fps))
        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 1, cv2.LINE_AA)
        
        cv2.imshow('image',frame)
        cv2.imshow('edge',edge)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('source.png', frame)
            print(intersection_points)
            break

       
       
    cv2.destroyAllWindows()       
    cap.release()        
        