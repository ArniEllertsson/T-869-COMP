import cv2
import time
import numpy as np

if __name__ == '__main__':

    #stream = cv2.VideoCapture('http://192.168.1.105:8080/video') #phone steam
    stream = cv2.VideoCapture(0) #computer webcam stream
   
    while True:
        startT = time.perf_counter()
        ret, frame = stream.read()
        if not ret:
            print("failed to grab frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        #cv2.circle(frame, maxLoc, 20, (255, 0, 0), 2) #blue circle for brightest value
        
        
        frame_hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        #create two mask for red in hsv space
        # lower mask (0-10)
        lower_red = np.array([0,100,100])
        upper_red = np.array([10,255,255])
        mask0 = cv2.inRange(frame_hsv, lower_red, upper_red)

        # upper mask (170-180)
        lower_red = np.array([170,100,100])
        upper_red = np.array([180,255,255])
        mask1 = cv2.inRange(frame_hsv, lower_red, upper_red)
        
        mask = mask0|mask1
        
        red_hsv = frame_hsv.copy()
        red_hsv = cv2.bitwise_and(red_hsv, red_hsv, mask= mask)
        
        red_rbg = frame.copy()
        red_rbg = cv2.bitwise_and(red_rbg, red_rbg, mask= mask)
        
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(red_hsv[:,:,1])
        #cv2.circle(frame, maxLoc, 10, (255, 0, 255), 2) #pink circle for reddest value
        
        '''
        #find brightest value with for loop
        maxVal = 0
        maxLoc = (0,0)
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                if (gray[i,j] > maxVal):
                    maxVal = gray[i,j]
                    maxLoc  = (j,i)
        
        
        cv2.circle(frame, maxLoc, 10, (0, 255, 255), 2) #yellow circle for brightest value
        
        #find reddest value with for loop
        maxVal = 0
        maxLoc = (0,0)

        for i in range(red_hsv.shape[0]):
            for j in range(red_hsv.shape[1]):
                if red_hsv[i,j,1] > maxVal:
                    maxVal = red_hsv[i,j,1]
                    maxLoc  = (j,i)
             
        cv2.circle(frame, maxLoc, 10, (255, 255, 0), 2) #cyan circle for reddest value
        '''
        
        stopT = time.perf_counter()
        fps = 1/(stopT - startT)
       
        fps = str(int(fps))
        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 1, cv2.LINE_AA)
        
        resized = cv2.resize(frame, (640,480), interpolation = cv2.INTER_AREA) #resized because of phone resulution
        resizedred = cv2.resize(red_rbg, (640,480), interpolation = cv2.INTER_AREA)
        cv2.imshow("camera", resized) 
        #cv2.imshow("redFilter", red_rbg)
        #cv2.imshow("gray", gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
        
       
    cv2.destroyAllWindows()       
    stream.release()
