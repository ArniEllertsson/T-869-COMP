import cv2 as cv
import time
import numpy as np

def detect(outputs,img):
    hT,wT,cT = img.shape
    boxs = []
    classIds = []
    confs = []
    confThreshold = 0.4
    nmsThreshold = 0.5
    
    for output in outputs: 
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w , h = int(det[2]*wT) , int(det[3]*hT)
                x , y = int((det[0]*wT)-w/2) , int((det[1]*hT) - h/2)
                boxs.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv.dnn.NMSBoxes(boxs,confs,confThreshold,nmsThreshold)
    
    for i in indices:      
        box = boxs[i[0]]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv.rectangle(img,(x,y),(x+w,y+h),colors[classIds[i[0]]],2)
        cv.putText(img,f'{classes[classIds[i[0]]].upper()} {int(confs[i[0]]*100)}%',
                    (x,y-10),cv.FONT_HERSHEY_SIMPLEX,0.6,colors[classIds[i[0]]],2)

                   #cv.rectangle(img,(x,y), (x+w, y+h),colors[classIds[i[0]]],2)
if __name__ == '__main__':
    modelConf = 'yolov3-tiny.cfg'
    modelWeights = 'yolov3-tiny.weights'
    net = cv.dnn.readNetFromDarknet(modelConf,modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    classes = []
    with open('coco.name', 'r') as f:
        classes = f.read().split('\n')
    
    cam=cv.VideoCapture('http://192.168.1.105:8080/video') #phone steam
    #cam = cv.VideoCapture(0) #computer webcam strea
    colors = np.random.uniform(0, 255, size=(len(classes), 3))   
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

    while True:
        startT = time.perf_counter()
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        # create blob from image

        #frame = frame[:416,:416]
        frame = cv.resize(frame, (416,416), interpolation = cv.INTER_AREA)
        blob = cv.dnn.blobFromImage(frame,1/255,(frame.shape[1],frame.shape[0]),[0,0,0],1,crop = False)
        # set the blob to the model
        net.setInput(blob)
        # forward pass through the model to carry out the detection
        output = net.forward(outputNames)
    
        detect(output,frame)
        stopT = time.perf_counter()
        fps = 1/(stopT - startT)
        cv.putText(frame,(f'FPS : {np.round(fps,3)}'),(7, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow("camera",frame)
        
        if cv.waitKey(1) == ord('q'):
            break

    cv.destroyAllWindows()       
    cam.release()        
        