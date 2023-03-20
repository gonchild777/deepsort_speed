from AIDetector_pytorch import Detector
import imutils
import cv2
import os
from collections import defaultdict
import tracker as tracker
import math
os.environ['KMP_DUPLICATE_LIB_OK']='True'



def main():

    name = 'demo'
    dict_box=dict()
    
    global frames
    #cap = cv2.VideoCapture('E:/视频/行人监控/test01.mp4')
    cap = cv2.VideoCapture('test.mp4')
    frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    det = Detector()
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)
    ##
    
    frame_num=0
    frame_dict={}


    videoWriter = None

    while True:

        # try:
        _, im = cap.read()
        if im is None:
            break
        
        result = det.feedCap(im)
        test = result['id_center']
        # check if frame_dict have thid key

        for key in test:
           if key not in frame_dict:
                tmp_arr=[]
                for  _ in range(frame_num+1):
                   tmp_arr.append([])
                frame_dict[key]=tmp_arr
                test[key].append(0)
                frame_dict[key][frame_num]= test[key]
           else:
                x,y=test[key]
                last_frame=frame_dict[key][frame_num-1]
                ## if last frame != 0
                if len(last_frame)==0:
                    test[key].append(0)
                else:
                    last_x,last_y,speed=last_frame
                    x_diff,y_diff=x-last_x,y-last_y
                    distance=round(math.sqrt(x_diff*x_diff+y_diff*y_diff),2)
                    test[key].append(distance)
                frame_dict[key].append(test[key])
           
        result = result['frame']
        
        
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))
        ######
    
        '''        if frames>2:  
            #print(tracker.id)  
            #ok
            for value in dict_box.items():
                #print("f")
                for a in range(len(value)-1):
                    color=[0,0, 255]
                    index_start=a                    
                    index_end=index_start+1
                    #print("test")
                    cv2.line(result,tuple(map(int(value[index_start]),tuple(map(int(value[index_end])))),color,thickness=5))          
        '''###
        videoWriter.write(result)
        cv2.imshow(name, result)
        k=cv2.waitKey(t)
        if k==27:  #press esc
            break
        frame_num+=1
        # if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
        #     # 点x退出
        #     break
        # except Exception as e:
        #     print(e)
        #     break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()
    
    # for key in frame_dict:
    #     print(key)
    #     print(frame_dict[key])
    # print(frame_dict[0]['1'],frame_dict[1]['1'])
if __name__ == '__main__':
    
    main()