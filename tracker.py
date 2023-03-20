from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
from collections import defaultdict

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

global center_point
def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    ##
    centroid_dic=defaultdict
    object_id_list=[]
    thickness = 2
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        if cls_id in ['person']:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        c1, c2 = (x1, y1), (x2, y2)
        cx = int(x1+x2)/2
        cy = int(y1+y2)/2
        global cc 
        cc=(cx,cy)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        ##
        cv2.circle(image,  (int((x1+x2)/2),int((y1+y2)/2)), 2, color, thickness)

        #建構出每一禎需要的資訊 中心點XY ID FRAME 做成dictionary
        '''centroid_dic[pos_id].append(cx)
        centroid_dic[pos_id].append(cy)

        if pos_id not in object_id_list:
            for id in range(len(pos_id)):
                object_id_list.append(pos_id)
                startpt=(cx,cy)
                endpt=(cx,cy)
                cv2.line(image,  startpt,endpt, 2, color, thickness)
        else:
            l=len(centroid_dic[pos_id])
            for pt in range (len(centroid_dic[pos_id])):
                if not pt+1 == 1:
                    startpt=(centroid_dic[pos_id][pt][0],centroid_dic[pos_id][pt][1])
                    endpt=(centroid_dic[pos_id][pt+1][0],centroid_dic[pos_id][pt+1][1])
                    cv2.line(image,  startpt,endpt, 2, color, thickness) '''


    return image

dict_box=dict()
def update_tracker(target_detector, image):
    global center_point
    center_point = defaultdict()
    id_center={}
    #each frame do
    new_faces = []
    _, bboxes = target_detector.detect(image)

    bbox_xywh = []
    confs = []
    clss = []

    for x1, y1, x2, y2, cls_id, conf in bboxes:

        obj = [
            int((x1+x2)/2), int((y1+y2)/2),
            x2-x1, y2-y1
        ]
        bbox_xywh.append(obj)
        confs.append(conf)
        clss.append(cls_id)

    xywhs = torch.Tensor(bbox_xywh)
    confss = torch.Tensor(confs)

    outputs = deepsort.update(xywhs, confss, clss, image)

    bboxes2draw = []
    face_bboxes = []
    current_ids = []

    for value in list(outputs):
        #value is detection boxes
        #track_id is id 
        x1, y1, x2, y2, cls_, track_id = value
        bboxes2draw.append(
            (x1, y1, x2, y2, cls_, track_id)
        )
        #print(track_id)
        #if frames>2:  
            #print(tracker.id)  
            #ok
        #for value in dict_box.items():
            #print("f")
            #for a in range(len(value)-1):
        color=[0,0, 255]       
        #找出某個ID的中心點(track_id) 然後存在外面的center_point
        #在這邊寫CV2.line 畫線
        #draw the line
        #cv2.line(image,tuple(map(int(value[index_start]),tuple(map(int(value[index_end])))),color,thickness=5))          
        center=[((x1+x2)/2),((y1+y2)/2)]
        current_ids.append(track_id)
        id_center[str(track_id)]=center
        dict_box.setdefault(track_id,[]).append(center)
        #要寫FOR 把每一禎都畫出來 不能只跟上一禎畫




        
        if cls_ == 'face':
            if not track_id in target_detector.faceTracker:
                target_detector.faceTracker[track_id] = 0
                face = image[y1:y2, x1:x2]
                new_faces.append((face, track_id))
            face_bboxes.append(
                (x1, y1, x2, y2)
            )
            







    ids2delete = []
    for history_id in target_detector.faceTracker:
        if not history_id in current_ids:
            target_detector.faceTracker[history_id] -= 1
        if target_detector.faceTracker[history_id] < -5:
            ids2delete.append(history_id)

    for ids in ids2delete:
        target_detector.faceTracker.pop(ids)
        print('-[INFO] Delete track id:', ids)

    image = plot_bboxes(image, bboxes2draw)

    return image, new_faces, face_bboxes,id_center
