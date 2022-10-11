import os
import argparse
import numpy as np
import cv2


class IoUTracker(object):
    def __init__(self, args):
        self.classNames = { 0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car',
                            8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person',
                            16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
        self.detector = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
        self.input_size = (300,300)
        self.video_path = args.video_path
        self.output_video = args.output_video
        self.sigmaL = args.sigmaL
        self.sigmaIoU = args.sigmaIoU
        self.sigmaH = args.sigmaH
        self.tMin = args.tMin

        self.active_tracks = []
        self.finished_tracks = []
        self.frame_num = 0
        self.track_id = 0

    def infer(self, img):
        """
        获取当前帧的检测结果, 用σL过滤掉得分过低的detection
        """
        img_resized = cv2.resize(img,self.input_size)
        blob = cv2.dnn.blobFromImage(img_resized, 0.007843, self.input_size, (127.5, 127.5, 127.5), False)
        self.detector.setInput(blob)
        detections = self.detector.forward()
        w = self.input_size[0]
        h = self.input_size[1]
        wFactor = img.shape[0]/w
        hFactor = img.shape[1]/h
        detections = (detections[0,0,:,1:7] * np.array([1, 1, h*hFactor, w*wFactor, h*hFactor, w*wFactor])).tolist()
        detections = [[int(cls), conf, int(xmin), int(ymin), int(xmax), int(ymax)] for cls, conf, xmin, ymin, xmax, ymax in detections if conf > self.sigmaL]
        return detections

    def draw_detections(self, img, detections):
        for cls, conf, xmin, ymin, xmax, ymax in detections:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (100,100,100))
            text = "%s:%.2f" % (self.classNames[cls], conf)
            cv2.putText(img, text, (max(xmin,15), max(ymin,15)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    def calc_IoU(self, bbox1, bbox2):
        xmin1, ymin1, xmax1, ymax1 = bbox1
        xmin2, ymin2, xmax2, ymax2 = bbox2
        s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        s2 = (xmax2 - xmin2) * (ymax2 - ymin2)
        sum_area = s1 + s2
        left = max(xmin2, xmin1)
        right = min(xmax2, xmax1)
        top = max(ymin2, ymin1)
        bottom = min(ymax2, ymax1)
        if left >= right or top >= bottom:
            return 0
        intersection = (right - left) * (bottom - top)
        iou = intersection / (sum_area - intersection ) * 1.0
        return iou

    def highest_IoU(self, bbox, detections):
        max_IoU = 0
        max_IoU_id = -1
        for i, (cls, conf, xmin, ymin, xmax, ymax) in enumerate(detections):
            IoU = self.calc_IoU(bbox, [xmin, ymin, xmax, ymax])
            if IoU > max_IoU:
                max_IoU = IoU
                max_IoU_id = i
        return max_IoU_id, max_IoU

    def draw_track(self, frame, track, draw_bboxes_num):
        bboxes, max_conf, start_frame_num, track_id = track
        xmin, ymin, xmax, ymax = bboxes[-1]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0))
        text = "id:%s len:%s" % (track_id, len(bboxes))
        cv2.putText(frame, text, (max(xmin,15), max(ymin,15)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        draw_track_start_index = len(bboxes) - draw_bboxes_num if len(bboxes) > draw_bboxes_num else 0
        for i in range(draw_track_start_index, len(bboxes)):
            xmin, ymin, xmax, ymax = bboxes[i]
            center = (xmin, ymin)
            radius = 1
            cv2.circle(frame,center,radius,(0,255,0),3)

    def draw_tracks(self, frame, draw_bboxes_num=20):
        for track in self.active_tracks:
            self.draw_track(frame, track, draw_bboxes_num)
        cv2.putText(frame, "Num of active_tracks: {}".format(len(self.active_tracks)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.putText(frame, "Num of finished_tracks: {}".format(len(self.finished_tracks)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                
    def tracker(self, detections):
        for i, track in enumerate(self.active_tracks):
            # 匹配: 每个激活状态轨迹的最新bbox与当前帧detections计算IoU,并用σIoU限制IoU最大的detection是否可加入该轨迹
            max_IoU_id, max_IoU = self.highest_IoU(track[0][-1], detections)
            if max_IoU_id != -1 and max_IoU >= self.sigmaIoU:
                track[0].append(detections[max_IoU_id][-4:]) # 更新轨迹bboxes
                if track[1] < detections[max_IoU_id][1]:
                    track[1] = detections[max_IoU_id][1] # 更新max_conf
                detections.pop(max_IoU_id)  # 删除已经匹配上的detection
                self.active_tracks[i] = track  # 更新track
            else:
                # 如果当前帧所有detection都没有与该轨迹匹配上, 该轨迹失活并从激活状态轨迹中剔除
                # 失活轨迹中至少一个detection得分高于σh；最小轨迹长度tMin
                if track[1] >= self.sigmaH and len(track[0]) >= self.tMin:
                    self.finished_tracks.append(track)  # 加入失活轨迹
                else:
                    self.track_id -= 1
                self.active_tracks.remove(track)  # 从激活状态轨迹中剔除
                i -= 1
        # 当前帧未匹配的detection，作为新的激活状态轨迹
        for cls, conf, xmin, ymin, xmax, ymax in detections:
            bbox = [xmin, ymin, xmax, ymax]
            max_conf = conf
            start_frame_num = self.frame_num
            track = [[bbox], max_conf, start_frame_num, self.track_id]
            self.active_tracks.append(track)
            self.track_id += 1

    def process_video(self):
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
        else:
            cap = cv2.VideoCapture(0)
        if self.output_video:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_video_path = os.path.splitext(self.video_path)[0]+"_out.mp4v"
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), True)
        while True:
            ret, frame = cap.read()
            if ret:
                detections = self.infer(frame)  
                self.draw_detections(frame, detections)
                self.tracker(detections)
                self.draw_tracks(frame)

                self.frame_num += 1
                cv2.imshow("IoUTracker", frame)
                if cv2.waitKey(1) == 27:  # Press ESC key to exit
                    break
                if self.output_video:
                    video_writer.write(frame)
            else:
                break
        # 获取最终所有的轨迹(可省略)
        for track in self.active_tracks:
            if track[1] >= self.sigmaH and len(track[0]) >= self.tMin:
                self.finished_tracks.append(track)
        return self.finished_tracks


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="testdata/test1.mp4", help="path to video file. If empty, camera's stream will be used")
    parser.add_argument("--output_video", default=True, action="store_true", help="output processed video")
    parser.add_argument("--prototxt", default="testdata/MobileNetSSD_deploy.prototxt", help='Path to text network file')
    parser.add_argument("--weights", default="testdata/MobileNetSSD_deploy.caffemodel", help='Path to weights')
    parser.add_argument("--sigmaL", default=0.3, type=float, help="用sigmaL滤除得分过低的detections")
    parser.add_argument("--sigmaH", default=0.8, type=float, help="轨迹中至少一个detection得分高于sigmaH")
    parser.add_argument("--sigmaIoU", default=0.5, type=float, help="每个激活状态轨迹的最新bbox与当前帧detections计算IoU,并用sigmaIoU限制IoU最大的detection是否可加入该轨迹")
    parser.add_argument("--tMin", default=3, type=int, help="最小轨迹长度tMin")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    tracker = IoUTracker(get_arguments())
    finished_tracks = tracker.process_video()