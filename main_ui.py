import os
import cv2
import time
import torch
import argparse
import numpy as np
from tkinter import filedialog
from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls
from PIL import ImageTk
from PIL import Image as tkimg
from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG
from tkinter import *
from tkinter.ttk import *
def preproc(image):

    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):

    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('--detection_input_size', type=int, default=384,
                        help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_input_size', type=str, default='224x160',
                        help='Size of input in pose model must be divisible by 32 (h, w)')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                        help='Backbone model for SPPE FastPose model.')
    par.add_argument('--show_detected', default=False, action='store_true',
                        help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                        help='Show skeleton pose.')
    par.add_argument('--save_out', type=str, default='',
                        help='Save display to video file.')
    par.add_argument('--device', type=str, default='cpu',
                        help='Device to run model on cpu or cuda.')
    args = par.parse_args()

    device = args.device

    inp_dets = args.detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    action_model = TSSTG()

    resize_fn = ResizePadding(inp_dets, inp_dets)
    

def run(val):
    flag=True
    fps_time = 0
    f = 0
    if type(val) is str and os.path.isfile(val):
        cam = CamLoader_Q(val, queue_size=1000, preprocess=preproc).start()
    else:
        cam = CamLoader(int(val) if val.isdigit() else val,
                        preprocess=preproc).start()
    print('Running. Press CTRL-C to exit.')
    time.sleep(0.1) #wait for serial to open
    if True:
        while cam.grabbed():
            f += 1
            frame = cam.getitem()
            image = frame.copy()

            detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

            tracker.predict()
            for track in tracker.tracks:
                det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
                detected = torch.cat([detected, det], dim=0) if detected is not None else det

            detections = []
            if detected is not None:
                poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

                detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                        np.concatenate((ps['keypoints'].numpy(),
                                                        ps['kp_score'].numpy()), axis=1),
                                        ps['kp_score'].mean().numpy()) for ps in poses]

                if args.show_detected:
                    for bb in detected[:, 0:5]:
                        frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

            tracker.update(detections)

            for i, track in enumerate(tracker.tracks):
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_tlbr().astype(int)
                center = track.get_center().astype(int)

                action = 'pending..'
                clr = (0, 255, 0)
                if len(track.keypoints_list) == 30:
                    pts = np.array(track.keypoints_list, dtype=np.float32)
                    out = action_model.predict(pts, frame.shape[:2])
                    action_name = action_model.class_names[out[0].argmax()]
                    action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                    if action_name == 'Fall Down':
                        clr = (255, 0, 0)
                    elif action_name == 'Lying Down':
                        flag=True
                        clr = (255, 200, 0)

                if track.time_since_update == 0:
                    if args.show_skeleton:
                        frame = draw_single(frame, track.keypoints_list[-1])
                    frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                    frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                        0.4, (255, 0, 0), 2)
                    frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                        0.4, clr, 1)

            frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
            frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            frame = frame[:, :, ::-1]
            fps_time = time.time()

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cam.stop() 
    cv2.destroyAllWindows()

def browse():
    file_path = filedialog.askopenfilename()
    if(file_path):
        run(file_path)

root = Tk()

root.geometry("500x300")

image = tkimg.open("b.jpg")
# Resize the image if desired
image = image.resize((1400, 700))
tk_image = ImageTk.PhotoImage(image)

label = Label(root, image=tk_image)
label.pack()

label = Label(root, text="Human Activity")
label.place(x=250, y=25, anchor="center")

btn = Button(root, text="Webcam", command=lambda: run('0'))
btn.place(x=250, y=150, anchor="center")

Button(root, text="Browse", command=browse).place(x=250, y=200, anchor="center")

mainloop()