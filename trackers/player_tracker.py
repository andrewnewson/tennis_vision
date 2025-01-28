from ultralytics import YOLO
import cv2
import pickle

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) # load yolo model from path

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames: # iterate through each video frame
            player_dict = self.detect_frame(frame) # detect players in the frame
            player_detections.append(player_dict) # append the player detections to the list

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(player_detections, f)

        return player_detections
    
    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0] # run object detection on the frame
        id_name_dict = results.names # get the class names from the model

        player_dict = {}
        for box in results.boxes: # iterate through each detected box
            track_id = int(box.id.tolist()[0]) # get the track id of the object
            result = box.xyxy.tolist()[0] # get the bounding box coordinates
            object_cls_id = box.cls.tolist()[0] # get the class id of the object
            object_cls_name = id_name_dict[object_cls_id] # get the class name of the object
            if object_cls_name == "person": # check if the detected object is a person
                player_dict[track_id] = result # add the bounding box coordinates to the player dictionary

        return player_dict
    
    def draw_bounding_boxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections): # iterate through each frame and player detection
            for track_id, bbox in player_dict.items(): # iterate through each player detection
                x1, y1, x2, y2 = bbox # get the bounding box coordinates
                cv2.putText(frame, f"Player ID {track_id}", (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) # add player ID text to the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2) # draw bounding box around player
            output_video_frames.append(frame) # append the frame with bounding boxes to the output list

        return output_video_frames