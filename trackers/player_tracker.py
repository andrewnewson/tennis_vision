from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append("../")
from utils import *

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) # load yolo model from path

    def select_players_only(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0] # get player detections from the first frame
        chosen_player = self.choose_player(court_keypoints, player_detections_first_frame) # choose a player based on court keypoints
        filtered_player_detections = []
        for player_dict in player_detections: # iterate through each player detection
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player} # filter out players that are not chosen
            filtered_player_detections.append(filtered_player_dict) # append the filtered player detections to the list

        return filtered_player_detections

    def choose_player(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items(): # iterate through each player detection
            player_centre = get_bbox_centre(bbox) # get the centre of the player bounding box

            min_distance = float("inf")
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1]) # get the court keypoint
                distance = measure_distance(player_centre, court_keypoint) # calculate the distance between the player and the court keypoint
                if distance < min_distance: # check if the distance is less than the minimum distance
                    min_distance = distance # update the minimum distance
            distances.append((track_id, min_distance)) # append the track id and distance to the list

        distances.sort(key=lambda x: x[1]) # sort the distances in ascending order
        chosen_players = [distances[0][0], distances[1][0]] # choose the two players with the smallest distances (would need changing for doubles)

        return chosen_players
            
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