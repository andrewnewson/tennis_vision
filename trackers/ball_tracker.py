from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) # load yolo model from path

    def interpolate_ball_position(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions] # get the ball positions from the ball detections
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"]) # create a dataframe from the ball positions
        df_ball_positions = df_ball_positions.interpolate() # interpolate the missing ball positions
        df_ball_positions = df_ball_positions.bfill() # backfill the missing ball positions at beginning of the video

        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()] # convert the ball positions back to the original format

        return ball_positions

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames: # iterate through each video frame
            ball_dict = self.detect_frame(frame) # detect ball in the frame
            ball_detections.append(ball_dict) # append the ball detections to the list

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f)

        return ball_detections
    
    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0] # run object detection on the frame

        ball_dict = {}
        for box in results.boxes: # iterate through each detected box
            result = box.xyxy.tolist()[0] # get the bounding box coordinates
            ball_dict[1] = result # add the bounding box coordinates to the ball dictionary

        return ball_dict
    
    def draw_bounding_boxes(self, video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections): # iterate through each frame and ball detection
            for track_id, bbox in ball_dict.items(): # iterate through each ball detection
                x1, y1, x2, y2 = bbox # get the bounding box coordinates
                cv2.putText(frame, f"Ball ID {track_id}", (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2) # add ball ID text to the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2) # draw bounding box around ball
            output_video_frames.append(frame) # append the frame with bounding boxes to the output list

        return output_video_frames