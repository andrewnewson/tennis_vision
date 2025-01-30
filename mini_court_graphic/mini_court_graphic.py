import cv2
import sys
sys.path.append("../")
import constants
from utils import *
import numpy as np

class MiniCourtGraphic():
    def __init__(self, frame):
        self.drawing_rectangle_width = 250 # width of the rectangle that the mini court will be drawn in
        self.drawing_rectangle_height = 500 # height of the rectangle that the mini court will be drawn in
        self.border = 50 # border around the rectangle
        self.padding_court = 20 # padding around the court

        self.set_background_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_keypoints()
        self.set_court_lines()

    def convert_metres_to_pixels(self, metres_distance):
        return convert_metres_distance_to_pixels(metres_distance, constants.DOUBLE_COURT_WIDTH, self.court_drawing_width)

    def set_court_drawing_keypoints(self):
        drawing_keypoints = [0] * 28 # x, y for each of 14 keypoints

        drawing_keypoints[0], drawing_keypoints[1] = int(self.court_start_x), int(self.court_start_y) # top left x, y
        drawing_keypoints[2], drawing_keypoints[3] = int(self.court_end_x), int(self.court_start_y) # top right x, y
        
        drawing_keypoints[4] = int(self.court_start_x) # bottom left x
        drawing_keypoints[5] = self.court_start_y + self.convert_metres_to_pixels(constants.HALF_COURT_LENGTH*2) # bottom left y
        
        drawing_keypoints[6] = drawing_keypoints[0] + self.court_drawing_width # bottom right x
        drawing_keypoints[7] = drawing_keypoints[5] # bottom right y
        
        drawing_keypoints[8] = drawing_keypoints[0] + self.convert_metres_to_pixels(constants.TRAM_LINE_WIDTH) # singles top left x
        drawing_keypoints[9] = drawing_keypoints[1] # singles top left y

        drawing_keypoints[10] = drawing_keypoints[4] + self.convert_metres_to_pixels(constants.TRAM_LINE_WIDTH) # singles bottom left x
        drawing_keypoints[11] = drawing_keypoints[5] # singles bottom left y

        drawing_keypoints[12] = drawing_keypoints[2] - self.convert_metres_to_pixels(constants.TRAM_LINE_WIDTH) # singles top right x
        drawing_keypoints[13] = drawing_keypoints[3] # singles top right y

        drawing_keypoints[14] = drawing_keypoints[6] - self.convert_metres_to_pixels(constants.TRAM_LINE_WIDTH) # singles bottom right x
        drawing_keypoints[15] = drawing_keypoints[7] # singles bottom right y

        drawing_keypoints[16] = drawing_keypoints[8] # service box top left x
        drawing_keypoints[17] = drawing_keypoints[9] + self.convert_metres_to_pixels(constants.SERVICE_BOX_TO_BASELINE_LENGTH) # service box top left y

        drawing_keypoints[18] = drawing_keypoints[16] + self.convert_metres_to_pixels(constants.SINGLE_COURT_WIDTH) # service box top right x
        drawing_keypoints[19] = drawing_keypoints[17] # service box top right y

        drawing_keypoints[20] = drawing_keypoints[10] # service box bottom left x
        drawing_keypoints[21] = drawing_keypoints[11] - self.convert_metres_to_pixels(constants.SERVICE_BOX_TO_BASELINE_LENGTH) # service box bottom left y

        drawing_keypoints[22] = drawing_keypoints[20] + self.convert_metres_to_pixels(constants.SINGLE_COURT_WIDTH) # service box bottom right x
        drawing_keypoints[23] = drawing_keypoints[21] # service box bottom right y

        drawing_keypoints[24] = int((drawing_keypoints[16] + drawing_keypoints[18]) // 2) # service box top middle x
        drawing_keypoints[25] = drawing_keypoints[17] # service box top middle y

        drawing_keypoints[26] = int((drawing_keypoints[20] + drawing_keypoints[22]) // 2) # service box bottom middle x
        drawing_keypoints[27] = drawing_keypoints[21] # service box bottom middle y

        self.court_drawing_keypoints = drawing_keypoints

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),

            (0, 1),
            (8, 9),
            (10, 11),
            (2, 3)
        ]

    def set_background_position(self, frame):
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.border
        self.end_y = self.border + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def draw_court(self, frame):
        for i in range(0, len(self.court_drawing_keypoints), 2):
            x = int(self.court_drawing_keypoints[i])
            y = int(self.court_drawing_keypoints[i+1])
            cv2.circle(frame, (x, y), 5, (0, 0, 0), -1)

        for line in self.lines: # draw lines
            start_point = (int(self.court_drawing_keypoints[line[0]*2]), int(self.court_drawing_keypoints[line[0]*2+1]))
            end_point = (int(self.court_drawing_keypoints[line[1]*2]), int(self.court_drawing_keypoints[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 255), 2)

        net_start_point = (self.court_drawing_keypoints[0], int((self.court_drawing_keypoints[1] + self.court_drawing_keypoints[5]) // 2))
        net_end_point = (self.court_drawing_keypoints[2], int((self.court_drawing_keypoints[1] + self.court_drawing_keypoints[5]) // 2))
        cv2.line(frame, net_start_point, net_end_point, (0, 0, 0), 2)

        return frame

    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), -1) # draw white rectangle

        out = frame.copy()
        alpha = 0.5 # transparency factor
        mask = shapes.astype(bool) # create a mask
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask] # overlay the white rectangle on the frame

        return out

    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        
        return output_frames
    
    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)
    
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    
    def get_mini_court_drawing_keypoints(self):
        return self.court_drawing_keypoints
    
    def get_mini_court_coordinates(self, object_position, closest_keypoint, closest_keypoint_index, player_height_pxl, player_height_metres):
        distance_from_keypoint_x_pxl, distance_from_keypoint_y_pxl = measure_xy_distance(object_position, closest_keypoint)
        
        distance_from_keypoint_x_metres = convert_pixel_distance_to_metres(distance_from_keypoint_x_pxl, player_height_metres, player_height_pxl) # convert the distance from pixels to metres
        distance_from_keypoint_y_metres = convert_pixel_distance_to_metres(distance_from_keypoint_y_pxl, player_height_metres, player_height_pxl) # convert the distance from pixels to metres

        mini_court_x_distance_pxl = self.convert_metres_to_pixels(distance_from_keypoint_x_metres) # convert the distance from metres to pixels
        mini_court_y_distance_pxl = self.convert_metres_to_pixels(distance_from_keypoint_y_metres) # convert the distance from metres to pixels

        closest_mini_court_keypoint = (self.court_drawing_keypoints[closest_keypoint_index*2], self.court_drawing_keypoints[closest_keypoint_index*2+1])

        mini_court_player_position = (closest_mini_court_keypoint[0] + mini_court_x_distance_pxl, closest_mini_court_keypoint[1] + mini_court_y_distance_pxl)

        return mini_court_player_position
    
    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_keypoints):
        player_heights = {
            1: constants.PLAYER_1_HEIGHT,
            2: constants.PLAYER_2_HEIGHT,
        }

        output_player_boxes = []
        output_ball_boxes = []

        for frame_number, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_number][1]
            ball_position = get_bbox_centre(ball_box)
            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: measure_distance(ball_position, get_bbox_centre(player_bbox[x])))

            output_player_bboxes_dict = {}

            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                closest_keypoint_index = get_closest_keypoint_index(foot_position, original_court_keypoints, [0,2,12,13])
                closest_keypoint = (original_court_keypoints[closest_keypoint_index*2], original_court_keypoints[closest_keypoint_index*2+1])

                frame_index_min = max(0, frame_number - 20)
                frame_index_max = min(len(player_boxes), frame_number + 50)
                bboxes_height_pxl = [get_height_of_bounding_box(player_boxes[i][player_id]) for i in range(frame_index_min, frame_index_max)]
                max_player_height_pxl = max(bboxes_height_pxl)

                mini_court_player_position = self.get_mini_court_coordinates(foot_position, closest_keypoint, closest_keypoint_index, max_player_height_pxl, player_heights[player_id])

                output_player_bboxes_dict[player_id] = mini_court_player_position

                if closest_player_id_to_ball == player_id:
                    closest_keypoint_index = get_closest_keypoint_index(ball_position, original_court_keypoints, [0,2,12,13])
                    closest_keypoint = (original_court_keypoints[closest_keypoint_index*2], original_court_keypoints[closest_keypoint_index*2+1])
                    mini_court_player_position = self.get_mini_court_coordinates(ball_position, closest_keypoint, closest_keypoint_index, max_player_height_pxl, player_heights[player_id])

                    output_ball_boxes.append({1: mini_court_player_position})

            output_player_boxes.append(output_player_bboxes_dict)

        return output_player_boxes, output_ball_boxes
    
    def draw_points_on_mini_court(self, frames, positions, colour=(0, 255, 0)):
        for frame_number, frame in enumerate(frames):
            for _, position in positions[frame_number].items():
                x, y = position
                x = int(x)
                y = int(y)
                cv2.circle(frame, (x, y), 5, colour, -1)

        return frames