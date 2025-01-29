import cv2
import sys
sys.path.append("../")
import constants
from utils import *
import numpy as np

class MiniCourtGraphic():
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.border = 50
        self.padding_court = 20

        self.set_background_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_keypoints()
        self.set_court_lines()

    def convert_metres_to_pixels(self, metres_distance):
        return convert_metres_distance_to_pixels(metres_distance, constants.DOUBLE_COURT_WIDTH, self.court_drawing_width)

    def set_court_drawing_keypoints(self):
        drawing_keypoints = [0] * 28 # x, y for each of 14 keypoints

        drawing_keypoints[0], drawing_keypoints[1] = int(self.court_start_x), int(self.court_start_y)
        drawing_keypoints[2], drawing_keypoints[3] = int(self.court_end_x), int(self.court_start_y)
        
        drawing_keypoints[4] = int(self.court_start_x)
        drawing_keypoints[5] = self.court_start_y + self.convert_metres_to_pixels(constants.HALF_COURT_LENGTH*2)
        
        drawing_keypoints[6] = drawing_keypoints[0] + self.court_drawing_width
        drawing_keypoints[7] = drawing_keypoints[5]
        
        drawing_keypoints[8] = drawing_keypoints[0] + self.convert_metres_to_pixels(constants.TRAM_LINE_WIDTH)
        drawing_keypoints[9] = drawing_keypoints[1]

        drawing_keypoints[10] = drawing_keypoints[4] + self.convert_metres_to_pixels(constants.TRAM_LINE_WIDTH)
        drawing_keypoints[11] = drawing_keypoints[5]

        drawing_keypoints[12] = drawing_keypoints[2] - self.convert_metres_to_pixels(constants.TRAM_LINE_WIDTH)
        drawing_keypoints[13] = drawing_keypoints[3]

        drawing_keypoints[14] = drawing_keypoints[6] - self.convert_metres_to_pixels(constants.TRAM_LINE_WIDTH)
        drawing_keypoints[15] = drawing_keypoints[7]

        drawing_keypoints[16] = drawing_keypoints[8]
        drawing_keypoints[17] = drawing_keypoints[9] + self.convert_metres_to_pixels(constants.SERVICE_BOX_TO_BASELINE_LENGTH)

        drawing_keypoints[18] = drawing_keypoints[16] + self.convert_metres_to_pixels(constants.SINGLE_COURT_WIDTH)
        drawing_keypoints[19] = drawing_keypoints[17]

        drawing_keypoints[20] = drawing_keypoints[10]
        drawing_keypoints[21] = drawing_keypoints[11] - self.convert_metres_to_pixels(constants.SERVICE_BOX_TO_BASELINE_LENGTH)

        drawing_keypoints[22] = drawing_keypoints[20] + self.convert_metres_to_pixels(constants.SINGLE_COURT_WIDTH)
        drawing_keypoints[23] = drawing_keypoints[21]

        drawing_keypoints[24] = int((drawing_keypoints[16] + drawing_keypoints[18]) // 2)
        drawing_keypoints[25] = drawing_keypoints[17]

        drawing_keypoints[26] = int((drawing_keypoints[20] + drawing_keypoints[22]) // 2)
        drawing_keypoints[27] = drawing_keypoints[21]

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