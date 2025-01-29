# Function to get bounding box centre
def get_bbox_centre(bbox):
    x1, y1, x2, y2 = bbox
    centre_x = int((x1 + x2) / 2) # calculate the x-coordinate of the centre of the bounding box
    centre_y = int((y1 + y2) / 2) # calculate the y-coordinate of the centre of the bounding box

    return (centre_x, centre_y)

# Function to measure distance between two points
def measure_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5 # calculate Euclidean distance between two points

# Function to get foot position of player
def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2) # calculate the foot position of the player

# Function to get closest keypoint index to a point
def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    closest_distance = float('inf')
    closest_keypoint_index = keypoint_indices[0]
    for idx in keypoint_indices:
        keypoint = keypoints[idx*2], keypoints[idx*2+1]
        distance = abs(point[1] - keypoint[1])

        if distance < closest_distance:
            closest_distance = distance
            closest_keypoint_index = idx

    return closest_keypoint_index

def get_height_of_bounding_box(bbox):
    x1, y1, x2, y2 = bbox
    return y2 - y1

def measure_xy_distance(p1, p2):
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])