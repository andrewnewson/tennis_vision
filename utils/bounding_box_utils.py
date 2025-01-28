# Function to get bounding box centre
def get_bbox_centre(bbox):
    x1, y1, x2, y2 = bbox
    centre_x = int((x1 + x2) / 2) # calculate the x-coordinate of the centre of the bounding box
    centre_y = int((y1 + y2) / 2) # calculate the y-coordinate of the centre of the bounding box

    return (centre_x, centre_y)

# Function to measure distance between two points
def measure_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5 # calculate Euclidean distance between two points