def convert_pixel_distance_to_metres(pixel_distance, reference_height_in_metres, reference_height_in_pixels):
    return (pixel_distance * reference_height_in_metres) / reference_height_in_pixels # calculate the distance in metres

def convert_metres_distance_to_pixels(metres_distance, reference_height_in_metres, reference_height_in_pixels):
    return (metres_distance * reference_height_in_pixels) / reference_height_in_metres # calculate the distance in pixels