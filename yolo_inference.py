import time
from ultralytics import YOLO

start_time = time.time()

#model = YOLO('models/yolov5_ball_best.pt')
model = YOLO('models/yolo11x.pt')

result = model.track('input_media/input_video.mp4', conf=0.2, save=True)
# print(result)
# print("------------------")
# print("------------------")
# print("Boxes:")
# for box in result[0].boxes:
#     print(box)
#     print("------------------")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")