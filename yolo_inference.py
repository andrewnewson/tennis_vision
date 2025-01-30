from ultralytics import YOLO

#model = YOLO('models/yolov5_ball_best.pt')
model = YOLO('models/yolov8x.pt')

result = model.track('input_media/input_video.mp4', conf=0.2, save=True)
# print(result)
# print("------------------")
# print("------------------")
# print("Boxes:")
# for box in result[0].boxes:
#     print(box)
#     print("------------------")