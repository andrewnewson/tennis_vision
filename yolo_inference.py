from ultralytics import YOLO

model = YOLO('yolov8x')

result = model.predict('input_media/input_video.mp4', save=True)
print(result)
print("------------------")
print("------------------")
print("Boxes:")
for box in result[0].boxes:
    print(box)
    print("------------------")