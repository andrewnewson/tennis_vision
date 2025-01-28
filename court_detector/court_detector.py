import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

class CourtDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(weights=None) # load resnet50 model
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) # 14 keypoints with x and y coordinates
        self.model.load_state_dict(torch.load(model_path, map_location="cpu")) # load model weights

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert image to RGB
        image_tensor = self.transforms(img_rgb).unsqueeze(0) # apply transforms and add batch dimension (puts single image into list)

        with torch.no_grad():
            output = self.model(image_tensor) # forward pass

        keypoints = output.squeeze().cpu().numpy() # remove batch dimension and convert to numpy array
        original_h, original_w = image.shape[:2] # get original image height and width

        keypoints[::2] *= original_w/224 # scale x coordinates
        keypoints[1::2] *= original_h/224 # scale y coordinates

        return keypoints
    
    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2): # iterate through x coordinates
            x = int(keypoints[i]) # get x coordinate
            y = int(keypoints[i+1]) # get y coordinate

            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 150), 2) # add keypoint ID text to the frame
            cv2.circle(image, (x, y), 5, (0, 150, 150), -1) # draw circle at keypoint location

        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames: # iterate through each frame
            frame = self.draw_keypoints(frame, keypoints) # draw keypoints on frame
            output_video_frames.append(frame) # append frame to output list

        return output_video_frames