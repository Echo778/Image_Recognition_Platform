import os

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
#import cv2
import matplotlib.rcsetup as rcsetup
from utils_functions import calculate_iou, non_max_suppression

print(rcsetup.all_backends)
matplotlib.use("TkAgg")
cwd = os.getcwd()
print(cwd)


# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(640),  # Resize image to model input size
    transforms.CenterCrop(640),  # Center crop to maintain aspect ratio
    transforms.ToTensor(),  # Convert image to tensor
])

# Load and preprocess image
def run_yolo(image_path):
    #image_path = 'cat.png'
    original_image = Image.open(image_path).convert('RGB')
    original_image_np = np.asarray(original_image)
    print(np.shape(original_image_np))
    plt.imshow(original_image_np)
    plt.show()
    plt.imsave(f'original_img.png', original_image_np)

    image_tensor = transform(original_image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    print(image_tensor.shape)


    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Convert normalized image values back to the range [0, 255] for uint8 datatype
    image_np = (image_np * 255).astype(np.uint8)


    image_np_ori = Image.fromarray(image_np)
    print(np.shape(image_np))

    # Perform inference
    with torch.no_grad():
        detections = model(image_tensor)

    print('detections: \n')
    print(detections)

    # Filter detections with confidence > %
    conf_threshold = 0.5



    results = detections[0][detections[0][:, 4] > conf_threshold]


    print(results)

    # Convert image to numpy array
    # image_np = transforms.ToPILImage()(image_tensor.squeeze(0).cpu())

    # Print results
    print("Detected Objects:")
    for obj in results:
        print(f"Bounding Box: {obj[:4].cpu().numpy()}, Confidence: {obj[4].item()}")


    H, W = image_tensor.shape[:2]

    # Draw bounding boxes on the original image
    draw = ImageDraw.Draw(image_np_ori)

    bbox_arr = []
    class_scores_arr = []
    predicted_class_index_arr = []
    predicted_class_confidence_arr = []

    # extract bbox and confidence into array
    for i, obj in enumerate(results):
        bbox = obj[:4].cpu().numpy()  # Extract bounding box coordinates
        (x, y) = (bbox[0], bbox[1])
        (w, h) = (bbox[2], bbox[3])
        # x1, y1, x2, y2
        bbox_vertices = [x-w//2, y-h//2, x + w//2, y + h//2]
        bbox_arr.append(bbox_vertices)

        class_scores = results[i, 5:].cpu().numpy()
        class_scores_arr.append(class_scores)

        predicted_class_index = np.argmax(class_scores)
        predicted_class_index_arr.append(predicted_class_index)

        predicted_class_confidence = class_scores[predicted_class_index]
        predicted_class_confidence_arr.append(predicted_class_confidence)

        #confidence = obj[4].item()  # Extract confidence score
        label = int(obj[5])  # Extract predicted class index
        category = model.names[label]
        print(category)


    print('selected bbox before NMS: ', bbox_arr)
    print('length for bbox before NMS', len(bbox_arr))


    iou_threshold = 0.01

    # NMS supression:
    selected_indices = non_max_suppression(bbox_arr, predicted_class_confidence_arr, iou_threshold)

    print('selected indx: ', selected_indices)

    for idx in selected_indices:
        box = bbox_arr[idx]
        confidence = predicted_class_confidence_arr[idx]


        # Extract class scores and find the class with the highest score
        predicted_class_index = predicted_class_index_arr[idx]
        predicted_class_name = model.names[predicted_class_index_arr[idx]]

        print('predicted_Calss index: ', predicted_class_index)
        print('predicted_Calss name: ', predicted_class_name)

        draw.rectangle([(bbox_arr[idx][0], bbox_arr[idx][1]), (bbox_arr[idx][2], bbox_arr[idx][3])], outline="red", width=1)
        draw.text((bbox_arr[idx][0], bbox_arr[idx][1] - 10), f"{predicted_class_name}: {confidence:.2f}", fill="blue")

        print((bbox[0], bbox[1]), (bbox[2], bbox[3]))
    print(np.shape(image_np_ori))


    # Display original image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image_np_ori)
    plt.show()


    image_np_ori = np.asarray(image_np_ori)
    plt.imshow(original_image_np)
    plt.show()
    plt.imsave(f'img_with_bb.png', image_np_ori)
    return image_np_ori

