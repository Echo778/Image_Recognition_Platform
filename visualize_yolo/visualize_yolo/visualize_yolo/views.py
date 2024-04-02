from django.shortcuts import render
import os
from django.http import JsonResponse
import io
import base64
from run_proj import run_yolo
from PIL import Image


def test_view(request):
    template_path = os.path.join('visualize_yolo', 'test.html')
    return render(request, template_path, {})

def home_view(request):
    template_path = os.path.join('visualize_yolo', 'home.html')
    return render(request, template_path, {})


def display_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']

        # Read the content of the image file
        image_data = image_file.read()

        display_image_base64 = base64.b64encode(image_data).decode('utf-8')


        # return JsonResponse({'processed_image_data': processed_image_base64})
        template_path = os.path.join('visualize_yolo', 'home.html')
        return render(request, template_path,
                      {'display_image_data': display_image_base64})
    else:
        return JsonResponse({'error': 'image could not be uploaded'}, status=400)

def process_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']

        # Process the image (example: convert to grayscale)
        image = run_yolo(image_file)
        # image = Image.open(image_file)
        processed_image = Image.fromarray(image)

        # Save the processed image to a temporary file
        with io.BytesIO() as output:
            processed_image.save(output, format='PNG')
            processed_image_data = output.getvalue()

        processed_image_base64 = base64.b64encode(processed_image_data).decode('utf-8')


        # return JsonResponse({'processed_image_data': processed_image_base64})
        template_path = os.path.join('visualize_yolo', 'process-image.html')
        return render(request, template_path,
                      {'processed_image_data': processed_image_base64})
    else:
        return JsonResponse({'error': 'No image provided'}, status=400)