from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage, FileSystemStorage
from .style_transfer import StyleTransfer
import os

@csrf_exempt
def style_transfer_view(request):
    if request.method == "POST":
        # Get the uploaded images
        content_file = request.FILES['content_image']
        style_file = request.FILES['style_image']

        # Save the content and style images in the media directory
        content_path = default_storage.save('content.jpg', content_file)
        style_path = default_storage.save('style.jpg', style_file)

        # Get the absolute path of the saved images
        content_absolute_path = default_storage.path(content_path)  # Absolute path to content image
        style_absolute_path = default_storage.path(style_path)  # Absolute path to style image

        # Create an instance of the StyleTransfer class
        style_transfer = StyleTransfer(content_weight=1, style_weight=1000000, num_steps=500)

        # Preprocess the images (pass absolute paths)
        content_img = style_transfer.preprocess_image(content_absolute_path)
        style_img = style_transfer.preprocess_image(style_absolute_path)

        # Run the style transfer model
        output = style_transfer.run_style_transfer(content_img, style_img)

        # Define the output image path (in the media folder)
        fs = FileSystemStorage()
        output_filename = fs.save('styled_image.jpg', output)  # Saving output image
        output_url = fs.url(output_filename)  # Get the URL of the saved image

        # Respond with the URL of the generated image
        response = JsonResponse({
            'message': 'Style Transfer Successful',
            'image_url': output_url
        })
        return response
