from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from io import BytesIO
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage, FileSystemStorage
from .style_transfer import StyleTransfer
import numpy as np

def save_image_to_django_storage(image_tensor, output_filename):
    """
    Save the generated image tensor to the Django storage system.
    """
    image_tensor = image_tensor.squeeze().cpu().detach().numpy()
    image_tensor = np.moveaxis(image_tensor, 0, -1) * 255
    image = Image.fromarray(image_tensor.astype(np.uint8))
    
    image_bytes_io = BytesIO()
    image.save(image_bytes_io, format='JPEG')
    image_bytes_io.seek(0)

    file_name = default_storage.save(output_filename, ContentFile(image_bytes_io.read()))
    
    return file_name

@csrf_exempt
def style_transfer_view(request):
    if request.method == "POST":
        content_file = request.FILES['content_image']
        style_file = request.FILES['style_image']

        content_path = default_storage.save('content.jpg', content_file)
        style_path = default_storage.save('style.jpg', style_file)

        content_absolute_path = default_storage.path(content_path)
        style_absolute_path = default_storage.path(style_path)

        style_transfer = StyleTransfer(content_weight=1, style_weight=1000000, num_steps=10)
        content_img = style_transfer.preprocess_image(content_absolute_path)
        style_img = style_transfer.preprocess_image(style_absolute_path)

        output = style_transfer.run_style_transfer(content_img, style_img)

        output_filename = 'styled_image.jpg'

        output_path = save_image_to_django_storage(output, output_filename)

        response = JsonResponse({'message': 'Style Transfer Successful', 'image_url': f'/media/{output_path}'})
        return response
