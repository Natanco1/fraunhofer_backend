from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from .style_transfer import StyleTransfer

def save_image_to_django_storage(image_tensor, output_filename):
    if isinstance(image_tensor, tf.Tensor):
        image_tensor = image_tensor.numpy() * 255
    if isinstance(image_tensor, Image.Image):
        image = image_tensor
    else:
        image = Image.fromarray(image_tensor.astype(np.uint8).squeeze())
    
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

        style_transfer = StyleTransfer(img_size=400, vgg_weights_path='/home/nata/projects/personal/fraunhofer/fraunhofer_backend/styletransfer/models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
        generated_image = style_transfer.train(content_absolute_path, style_absolute_path, epochs=30)

        output_filename = 'styled_image.jpg'
        output_path = save_image_to_django_storage(style_transfer.tensor_to_image(generated_image), output_filename)

        response = JsonResponse({'message': 'Style Transfer Successful', 'image_url': f'/media/{output_path}'})
        return response
