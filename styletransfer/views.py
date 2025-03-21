from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.utils.timezone import now
import os
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from .style_transfer import StyleTransfer
import tensorflow as tf

@csrf_exempt
def style_transfer_view(request):
    if request.method == 'POST' and request.FILES.get('content_image') and request.FILES.get('style_image'):
        content_image = request.FILES['content_image']
        style_image = request.FILES['style_image']

        if content_image.size == 0 or style_image.size == 0:
            return JsonResponse({'error': 'One or both uploaded images are empty.'}, status=400)

        if 'image' not in content_image.content_type or 'image' not in style_image.content_type:
            return JsonResponse({'error': 'Please upload valid image files.'}, status=400)

        folder_name = now().strftime("%Y-%m-%d_%H-%M-%S")
        media_path = os.path.join(settings.MEDIA_ROOT, folder_name)
        os.makedirs(media_path, exist_ok=True)

        content_image_path = os.path.join(media_path, 'content_image.png')
        style_image_path = os.path.join(media_path, 'style_image.png')

        fs = FileSystemStorage(location=media_path)
        fs.save('content_image.png', content_image)
        fs.save('style_image.png', style_image)

        style_transfer = StyleTransfer(content_image_path, style_image_path)

        stylized_image = style_transfer.transfer_style()

        result_image_path = os.path.join(media_path, 'style_transferred_image.png')
        stylized_image.save(result_image_path)

        return JsonResponse({
            'content_image': content_image_path,
            'style_image': style_image_path,
            'style_transferred_image': result_image_path
        })

    return JsonResponse({'error': 'Please upload both content and style images.'}, status=400)
