from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.utils.timezone import now
import os
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

from .style_transfer import StyleTransfer

@csrf_exempt
def style_transfer_view(request):
    if request.method == 'POST' and request.FILES.get('content_image') and request.FILES.get('style_image'):
        content_image = request.FILES['content_image']
        style_image = request.FILES['style_image']

        folder_name = now().strftime("%Y-%m-%d_%H-%M-%S")
        media_path = os.path.join(settings.MEDIA_ROOT, folder_name)
        os.makedirs(media_path, exist_ok=True)

        content_image_path = os.path.join(media_path, 'content_image.png')
        style_image_path = os.path.join(media_path, 'style_image.png')
        
        fs = FileSystemStorage(location=media_path)
        fs.save('content_image.png', content_image)
        fs.save('style_image.png', style_image)

        style_transfer = StyleTransfer(content_weight=1e3, style_weight=1e2)

        result_image = style_transfer.transfer_style(content_image_path, style_image_path, num_iterations=50, learning_rate=1e-2)

        result_image_path = os.path.join(media_path, 'style_transferred_image.png')
        result_image.save(result_image_path)

        return JsonResponse({
            'content_image': content_image_path,
            'style_image': style_image_path,
            'style_transferred_image': result_image_path
        })

    return JsonResponse({'error': 'Please upload both content and style images.'}, status=400)
