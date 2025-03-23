from django.http import JsonResponse
from django.utils.timezone import now
import os
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
import base64
from io import BytesIO
from PIL import Image
from .style_transfer import StyleTransfer
import json
import logging

logger = logging.getLogger(__name__)

@csrf_exempt
def style_transfer_view(request):
    if request.method == 'POST':
        logger.debug(f"Received body: {request.body.decode('utf-8')}")
        
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data.'}, status=400)

        content_image_base64 = data.get('content_image')
        style_image_base64 = data.get('style_image')

        if not content_image_base64 or not style_image_base64:
            return JsonResponse({'error': 'Please upload both content and style images.'}, status=400)

        try:
            content_image_data = base64.b64decode(content_image_base64)
            style_image_data = base64.b64decode(style_image_base64)

            content_image = Image.open(BytesIO(content_image_data))
            style_image = Image.open(BytesIO(style_image_data))
        except Exception as e:
            logger.error(f"Error decoding images: {e}")
            return JsonResponse({'error': 'Failed to decode image data.'}, status=400)

        folder_name = now().strftime("%Y-%m-%d_%H-%M-%S")
        media_path = os.path.join(settings.MEDIA_ROOT, folder_name)
        os.makedirs(media_path, exist_ok=True)

        content_image_path = os.path.join(media_path, 'content_image.png')
        style_image_path = os.path.join(media_path, 'style_image.png')

        content_image.save(content_image_path)
        style_image.save(style_image_path)

        style_transfer = StyleTransfer(content_image_path, style_image_path)
        stylized_image = style_transfer.transfer_style()

        result_image_path = os.path.join(media_path, 'style_transferred_image.png')
        stylized_image.save(result_image_path)

        with open(result_image_path, "rb") as f:
            result_image_base64 = base64.b64encode(f.read()).decode('utf-8')

        return JsonResponse({
            'style_transferred_image': result_image_base64
        })

    return JsonResponse({'error': 'Invalid request method.'}, status=400)
