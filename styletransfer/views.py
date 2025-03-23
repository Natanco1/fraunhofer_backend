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
import uuid

from styletransfer.query import collection as col

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
        collection_name = data.get('name')

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

        request_id = str(uuid.uuid4())

        media_path = os.path.join(settings.MEDIA_ROOT)
        os.makedirs(media_path, exist_ok=True)

        content_image_folder = os.path.join(media_path, 'content')
        style_image_folder = os.path.join(media_path, 'style')
        generated_image_folder = os.path.join(media_path, 'generated')

        os.makedirs(content_image_folder, exist_ok=True)
        os.makedirs(style_image_folder, exist_ok=True)
        os.makedirs(generated_image_folder, exist_ok=True)

        content_image_path = os.path.join(content_image_folder, f'{request_id}.png')
        style_image_path = os.path.join(style_image_folder, f'{request_id}.png')

        content_image.save(content_image_path)
        style_image.save(style_image_path)

        style_transfer = StyleTransfer(content_image_path, style_image_path)
        stylized_image = style_transfer.transfer_style()

        result_image_path = os.path.join(generated_image_folder, f'{request_id}.png')
        stylized_image.save(result_image_path)

        with open(result_image_path, "rb") as f:
            result_image_base64 = base64.b64encode(f.read()).decode('utf-8')

        try:
            col.insert_collection_record(request_id, collection_name)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

        return JsonResponse({
            'style_transferred_image': result_image_base64,
            'content_image_url': os.path.join(settings.MEDIA_URL, 'content', f'{request_id}.png'),
            'style_image_url': os.path.join(settings.MEDIA_URL, 'style', f'{request_id}.png'),
            'generated_image_url': os.path.join(settings.MEDIA_URL, 'generated', f'{request_id}.png')
        })

    return JsonResponse({'error': 'Invalid request method.'}, status=400)


def get_all_collections_view(request):
    """Handle GET requests to fetch all collection records."""
    try:
        collections = col.get_all_collections()

        collections_data = []
        for collection in collections:
            collection_id, name, created_at, updated_at = collection
            collections_data.append({
                'id': collection_id,
                'name': name,
                'createdAt': created_at,
                'updatedAt': updated_at
            })

        return JsonResponse({'collections': collections_data}, status=200)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    

@csrf_exempt
def update_collection_view(request):
    """Handle PUT requests to update collection name by id."""
    if request.method == 'PUT':
        try:
            data = json.loads(request.body)
            collection_id = data.get('id')
            new_name = data.get('name')

            if not collection_id or not new_name:
                return JsonResponse({'error': 'Collection ID and new name are required.'}, status=400)

            col.update_collection_name(collection_id, new_name)

            return JsonResponse({'message': 'Collection updated successfully.'}, status=200)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=400)


def get_collection_view(request, collection_id):
    """Handle GET requests to fetch a single collection by its ID."""
    try:
        collection = col.get_collection_record_by_id(collection_id)
        
        if collection:
            collection_data = {
                'id': collection[0],
                'name': collection[1],
                'createdAt': collection[2],
                'updatedAt': collection[3]
            }
            return JsonResponse({'collection': collection_data}, status=200)
        else:
            return JsonResponse({'error': 'Collection not found'}, status=404)

    except Exception as e:
        logger.error(f"Error fetching collection with ID {collection_id}: {e}")
        return JsonResponse({'error': str(e)}, status=500)
