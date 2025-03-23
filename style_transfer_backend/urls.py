from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from styletransfer import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/style-transfer/', views.style_transfer_view, name='style_transfer'),
    path('api/get-all/', views.get_all_collections_view, name='get_all'),
    path('api/update-collection/', views.update_collection_view, name='update_collection'),
    path('api/get-collection/<uuid:collection_id>/', views.get_collection_view, name='get_collection'),
    path('api/delete-collection/<uuid:collection_id>/', views.delete_collection_view, name='delete_collection'),
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
