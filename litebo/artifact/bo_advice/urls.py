from django.urls import path

from litebo.artifact.bo_advice import views

urlpatterns = [
    # ex: /bo_advice/task_register/
    path('task_register/', views.task_register, name='task_register'),
    # ex: /bo_advice/get_suggestion/
    path('get_suggestion/', views.get_suggestion, name='get_suggestion'),
    # ex: /bo_advice/update_observation/
    path('update_observation/', views.update_observation, name='update_observation'),
    # ex: /bo_advice/task_done/
    path('task_done/', views.task_done, name='task_done'),

    path('test_upload/', views.test_upload, name='test_upload'),
    path('test_download/', views.test_download, name='test_download'),
]

