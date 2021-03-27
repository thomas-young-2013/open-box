from django.urls import path

from . import views

app_name = 'user_board'
urlpatterns = [
    # /user_board/
    path('', views.index, name='index'),
    # /user_board/605c76e8db226d5d47a5b409/show_task/
    path('<str:owner>/show_task/', views.show_task, name='show_task'),
    # /user_board/abc/detail/
    path('<str:task_id>/detail/', views.detail, name='detail'),
]