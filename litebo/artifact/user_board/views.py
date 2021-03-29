import os
import sys
import bson
import pprint
import datetime
import numpy as np
from bson import ObjectId
from litebo.optimizer import _optimizers
from litebo.artifact.data_manipulation.db_object import User, Task, Runhistory
from litebo.utils.config_space.space_utils import get_config_space_from_dict

from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse


def index(request):
    # TODO: return a proper html for GET that enable the user to post 'username' and 'password'
    # although now we haven't applied password logic...
    if request.method == 'GET':
        return HttpResponse('This page should be login.')
    elif request.method == 'POST':
        if request.POST:
            # Parse request
            owner = request.POST.get('username')
            config_space_json = request.POST.get('password')
            return HttpResponseRedirect(reverse('user_board:show_task', args=(owner,)))
        else:
            return HttpResponse('[bo_advice/views.py] empty post data')


def show_task(request, owner: str):
    # TODO: add a proper ./templates/user_board/show_task.html & use the commented return render statement instead
    if request.method == 'GET':
        task = Task()
        task_list = [x for x in task.find_one({'owner': owner})]
        return HttpResponse(str(task_list))
        # return render(request, 'user_board/show_task.html', {'owner': owner, 'task_list': task_list})


def detail(request, task_id: str):
    # TODO: add a proper ./templates/user_board/detail.html & use the commented return render statement instead
    if request.method == 'GET':
        runhistory = Runhistory()
        runhistory_list = [x for x in runhistory.find_many({'task_id': task_id})]
        return HttpResponse(str(runhistory_list))
        # return render(request, 'user_board/detail.html', {'task_id': task_id, 'runhistory_list': runhistory_list})
