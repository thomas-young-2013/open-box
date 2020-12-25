import os
import json
import pickle
import numpy as np
from django.http import HttpResponse
from django.core.handlers.wsgi import WSGIRequest
from ConfigSpace.read_and_write import json as config_json
from litebo.config_space import Configuration
from litebo.config_space.util import convert_configurations_to_array


"""
THIS FILE IS ONLY A FRAMEWORK !!!

the main routine is
1. Client come -> (POST)task_register -> 
2. (POST)get_suggestion -> (Clent run locally) -> (POST)update_observation
3. cycle routine 2 till the client is satisfied
4. ->(POST)task_done
"""

# naive implementation of id
def get_pickle_name_from_id(client_id):
    """
    not fully implemented
    should convert the client_id to a unique static file name
    might change to a database

    Parameters
    ----------
    client_id : an identifier to choose the correlated SMBO object's pickle

    Returns
    -------
    filepath : <string> filepath of the pickle file

    """
    filepath = os.path.join('litebo/artifact/bo_advice/static_storage', str(client_id) + '.config_space.pkl')
    return filepath


# POST API
def task_register(request):
    """
    receive a user's  (id, config_space) and dump it into a pickle file

    Parameters
    ----------
    request: a django.core.handlers.wsgi.WSGIRequest object with following keys:
        'id' : a unique user id
        'config_space_array' : json string of a config_space created by ConfigSpace.read_and_write.json.write()

    Returns
    -------
    (readable information) string in HttpResponse form

    """
    if request.method == 'POST':
        if request.POST:
            user_id = request.POST.get('id')
            pickle_name = get_pickle_name_from_id(user_id)
            config_space_array = request.POST.get('config_space_array')
            config_space = config_json.read(config_space_array)
            with open(pickle_name, 'wb') as f:
                pickle.dump(config_space, f, pickle.HIGHEST_PROTOCOL)
            return HttpResponse('[bo_advice/views.py] SUCCESS')
        else:
            return HttpResponse('[bo_advice/views.py] empty post data')
    else:
        return HttpResponse('[bo_advice/views.py] should be a POST request')


#
def get_suggestion(request):
    """
    take a (user_id) and reply a suggestion

    Parameters
    ----------
    request: a dict

    Returns
    -------
    SUCCESS : a new encoded (configuration)
    FAILED : a readable information string in HttpResponse form
    """

    if request.method == 'POST':
        if request.POST:
            user_id = request.POST.get('id')
            pickle_name = get_pickle_name_from_id(user_id)
            with open(pickle_name, 'rb') as f:
                config_space = pickle.load(f)

            """
            这里应该引入计算新config的逻辑，这里先为了实现交互， 返回的是一个随机的的config
            为了能用Http返回 这里采用了json格式，需要接受方把json格式转化成np.array再转化成config_space
            """
            sample = config_space.sample_configuration()
            config_vector = convert_configurations_to_array([sample])[0].tolist()
            print('---------------------')
            print(type(config_vector))
            print(config_vector)
            print('---------------------')
            res = json.JSONEncoder().encode(config_vector)

            return HttpResponse(res)
        else:
            return HttpResponse('[bo_advice/views.py] error6')
    else:
        return HttpResponse('[bo_advice/views.py] error7')


# POST API
# load the pickled SMBO
def update_observation(request):
    """
    take a user_id, take corresponding logging file, append a line in it
    之后可能会改为内存数据库，还在理解阶段，用中文标明这里是下一个迷茫点

    Parameters
    ----------
    request : a json with the following keys


    Returns
    -------
    a readable information string in HttpResponse form
    """
    if request.method == 'POST':
        if request.POST:
            user_id = request.POST.get('user_id')
            config_js = request.POST.get('config_json')
            config_list = json.JSONDecoder().decode(config_js)
            print(config_list)
            config_vector = np.array(config_list)
            perf = float(request.POST.get('perf'))

            pickle_name = get_pickle_name_from_id(user_id)
            with open(pickle_name, 'rb') as f:
                config_space = pickle.load(f)
            config = Configuration(config_space, vector=config_vector)

            dic = {'user_id': user_id, 'config': config, 'perf': perf}
            print('--------------')
            print(dic)
            print('--------------')
            return HttpResponse('[bo_advice/views.py] update SUCCESS')
        else:
            return HttpResponse('[bo_advice/views.py] error3')
    else:
        return HttpResponse('[bo_advice/views.py] error4')


# POST API for test
def task_done(request):
    """

    Parameters
    ----------
    request: a dict

    Returns
    -------
    a readable information string in HttpResponse form
    """
    if request.method == 'POST':
        if request.POST:
            user_id = request.POST.get('user_id')
            pickle_name = get_pickle_name_from_id(user_id)
            os.remove(pickle_name)
            return HttpResponse('[bo_advice/views.py] remove SUCCESS')
        else:
            return HttpResponse('[bo_advice/views.py] empty post data')
    else:
        return HttpResponse('[bo_advice/views.py] should be a POST request')


# POST API for test
def test_upload(request: WSGIRequest) -> HttpResponse:
    print('---------------------------------------')
    print(request)
    print(type(request))
    print('---------------------------------------')
    assert request.method == 'POST'
    id = request.POST.get('id')
    print(id)
    config = request.POST.getlist('config')
    print(config)
    filepath = get_pickle_name_from_id(id)
    print(filepath)
    with open(filepath, 'wb') as f:
        savedata = {'id': id, 'config': config}
        pickle.dump(savedata, f, pickle.HIGHEST_PROTOCOL)
        print('dump finished')
        return HttpResponse('upload_SUCCESS!')
    return HttpResponse('upload_FAIL')


def test_download(request):
    assert request.method == 'POST'
    id = request.POST.get('id', 0)
    filepath = get_pickle_name_from_id(id)
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(type(data))
    print(data)
    return HttpResponse(data)
