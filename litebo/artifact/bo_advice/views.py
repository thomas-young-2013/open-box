import json
from django.http import HttpResponse

from litebo.config_space import json as config_json
from litebo.config_space import Configuration
from litebo.core.base import Observation


# Global mapping from task_id to config advisor
advisor_dict = {}


def task_register(request):
    """
    Receive a task's (task_id, config_space) and dump it into a pickle file.

    Parameters
    ----------
    request: a django.core.handlers.wsgi.WSGIRequest object with following keys:
        'id' : a unique task id
        'config_space_json' : json string of a config_space created by ConfigSpace.read_and_write.json.write()

    Returns
    -------
    (readable information) string in HttpResponse form

    """
    if request.method == 'POST':
        if request.POST:
            # Parse request
            task_id = request.POST.get('task_id')
            config_space_json = request.POST.get('config_space_json')
            config_space = config_json.read(config_space_json)

            num_constraints = int(request.POST.get('num_constraints', 0))
            num_objs = int(request.POST.get('num_objs', 1))
            task_info = {'num_constraints': num_constraints, 'num_objs': num_objs}
            options = json.loads(request.POST.get('options', '{}'))

            # Create advisor
            advisor_type = request.POST.get('advisor_type', 'default')
            if advisor_type == 'default':
                from litebo.core.generic_advisor import Advisor
                config_advisor = Advisor(config_space, task_info, task_id=task_id, **options)
            elif advisor_type == 'tpe':
                from litebo.core.tpe_advisor import TPE_Advisor
                config_advisor = TPE_Advisor(config_space)
            else:
                raise ValueError('Invalid advisor type!')

            # Save advisor in a global dict
            advisor_dict[task_id] = config_advisor

            return HttpResponse('[bo_advice/views.py] SUCCESS')
        else:
            return HttpResponse('[bo_advice/views.py] empty post data')
    else:
        return HttpResponse('[bo_advice/views.py] should be a POST request')


def get_suggestion(request):
    """
    Print the suggestion according to the specified task id.

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
            task_id = request.POST.get('task_id')
            config_advisor = advisor_dict[task_id]

            suggestion = config_advisor.get_suggestion()
            print('-'*21)
            print('Get suggestion')
            print(suggestion, '-'*21, sep='')
            res = json.JSONEncoder().encode(suggestion.get_dictionary())

            return HttpResponse(res)
        else:
            return HttpResponse('[bo_advice/views.py] error6')
    else:
        return HttpResponse('[bo_advice/views.py] error7')


def update_observation(request):
    """
    Update observation in config advisor.

    Parameters
    ----------
    request : a dict

    Returns
    -------
    a readable information string in HttpResponse form
    """
    if request.method == 'POST':
        if request.POST:
            task_id = request.POST.get('task_id')
            config_advisor = advisor_dict[task_id]
            config_dict = json.loads(request.POST.get('config'))
            config = Configuration(config_advisor.config_space, config_dict)
            trial_state = int(request.POST.get('trial_state'))
            constraints = json.loads(request.POST.get('constraints'))
            objs = json.loads(request.POST.get('objs'))

            observation = Observation(config, trial_state, constraints, objs)
            config_advisor.update_observation(observation)

            print('-'*21)
            print('Update observation')
            print(observation)
            print('-'*21)
            return HttpResponse('[bo_advice/views.py] update SUCCESS')
        else:
            return HttpResponse('[bo_advice/views.py] error3')
    else:
        return HttpResponse('[bo_advice/views.py] error4')


def get_result(request):
    """
    Get BO result and history.

    Parameters
    ----------
    request: a dict

    Returns
    -------
    BO incumbents, BO history
    """

    if request.method == 'POST':
        if request.POST:
            task_id = request.POST.get('task_id')
            config_advisor = advisor_dict[task_id]

            incumbents = config_advisor.history_container.incumbents
            incumbents = [(k.get_dictionary(), v) for k, v in incumbents]
            history = config_advisor.history_container.data
            history = [(k.get_dictionary(), v) for k, v in history.items()]
            print('-'*21)
            print('BO result')
            print(incumbents, '-'*21, sep='')
            res = json.JSONEncoder().encode({'result': json.dumps(incumbents),
                                             'history': json.dumps(history)})

            return HttpResponse(res)
        else:
            return HttpResponse('[bo_advice/views.py] error6')
    else:
        return HttpResponse('[bo_advice/views.py] error7')
