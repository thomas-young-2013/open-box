# =====for vvv=====
# after the execution of this part
# goto the roor directory using the command `tensorboard --logdir`
import re
from tensorboardX import SummaryWriter

def visualize(logging_file_path):
    file = open(logging_file_path)
    log_entrys = file.readlines()
    writer = SummaryWriter()
    n_iter = 0
    config_dict = dict()
    score_dict = dict()
    for log_entry in log_entrys:
        if re.match('\[INFO\]', log_entry) is not None:  # start parsing
            # print("new iteration:")
            n_iter += 1
            config_dict = dict()
            score_dict = dict()
        elif re.match(', result is:', log_entry) is None:  # continue parsing configuration
            search_obj = re.search(r'(.*), Value: (.*)', log_entry)
            config_dict[str(search_obj.group(1))] = float(search_obj.group(2))
            # print('key is: ', str(search_obj.group(1)), ' value is : ', config_dict[str(search_obj.group(1))])
        else:  # parsing performance and end
            search_obj = re.search(', result is: (.*)', log_entry)
            score_dict['_perf'] = float(search_obj.group(1))
            # print("end parsing------------")
            # print("config_dict = ", config_dict)
            # print("score_dicr = ", score_dict)
            writer.add_hparams(config_dict, score_dict, name="trial" + str(n_iter))
            writer.add_scalar('_perf', score_dict['_perf'], n_iter)
            writer.add_scalars('data/timeline', score_dict, n_iter)

    # writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
