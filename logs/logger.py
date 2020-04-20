import logging
import os
from lightgbm.callback import _format_eval_result
from configs.local_parameter import *

def log_best(model, metric):
    logging.debug('===best_iteration===')
    logging.debug(model.best_iteration)
    print(model.best_score)
    logging.debug(model.best_score['valid_1'][metric])


def log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list \
                and (env.iteration + 1) % period == 0:
            result = '\t'.join([
                _format_eval_result(x, show_stdv)
                for x in env.evaluation_result_list
            ])
            logger.log(level, '[{}]\t{}'.format(env.iteration + 1, result))
    _callback.order = 10
    return _callback

def make_log(model_name,log):
    os.makedirs(f"{MAKE_PATH}",exist_ok=True)
    
    log_path = f"{MAKE_PATH}/{CASE}_{NOW:%Y%m%d%H%M%S}_{model_name}_score.txt"
    
    #同時刻での書き込みがなかった場合、最初に時刻を書き込み
     
    #そもそもファイルが存在しなかった場合はm、ファイルを作成して時刻を書き込み
    if not os.path.exists(log_path):
        with open(log_path, mode='w') as f:
            f.write("\n" + str(NOW) + "\n")

    #ファイルが存在する場合は、同時刻の書き込みがない場合のみ時刻を書き込み
    else:
        with open(log_path, mode='r') as f:
            lines = f.readlines()
            lines_strip = [line.strip() for line in lines]
            if str(NOW) not in lines_strip:
                with open(log_path, mode='a') as f:
                    f.write("\n" + str(NOW) + "\n")
                    
    #書きたい内容を書き込み
    print(log)
    with open(log_path, mode='a') as f:
        f.write(log+"\n")
