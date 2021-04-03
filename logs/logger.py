"""
lightgbmの学習履歴を標準入出力ではなくlogger経由で出力

参考：https://amalog.hateblo.jp/entry/lightgbm-logging-callback
"""

import logging
from lightgbm.callback import _format_eval_result


def log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([
                _format_eval_result(x, show_stdv)
                for x in env.evaluation_result_list
            ])
            logger.log(level, '[{}]\t{}'.format(env.iteration + 1, result))
    _callback.order = 10
    return _callback
