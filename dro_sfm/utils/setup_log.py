import logging
import datetime
import os.path as osp

def setup_log(log_name):
    # Initialize logging
    # simple_format = '%(levelname)s >>> %(message)s'
    medium_format = (
        '%(levelname)s : %(filename)s[%(lineno)d]'
        ' >>> %(message)s'
    )

    # Reference:
    #   http://59.125.118.185:8088/ALG/TestingTools/-/blob/master/model_performance_evaluation_tool/src/common/testingtools_log.py
    formatter = logging.Formatter(
                '[%(asctime)s] %(filename)s->%(funcName)s line:%(lineno)d [%(levelname)s]%(message)s')

    medium_format_new = (
        '[%(asctime)s] %(levelname)s : %(filename)s[%(lineno)d] %(funcName)s'
        ' >>> %(message)s'
    )

    get_log_file = osp.join(osp.dirname(__file__), f'../../{log_name}')

    logging.basicConfig(
        filename=get_log_file,
        filemode='w',
        level=logging.INFO,
        format=medium_format_new
    )
    logging.info('@{} created at {}'.format(
        get_log_file,
        datetime.datetime.now())
    )
    print('\n===== log_file: {}\n'.format(get_log_file))
