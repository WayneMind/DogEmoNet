import os
import logging
import datetime


def create_logger(cfg, phase='train'):
    """
    创建logger并返回
    :param cfg: 配置文件，包括日志路径(log_path)、日志级别(level)等参数
    :param phase: 当前阶段，train或val，默认为train
    :return: logger对象
    """

    log_path = cfg['log_path']
    level = cfg['level']

    # 如果log_path不存在，创建它
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # 配置日志文件的命名方式，如train_2022-01-02_15-44-35.log
    log_file = os.path.join(log_path, f"{phase}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

    # 创建logger并设置日志级别
    logger = logging.getLogger(phase)
    logger.setLevel(level)

    # 将日志写入文件
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)

    # 设置日志输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # 将日志处理程序添加到logger
    logger.addHandler(fh)

    return logger

"""
该脚本用于创建日志记录器(logger)，该记录器用于记录模型训练过程中的详细信息，以便在训练结束后分析和评估结果。在函数create_logger中，该脚本使用了Python内置的logging模块，它提供了一种灵活的方式来记录各种类型的信息。

具体来说，该函数接受两个参数：配置文件cfg和训练或验证的当前阶段phase。cfg包含了日志文件路径、日志级别等信息。在函数中，首先会检查是否存在该路径的日志文件，如果不存在，会新建该路径。接着，函数将当前时间添加到日志文件名中，以便区分不同的日志文件。

接下来，函数创建一个名为logger的logger对象，并设置它的日志级别。这里的日志级别是在配置文件中指定的。日志级别是一种筛选日志消息的方式，每种级别都有与其关联的消息类型。例如，如果将日志级别设置为INFO，则logger会记录INFO、WARNING、ERROR和CRITICAL级别的消息。在此示例中，我们将日志级别设置为一个字符串，它在logger对象中被转换为相应的日志级别。例如，如果日志级别是"INFO"，它将被转换为logging.INFO。
"""