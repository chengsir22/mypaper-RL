import logging


def get_logger(log_file, level=logging.INFO):
    # 创建日志记录器，指定日志级别
    logger = logging.getLogger("logger")
    logger.setLevel(level)
    # 创建文件处理器，指定日志文件地址
    handler = logging.FileHandler(log_file)
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    # 将处理器添加到日志记录器
    logger.addHandler(handler)
    return logger
