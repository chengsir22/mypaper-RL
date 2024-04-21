import utils

class Reinforce():
    def __init__(self):
        self.log_file = f"./out/log/04_reinforce.log"
        global logger
        logger = utils.setup_logger(log_file = self.log_file)
        utils.random_env()
        
     