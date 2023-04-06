from src.utils import Config

class BaseCPAggregator:
    def __init__(self, config: Config):
        self.config = config
        # init model, pass for the inheritance class

    def run(self, **kwargs):
        pass