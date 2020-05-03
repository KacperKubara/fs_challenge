class Runner():
    """ Abstract class to be inherited by Processor and EDA"""
    def run(self):
        raise NotImplementedError("run() has to be overridden")
