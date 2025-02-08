import time

class TimeUse(object):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name
        
    def __enter__(self):
        self.t = time.time()
    
    def __exit__(self,exc_type,exc_value,traceback):
        print("Module {} : Using {} seconds.".format(self.name, time.time()-self.t))
