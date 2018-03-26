
class Logger:
    """
    Logger class record the training feature at each training epoch.
    
    example:
    
    logger.entries = { 
                       1 :{
                           'epoch':    1,
                           'accuracy': 0.1,
                           'loss':     inf
                          }
                       2 :{
                           'epoch':    2,
                           'accuracy': 0.1,
                           'loss':     inf
                          }
                       ...
                     } 
    """
    def __init__(self):
        self.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return str(self.entries)
