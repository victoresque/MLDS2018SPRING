
class Logger:
    """
    Logger class record the training feature at each training epoch.
    
    example:
    
    logger.entries = { 
                       1 :{
                           epoch:    [1, 2, 3, ...],
                           accuracy: [0.1, 0.1, 0.1, ...]
                          }
                       2 :{
                           epoch:    [1, 2, 3, ...],
                           accuracy: [0.1, 0.1, 0.1, ...]
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
