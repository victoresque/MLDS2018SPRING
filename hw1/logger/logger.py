
class Logger:
    def __init__(self):
        self.entries = {}

    def add_entry(self, epoch, entry):
        self.entries[epoch] = entry

    def print(self):
        print(self.entries)

