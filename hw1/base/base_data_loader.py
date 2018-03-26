
class BaseDataLoader:
    def __init__(self, batch_size):
        """
        TODO:
            set self.n_batch = len(data) // self.batch_size
            initialize self.x, self.y
        """
        self.batch_size = batch_size

    def __iter__(self):
        """
        TODO:
            shuffle the data at the begining of each epoch
            initialize self.batch_idx = 0
            
            return self
        """
        return NotImplementedError

    def __next__(self):
        """
        TODO:
            return x_batch, y_batch
        """
        return NotImplementedError

    def __len__(self):
        """
        TODO:
            return self.n_batch
        """
        return NotImplementedError

