from torch.utils.data import Dataset


class LabeledDataset(Dataset):
    def __init__(self, ds: Dataset, active_label: int = 1):
        super().__init__()
        self.ds = ds
        self.active_label = active_label
    
    def __getitem__(self, index):
        sample = self.ds.__getitem__(index)
        return {'inputs': sample[0], 'labels': sample[self.active_label]}
    
    def __len__(self):
        return len(self.ds)
