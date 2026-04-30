
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, configs):
        super(BaseDataset, self).__init__()
        self.configs = configs

