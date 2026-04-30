from .base_dataset import BaseDataset
from datasets import load_dataset
from ..utils.arguments import DataArgs
from pathlib import Path


class WebSFTDataset(BaseDataset):
    def __init__(self, configs: DataArgs, **kwargs):
        super(WebSFTDataset, self).__init__(configs)
            self.dataset = load_dataset(f'{Path(configs.data_path).suffix[1:]}', configs.data_path, split=configs.split, streaming=configs.streaming)
        if configs.streaming:
            assert configs.head_len is not None, "When streaming is True, head_len must be specified."
            self.dataset = self.dataset.take(configs.head_len)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
    


if __name__ == "__main__":
    from ..utils.arguments import DataArgs
    data_args = DataArgs(data_name="data/openseeker/openseeker_v1_data.jsonl", split='train', streaming=True, head_len=10)
    dataset = WebSFTDataset(data_args)
    print(dataset[0])