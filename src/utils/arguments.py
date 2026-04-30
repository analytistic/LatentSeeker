from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataArgs:
    data_name: str = field(default="cifar10", metadata={"help": "The name of the dataset to use (via the datasets library)."})
    data_path: str = field(default='', metadata={"help": "Path or name of the dataset; if None, defaults to data_name."})
    split: str = field(default="train", metadata={"help": "The split of the dataset to use."})
    streaming: bool = field(default=False, metadata={"help": "Whether to use streaming loading for the dataset."})
    head_len: int | None = field(default=None, metadata={"help": "The length of the head part of the dataset."})

    def __post_init__(self):
        if self.data_path == '':
            self.data_path = self.data_name
        assert (self.streaming and self.head_len is not None) or (not self.streaming and self.head_len is None), "When streaming is True, head_len must be specified; when streaming is False, head_len must be None."

