import os
from datasets.common_dataset import CommonDataset
from datasets.reader import read_images_labels


class VisDA17(CommonDataset):
    def __init__(self, data_root: str, txt_file: str, domains: list, status: str, trim: int = 0):
        super().__init__(is_train=(status == 'train'))
        self.txt_file = txt_file
        self.lines = open(self.txt_file, 'r').readlines()
        self.data_root = data_root
        self._domains = ["aeroplane", "bicycle", "bus", "car", "horse", "motorbike", "person", "train", "truck",
                         "skateboard"]

        if domains[0] not in self._domains:
            raise ValueError(f'Expected \'domain\' in {self._domains}, but got {domains[0]}')
        _status = ['train', 'val', 'test']
        if status not in _status:
            raise ValueError(f'Expected \'status\' in {_status}, but got {status}')

        self.image_root = data_root

        # read txt files
        data = read_images_labels(
            os.path.join(self.txt_file),
            shuffle=(status == 'train'),
            trim=0
        )

        self.data = data
        self.domain_id = [0] * len(self.data)
        self.number_classes = 12

    def __len__(self):
        return len(self.lines)
