import json
import os
from eval.evaluator import Evaluator
from torch.utils.data import Dataset


class SQLDataset(Dataset):
    def __init__(self, examples):
        super(SQLDataset, self).__init__()
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class Example:
    @classmethod
    def configuration(cls, args):
        cls.dataset_dir = os.path.join('data', args.dataset)
        cls.evaluator = Evaluator(os.path.join(cls.dataset_dir, 'tables.json'), os.path.join(cls.dataset_dir, 'database'))

    @classmethod
    def load_dataset(cls, choice, args):
        assert choice in ['train', 'dev']
        with open(os.path.join('data', args.dataset, choice + '.json'), 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        return SQLDataset(dataset)

    @classmethod
    def use_database_testsuite(cls):
        cls.evaluator.change_database(os.path.join(cls.dataset_dir, 'database-testsuite'))
