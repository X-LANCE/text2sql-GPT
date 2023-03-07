import json
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
    def configuration(cls, table_path='data/tables.json', db_dir='data/database'):
        cls.evaluator = Evaluator(table_path, db_dir)

    @classmethod
    def load_dataset(cls, choice):
        assert choice in ['train', 'dev']
        with open(f'data/{choice}.json', 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        return SQLDataset(dataset)

    @classmethod
    def use_database_testsuite(cls, db_dir='data/database-testsuite'):
        cls.evaluator.change_database(db_dir)
