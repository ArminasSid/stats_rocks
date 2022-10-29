import os
from dataclasses import dataclass, field
from glob import glob
from collections import Counter

import matplotlib
matplotlib.use('Agg') # To avoid using embedded display, if display available can be removed 
from matplotlib import pyplot as plt

# Global font size change on images
plt.rcParams.update({'font.size': 14})

@dataclass
class TrainValid:
    train: list[float] = field(default_factory=list)
    valid: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.train = self._update_values(values_list=self.train)
        self.valid = self._update_values(values_list=self.valid)

    def _update_values(self, values_list: list) -> list:
        # Returns a new list with str values instead of int
        new_list = []
        for value in values_list:
            new_list.append(self._update_value(value=value))
        return new_list

    def _update_value(self, value: int) -> str:
        if value == 0:
            return 'boulder'
        if value == 1:
            return 'submerged boulder'
        raise KeyError('Don\'t know such key')

    def get_train_keys_amount(self):
        return Counter(self.train).keys(), Counter(self.train).values()

    def get_valid_keys_amount(self):
        return Counter(self.valid).keys(), Counter(self.valid).values()


def read_file(filename: str):
    """
    Read file and get only first symbol from every line
    """
    elements = []
    with open(file=filename) as f:
        for line in f:
            elements.append(int(line[0]))
    return elements


def read_train_valid_instances(train_files: list[str], valid_files: list[str]):
    train_elems, valid_elems = [], []
    for filename in train_files:
        train_elems.extend(read_file(filename=filename))
    for filename in valid_files:
        valid_elems.extend(read_file(filename=filename))
    return TrainValid(train=train_elems, valid=valid_elems)


def create_single_bar():
    # Possible options are:
    # [train1234-valid5, train1235-valid4, train1245-valid3, train1345-valid2, train2345-valid1]
    instance = 'train2345-valid1'

    output_file = f'results/common/bar-{instance}.png'
    os.makedirs(name=os.path.dirname(output_file), exist_ok=True)

    basepath = 'results/bar'
    train = glob(f'{basepath}/{instance}/train/*')
    valid = glob(f'{basepath}/{instance}/valid/*')

    # Get all train and valid class instances
    train_valid = read_train_valid_instances(train_files=train, valid_files=valid)

    # Get list of keys and amount of occurences of that key
    train_keys, train_occur = train_valid.get_train_keys_amount()
    valid_keys, valid_occur = train_valid.get_valid_keys_amount()

    # Create bar
    plt.bar(x=train_keys, height=train_occur, color='green', edgecolor='black')
    plt.bar(x=valid_keys, height=valid_occur, color='red', edgecolor='black')

    # Set limit of y axis
    plt.ylim(top=13000)

    # Set titles, etc.
    plt.title(label=f'{instance}')
    plt.ylabel(ylabel='Instances')
    plt.xlabel(xlabel='Class')
    plt.legend(['Training data', 'Validation data'], borderpad=0.1)

    # Save to file
    plt.savefig(output_file)

    print('ok')



if __name__=='__main__':
    create_single_bar()
