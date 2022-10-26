import csv
import statistics
import os
from dataclasses import dataclass, field
from turtle import title

import matplotlib
matplotlib.use('Agg') # To avoid using embedded display, if display available can be removed 
from matplotlib import pyplot as plt


@dataclass
class PlotXY:
    """
    Plotting class, contains a list of x and y values (values should be the same lenght)
    """
    x: list[int] = field(default_factory=list)
    y: list[float] = field(default_factory=list)


def read_csv(file: str, header_x: str, header_y: str):
    """
    Read csv file, return PlotXY class object
    """
    print(f'Reading file: {file}')
    list_x, list_y = [], []
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(f=csvfile, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE)
        for row in reader:
            list_x.append(float(row[header_x]))
            list_y.append(float(row[header_y]))
    return PlotXY(x=list_x, y=list_y)
    

def plot_epoch_recall_avg():
    plt.clf()
    output_file = './results/common/plot_epoch_recall_avg.png'
    # Create output path if not exist, if exists do not raise error
    os.makedirs(name=os.path.dirname(output_file), exist_ok=True)

    input_file1 = 'results/b3-train2345-valid1/results.csv'
    input_file2 = 'results/b3-train1345-valid2/results.csv'
    input_file3 = 'results/b3-train1245-valid3/results.csv'
    input_file4 = 'results/b3-train1235-valid4/results.csv'
    input_file5 = 'results/b3-train1234-valid5/results.csv'

    header_x = 'epoch'
    header_y = 'metrics/recall'

    plot_data1 = read_csv(file=input_file1, header_x=header_x, header_y=header_y)
    plot_data2 = read_csv(file=input_file2, header_x=header_x, header_y=header_y)
    plot_data3 = read_csv(file=input_file3, header_x=header_x, header_y=header_y)
    plot_data4 = read_csv(file=input_file4, header_x=header_x, header_y=header_y)
    plot_data5 = read_csv(file=input_file5, header_x=header_x, header_y=header_y)

    plot_data = PlotXY(
        x=plot_data1.x,
        y=[statistics.fmean(values) for values in zip(
            plot_data1.y,
            plot_data2.y,
            plot_data3.y,
            plot_data4.y,
            plot_data5.y)])

    print(f'Plotting...')
    plt.plot(plot_data.x, plot_data.y, label='Average')

    plt.title('Precision curve')
    plt.xlabel('Epoch')
    plt.ylabel('Average precision')

    print(f'Saving plot to: {output_file}')
    plt.savefig(output_file)


def plot_epoch_precision():
    plt.clf()
    output_file = './results/common/plot_epoch_precision.png'
    # Create output path if not exist, if exists do not raise error
    os.makedirs(name=os.path.dirname(output_file), exist_ok=True)

    input_file1 = 'results/b3-train2345-valid1/results.csv'
    input_file2 = 'results/b3-train1345-valid2/results.csv'
    input_file3 = 'results/b3-train1245-valid3/results.csv'
    input_file4 = 'results/b3-train1235-valid4/results.csv'
    input_file5 = 'results/b3-train1234-valid5/results.csv'

    header_x = 'epoch'
    header_y = 'metrics/precision'

    plot_data1 = read_csv(file=input_file1, header_x=header_x, header_y=header_y)
    plot_data2 = read_csv(file=input_file2, header_x=header_x, header_y=header_y)
    plot_data3 = read_csv(file=input_file3, header_x=header_x, header_y=header_y)
    plot_data4 = read_csv(file=input_file4, header_x=header_x, header_y=header_y)
    plot_data5 = read_csv(file=input_file5, header_x=header_x, header_y=header_y)

    print(f'Plotting...')
    plt.plot(plot_data1.x, plot_data1.y, label='train2345-valid1')
    plt.plot(plot_data2.x, plot_data2.y, label='train1345-valid2')
    plt.plot(plot_data3.x, plot_data3.y, label='train1245-valid3')
    plt.plot(plot_data4.x, plot_data4.y, label='train1235-valid4')
    plt.plot(plot_data5.x, plot_data5.y, label='train1234-valid5')

    plt.legend(title='k-fold group')
    plt.title('Precision curve')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')

    print(f'Saving plot to: {output_file}')
    plt.savefig(output_file)

    print('ok')


if __name__=='__main__':
    plot_epoch_precision()
    plot_epoch_recall_avg()
