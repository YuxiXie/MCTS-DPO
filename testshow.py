import numpy as np
import matplotlib.pyplot as plt

def shape_two_plots(data1, data2, dataset_name):
    plt.figure(figsize=(10, 6))
    colors = ['b', 'r', 'g', 'y']
    markers = ['*', '^', 'o', 's']
    for idx, (method_name, values) in enumerate(data1.items()):
        x, y = [], []
        for v in values:
            if isinstance(v[0], list):
                x.append(sum(vv[0] for vv in v) / len(v))
                y.append(sum(vv[1] for vv in v) / len(v))
            else:
                x.append(v[0])
                y.append(v[1])
        plt.plot(x, y, colors[idx], marker=markers[idx], linestyle='-',
                 markersize=10, label=method_name)
    for idx, (method_name, values) in enumerate(data2.items()):
        x, y = [], []
        for v in values:
            if isinstance(v[0], list):
                x.append(sum(vv[0] for vv in v) / len(v))
                y.append(sum(vv[1] for vv in v) / len(v))
            else:
                x.append(v[0])
                y.append(v[1])
        plt.plot(x, y, colors[idx], marker=markers[idx], linestyle='--',
                 markersize=10, label=method_name)
    plt.xlabel('# Tokens', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title(f'Cost-Performance Curves on {dataset_name}', fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.show()

def shape_plot(data, dataset_name, linestyle='-'):
    plt.rcParams.update({'font.size': 24})
    plt.figure(figsize=(10, 6))
    # colors = ['tab:blue', 'r', 'g', 'y']
    # markers = ['*', '^', 'o', 's']
    colors = ['y', 'tab:blue', 'g']
    markers = ['o', '*', '^']
    for idx, (method_name, values) in enumerate(data.items()):
        x, y = [], []
        for v in values:
            if isinstance(v[0], list):
                x.append(sum(vv[0] for vv in v) / len(v))
                y.append(sum(vv[1] for vv in v) / len(v))
            else:
                x.append(v[0])
                y.append(v[1])
        plt.plot(x, y, colors[idx], marker=markers[idx], linestyle=linestyle,
                 markersize=10, label=method_name)
        for xi, yi in zip(x, y):
            # delta_x = -100 if colors[idx] == 'g' else 300
            # delta_x = 500 if colors[idx] == 'r' else delta_x
            # delta_y = 0 if colors[idx] == 'b' else (-.02 if colors[idx] == 'g' else -.5)
            # delta_y = -.75 if colors[idx] == 'g' else delta_y
            delta_y = -.5 if colors[idx] == 'tab:blue' else 0
            delta_x = 600 if colors[idx] == 'tab:blue' else 0
            if (colors[idx] == 'g' and round(yi, 1) == 40.6) or \
                (colors[idx] == 'y' and round(yi, 1) == 42.9):
                delta_y, delta_x = -1, 500
            plt.annotate(f'{yi:.1f}', (xi + delta_x, yi + delta_y), textcoords="offset points",
                         color=colors[idx], xytext=(0,5), ha='center')
    plt.xlabel('Cost (# Tokens)', fontsize="large")
    plt.ylabel('Accuracy', fontsize="large")
    # plt.title(f'Cost-Performance Curves on {dataset_name}', fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize="large")

    # plt.tight_layout()
    tsk_type = 'cot' if 'cot' in dataset_name.lower() else 'pal'
    plt.subplots_adjust(wspace=0.15, bottom=0.15, top=0.95, left=0.1, right=0.97)
    plt.savefig(f'figures/gsm8k_{tsk_type}.pdf', format='pdf')
    plt.show()
    
def shape_k_plot(data, baselines, fname, xlabel='# Iterations', ylabel='Accuracy'):
    plt.rcParams.update({'font.size': 26})
    plt.figure(figsize=(10, 8))
    colors = ['tab:blue', 'g', 'y', 'r']
    linestyles = ['-', '-', '-', '-']
    markers = ['*', '^', '^', '^']
    # colors = ['b', 'r', 'g', 'y']
    # linestyles = ['-', '-']
    # markers = ['*', '^', 'o', 's']
    for idx, (method_name, values) in enumerate(data.items()):
        x, y = [], []
        for v in values:
            if isinstance(v[0], list):
                x.append(sum(vv[0] for vv in v) / len(v))
                y.append(sum(vv[1] for vv in v) / len(v))
            else:
                x.append(v[0])
                y.append(v[1])
        if len(x) == 1:
            plt.scatter(x, y, c=colors[idx], marker=markers[idx],
                        label=method_name)
        else:
            plt.plot(x, y, colors[idx], marker=markers[idx], linestyle=linestyles[idx % 2],
                     markersize=14, label=method_name)
        for xi, yi in zip(x, y):
            # delta_y = -2.8 if idx % 2 == 3 and yi > 32.5 else 0
            delta_y = 0
            if 'Sampling' in method_name or 'Greedy' in method_name:
                if yi != 72.2:
                    delta_y -= 2.5
            plt.annotate(f'{yi:.1f}', (xi, yi + delta_y), textcoords="offset points", 
                         xytext=(0,5), ha='center', color=colors[idx])
    for idx, (method_name, value) in enumerate(baselines.items()):
        plt.axhline(y=value, color=colors[idx], linestyle='--', label=method_name)
        plt.annotate(f'{value:.1f}', (10, value), color='tab:blue' if idx == 0 else 'g',
                     horizontalalignment='center', 
                     verticalalignment='bottom')
    plt.xlabel(xlabel, fontsize="large")
    plt.ylabel(ylabel, fontsize="large")
    # plt.title(f'BeamSize-Performance Curves', fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='lower right', fontsize=18)

    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, bottom=0.13, top=0.95, left=0.13, right=0.97)
    plt.savefig(fname, format='pdf')
    # plt.show()

baselines = {
    'Greedy': 60.6
}

K_plots = {
    'Iterative Learning (Greedy)': [
        # [0, 60.6],
        [512/80, 72.2],
        [1024/80, 73.6],
        [1536/80, 74.7],
        [2048/80, 74.9],
        [2560/80, 76.4],
        [3072/80, 75.8],
        [3584/80, 71.4],
    ],
    'Iterative Learning': [
        [0/80, 60.6],
        [512/80, 79.7],
        [1024/80, 86.9],
        [1536/80, 90.0],
        [2048/80, 91.3],
        [2560/80, 92.4],
        [3072/80, 93.3],
        [3584/80, 94.1],
    ],
    'Sampling Only': [
        [0/80, 60.6],
        [512/80, 71.9],
        [1024/80, 80.6],
        [1536/80, 86.6],
        [2048/80, 89.3],
        [2560/80, 91.7],
        [3072/80, 92.9],
        [3584/80, 93.5],
    ],
}

shape_k_plot(K_plots, baselines, 'test.pdf', xlabel='Training Progress (%)', ylabel='Cumulative Pass Rate')

K_plots = {
    'Iterative Learning (Greedy)': [
        # [0, 60.6],
        [1, 72.2],
        [2, 73.6],
        [3, 74.7],
        [4, 74.9],
        [5, 76.4],
        [6, 75.8],
        [7, 71.4],
    ],
    'Iterative Learning (Cumulative)': [
        [0/80, 60.6],
        [1, 79.7],
        [2, 86.9],
        [3, 90.0],
        [4, 91.3],
        [5, 92.4],
        [6, 93.3],
        [7, 94.1],
    ],
    'Sampling Only (Cumulative)': [
        [0/80, 60.6],
        [1, 71.9],
        [2, 80.6],
        [3, 86.6],
        [4, 89.3],
        [5, 91.7],
        [6, 92.9],
        [7, 93.5],
    ],
}

shape_k_plot(K_plots, baselines, 'test2.pdf', ylabel='Cumulative Pass Rate')

K_plots = {
    r'Sampling Only (SC@$k$)': [
        [0/80, 60.6],
        [1, 71.9],
        [2, 80.6],
        [3, 86.6],
        [4, 89.3],
        [5, 91.7],
        [6, 92.9],
        [7, 93.5],
    ],
}

shape_k_plot(K_plots, baselines, 'test2.pdf', xlabel=r'$k$')