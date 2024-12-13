#Contributors: Jake Horio
#              Mason Mac
#Last Modified: 12/12/24
#Description: Code for plotting 3 feed-forward latency charts correlating CNN model
#             variations with target hardware models

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

kernel_sizes = ['3x3', '5x5', '7x7']
cnn_variations = ['CNN_Conv_1_CIFAR10', 'CNN_LeNet_CIFAR10', 'CNN_Conv_5_CIFAR10', 'CNN_Conv_7_CIFAR10']
hardware_models = ['cortexA76cpu_tflite21', 'adreno640gpu_tflite21', 'adreno630gpu_tflite21', 'myriadvpu_openvino2019r2']

data_3x3 = pd.DataFrame({
    'cortexA76cpu_tflite21': [0.32508133108673115, 0.46252407054838723, 0.6034920027318948, 5.900768357126534],
    'adreno640gpu_tflite21': [0.020352038554954967, 0.08856007902645804, 0.3396798220715028, 2.8836678150775086],
    'adreno630gpu_tflite21': [0.025663486496805915, 0.12238320769486613, 0.4016783799135389, 3.814688506660472],
    'myriadvpu_openvino2019r2': [0.34255892301392266, 0.7002523248846924, 0.8003301176774851, 1.6256752691926366],
}, index=cnn_variations)

data_5x5 = pd.DataFrame({
    'cortexA76cpu_tflite21': [0.39381903840380433, 0.6558726766459484, 1.0326212332196998, 15.803463989321637],
    'adreno640gpu_tflite21': [0.04047886748828837, 0.22044009940423626, 0.8704045024492809, 8.607780603455264],
    'adreno630gpu_tflite21': [0.040710573314987716, 0.24090948545850244, 0.9567908380408119 , 10.15868465351502],
    'myriadvpu_openvino2019r2': [0.3408172563472561, 0.7110006582180257, 0.8868517843441516, 3.0981752691926365],
}, index=cnn_variations)

data_7x7 = pd.DataFrame({
    'cortexA76cpu_tflite21': [0.5159449793794137, 0.9556000551825339, 1.6665431898050649, 25.970806596882646],
    'adreno640gpu_tflite21': [0.06168773506606606, 0.36214122382645775, 1.531740268649282 , 15.481403170988598],
    'adreno630gpu_tflite21': [0.06349423740589673, 0.3656390111675927, 1.7922619993862647, 20.72647521904232],
    'myriadvpu_openvino2019r2': [0.35566892301392267, 0.7469956582180258, 1.073961784344152, 4.9190536025259695],
}, index=cnn_variations)

data_dict = {'3x3': data_3x3, '5x5': data_5x5, '7x7': data_7x7}

for kernel_size, data in data_dict.items():
    x = np.arange(len(hardware_models))
    width = 0.15
    bar_offsets = np.linspace(-width * 2, width * 2, len(cnn_variations))
    plt.figure(figsize=(12, 8))
    for i, cnn_var in enumerate(cnn_variations):
        bars = plt.bar(
            x + bar_offsets[i],
            data.iloc[i],
            width=width,
            label=cnn_var
        )


        for bar, value in zip(bars, data.iloc[i]):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{value:.2f}',
                ha='center',
                va='bottom',
                fontsize=14
            )

    plt.title(f'Feed-Forward Latencies for CNNs with Kernel Size {kernel_size}', fontsize=24)
    plt.xlabel('Target New-Edge Hardware', fontsize=24)
    plt.ylabel('Latency Timing (ms)', fontsize=24)
    plt.xticks(x, hardware_models, fontsize=14)
    plt.legend(title='CNN Models', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

vgg_data = pd.DataFrame({
    'cortexA76cpu_tflite21': [29.254222800750536, 59.524309990994446, 100.0079411227019],
    'adreno640gpu_tflite21': [8.714405592539123, 23.596202286094627, 41.195249064983436],
    'adreno630gpu_tflite21': [11.340690887350455, 30.301412506986843, 59.514694687350406],
    'myriadvpu_openvino2019r2': [9.61009662054399, 13.827399953877322, 19.70867995387732],
}, index=['3x3', '5x5', '7x7'])

x = np.arange(len(hardware_models))
width = 0.2
bar_offsets = np.linspace(-width * 1.3, width * 1.3, len(vgg_data.index))
plt.figure(figsize=(12, 8))

for i, kernel_size in enumerate(vgg_data.index):
    bars = plt.bar(
        x + bar_offsets[i],
        vgg_data.loc[kernel_size],
        width=width,
        label=f'{kernel_size} Kernel'
    )

    for bar, value in zip(bars, vgg_data.loc[kernel_size]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{value:.2f}',
            ha='center',
            va='bottom',
            fontsize=12
        )

plt.title('Feed-Forward Latencies for VGG CNN by Kernel Size', fontsize=24)
plt.xlabel('Target New-Edge Hardware', fontsize=24)
plt.ylabel('Latency Timing (ms)', fontsize=24)
plt.xticks(x, hardware_models, fontsize=14)
plt.legend(title='Kernel Sizes', fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()