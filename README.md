Project Overview
With the rise of on-device deep learning, inference latency has become a critical metric for deploying Deep Neural Network (DNN) models on various mobile and edge devices. Precise latency prediction is vital, particularly for scenarios where measuring latency directly on real devices is impractical due to resource constraints or high costs, such as in the search for optimal DNN models within large model-design spaces.

nn-Meter is an efficient and innovative system that accurately predicts the inference latency of DNN models on diverse edge devices. Instead of relying on direct device measurements, nn-Meter divides the inference process into kernels, which are the fundamental execution units on devices. By predicting latency at the kernel level, nn-Meter achieves higher accuracy in its latency estimations across a wide range of edge devices.. We are currently
evaluating platforms on a large dataset of the MNIST which has 60,000 images for training and 10,000 images for testing. 

This project aims to set up nn-Meter and use it to evaluate and correlate the latency behavior of various neural networks (NNs) on different device cores.

Key Components of nn-Meter
nn-Meter incorporates two core techniques to enhance latency prediction:

Kernel Detection: Automatically identifies the device's kernel or execution units using well-designed test cases. This enables the prediction of each DNN layer's execution time on a device-specific basis.
Adaptive Sampling: Efficiently samples the most impactful configurations from the large model space, building accurate kernel-level latency predictors and minimizing the required sampling efforts.

Objectives
Setup nn-Meter: Install and configure nn-Meter for testing DNN inference latency on multiple edge devices.
Latency Analysis: Utilize nn-Meter to conduct kernel-level latency predictions on diverse DNN models.
Correlation Study: Analyze and correlate the behavior of different NNs on distinct device cores, assessing the impact of model design and runtime optimizations on latency.

Usage
Model Configuration: Prepare the DNN models you wish to test in the models/ directory. Ensure each model is in a compatible format for nn-Meter.

Run Latency Prediction: Use nn-Meter's CLI or Python API to predict latency for each model.

Analyze Results: nn-Meter outputs latency predictions for each kernel in the model. Use these results to compare latency behaviors across different NNs and device cores.

Results and Analysis
After running latency predictions, review the results to determine the impact of model design on inference latency. By analyzing latency behavior on different cores, insights can be gained on optimal DNN model designs for specific device constraints.

We enable the GPU and select either "cuda" if available or "cpu".

We must also select 1 of 4 total predictors being: 1: "cortexA76cpu_tflite21", 2: "adreno640gpu_tflite21", 3: "adreno630gpu_tflite21", 4: "myriadvpu_openvino2019r2"

cortexA76cpu_tflite21:     Processor (CortexA76 CPU),  Device (Pixel4)
adreno640gpu_tflite21:     Processor (Adreno 640 GPU), Device (Mi9)
adreno630gpu_tflite21:     Processor (Adreno 630 GPU), Device (Pixel3XL)
myriadvpu_openvino2019r2:  Processor (Myriad VPU),     Device (Intel Movidius NCS2)

Submissions
Presentation of the Project work
6 to 8 page IEEE-formatted paper
Finalized GitHub repository

Contributing
Jake Horio & Mason Mac
