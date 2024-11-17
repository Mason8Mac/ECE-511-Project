nn-Meter is a novel and efficient system to accurately predict the inference latency of DNN models on diverse edge devices. 
The main idea is to divide a whole model inference into kernels and to conduct kernel-level predictions. We are currently
evaluating platforms on a large dataset of the MNIST which has 60,000 images for training and 10,000 images for testing. 

We enable the GPU and select either "cuda" if available or "cpu".

We must also select 1 of 4 total predictors being: 1: "cortexA76cpu_tflite21", 2: "adreno640gpu_tflite21", 3: "adreno630gpu_tflite21", 4: "myriadvpu_openvino2019r2"

cortexA76cpu_tflite21:     Processor (CortexA76 CPU),  Device (Pixel4)
adreno640gpu_tflite21:     Processor (Adreno 640 GPU), Device (Mi9)
adreno630gpu_tflite21:     Processor (Adreno 630 GPU), Device (Pixel3XL)
myriadvpu_openvino2019r2:  Processor (Myriad VPU),     Device (Intel Movidius NCS2)
