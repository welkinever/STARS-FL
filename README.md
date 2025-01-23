# STARS-FL
STARS-FL: Accelerating Federated Learning over Heterogeneous Mobile Devices via Spatial-Temporal Aware Reconfigured Subnetworks


## Structure

### Server
This part of the code is used to conduct large-scale simulation experiments. It tests the performance of the federated learning system.

### Clients
This part of the code is used to train models on actual devices using their local data. It helps to get the communication and computation time for different devices. 

## Dataset Preparation
All the datasets(MNIST, CIFAR10, HAR) should be downloaded in "./data/".
## Training
To train the model(s) with Non-iid MNIST Dataset, run this command:

```bash
python ./Server/main.py --data_name MNIST --model_name conv --control_name 1_20_1_non-iid-2_fix_a2-b8_bn_1_1 --algo order-layerwise --lwr 1_1_1_1 --id MNIST-conv

```

## License
This project is currently non-licensed.
