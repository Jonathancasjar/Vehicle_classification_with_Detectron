# Vehicle classification from images

## 1. Install

I'll use `Docker` to install all the needed packages and libraries easily. Two Dockerfiles are provided for both CPU and GPU support.

- **CPU:**

```bash
$ docker build -t vehicle_classification --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f docker/Dockerfile .
```

- **GPU:**

```bash
$ docker build -t vehicle_classification --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f docker/Dockerfile_gpu .
```

### Run Docker

- **CPU:**

```bash
$ docker run --rm --net host -it \
    -v $(pwd):/home/app/src \
    --workdir /home/app/src \
    vehicle_classification \
    bash
```

- **GPU:**

```bash
$ docker run --rm --net host --gpus all -it \
    -v $(pwd):/home/app/src \
    --workdir /home/app/src \
    vehicle_classification \
    bash
```

### Run Unit test


```bash
$ pytest tests/
```

## Working on the project

Below I will the steps and the order in which I solve this project.

### 2. Prepare the data

As a first step, I must extract the images from the file `car_ims.tgz` and put them inside the `data/` folder. Also place the annotations file (`car_dataset_labels.csv`) in the same folder. It should look like this:

```
data/
    ├── car_dataset_labels.csv
    ├── car_ims
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   ├── ...
```

Then, I able to run the script `scripts/prepare_train_test_dataset.py`. It will format my data in a way Keras can use for training the CNN model.

### 3. Train the first CNN (Resnet50)

After have images in place, it's time to create the first CNN and train it on the dataset. To do so, I will make use of `scripts/train.py`.

The only input argument it receives is a YAML file with all the experiment settings like dataset, model output folder, epochs,
learning rate, data augmentation, etc.

Each time I going to train a new model, I'll create a new folder inside the `experiments/` folder with the experiment name. Inside this new folder, create a `config.yml` with the experiment settings. I'll store the model weights and training logs inside the same experiment folder to avoid mixing things between different runs. The folder structure should look like this:

```bash
experiments/
    ├── exp_001
    │   ├── config.yml
    │   ├── logs
    │   ├── model.01-6.1625.h5
    │   ├── model.02-4.0577.h5
    │   ├── model.03-2.2476.h5
    │   ├── model.05-2.1945.h5
    │   └── model.06-2.0449.h5
    ├── exp_002
    │   ├── config.yml
    │   ├── logs
    │   ├── model.01-7.4214.h5
    ...
```

- `utils.load_config()`: Takes as input the path to an experiment YAML configuration file, loads it, and returns a dict.
- `resnet50.create_model()`: Returns a CNN ready for training or for evaluation, depending on the input parameters received. 
- `data_aug.create_data_aug_layer()`: Used by `resnet50.create_model()`. This function adds data augmentation layers to the model that will be used only while training.

### 4. Evaluate the trained model

After running many experiments it's time to check its performance on the test dataset and prepare a nice report with some evaluation metrics.

I use the notebook `notebooks/Model Evaluation.ipynb` to do it. 

### 5. Improve classification by removing noisy background

As I already saw in the `notebooks/EDA.ipynb` file. Most of the images have a background that may affect the model learning during the training process.

It's a good idea to remove this background. One thing I do is to use a Vehicle detector to isolate the car from the rest of the content in the picture.

I will use [Detectron2](https://github.com/facebookresearch/detectron2) framework for this. It offers a lot of different models.

For this assignment i use the model called "R101-FPN".

In particular, I will use a detector model trained on [COCO](https://cocodataset.org) dataset which has a good balance between accuracy and speed. This model can detect up to 80 different types of objects but here I'm only interested on getting two out of those 80, those are the classes "car" and "truck".

For this assignment, I'll run the following two files:

- `scripts/remove_background.py`: It will process the initial dataset used for training the model on **item (3)**, removing the background from pictures and storing the resulting images on a new folder.
- `utils/detection.py`: This module loads the detector and implements the logic to get the vehicle coordinate from the image.

Now I have the new dataset in place, I'll start training a new model and checking the results in the same way as I did for steps items **(3)** and **(4)**.

