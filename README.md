install latest torch (currently 12.6)
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

install ultralytics for the cuda-toolkits, to enable using cuda 
```
pip install ultralytics
```

# usage 
training a resnet-18 model on classification task 
given a folder ./dataset and file ./class.yaml 
the workflows sequentially balance datastet -> augment dataset -> split into train/val/test -> train resnet on train dataset -> inference on 1 image 
## prepare
folder ./dataset has structure like 
dataset/
├── class_1/
│   ├── image_1.jpg
│   ├── imgae_2.jpg
├── class_2/
│   ├── image_1.jpg
│   ├── imgae_2.jpg
└── class_3/
...

in ./class.yaml, replace classes with your own classes name 

## running
0. create a more balance dataset , with each class have number of samples equals to the total_samples/number_of_classes, sample from each class are collected randomly
```
python 0_balance.py
```
1. augment dataset 
```
python 1_augment.py
```
2. split dataset 
```
python 2_split_dataset.py
```
3. train resnet18 model
```
python 3_train.py
```
4. inference on 1 image
```
python 4_infer.py --image_path <path_image>
```