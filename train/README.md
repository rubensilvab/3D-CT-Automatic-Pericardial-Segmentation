# Train the model

The architecture of the CAN 3D is present in the [can3d_multi_noskip.py](can3d_multi_noskip.py) script. In order to train the model you just have to load your CSV and chose the folds for training and for validation.


```python
"""Load CSV""" 
# Replace 'file_path.csv' with the path to your CSV file 
file_path_abd = 'C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/Abdominal_5.csv'

# Read the CSV file into a DataFrame
csv_file_abd = pd.read_csv(file_path_abd)
  
# Choose fold for train, val and test
folds_train=[1,2,0]
folds_val=[3]

```

You can also choose the hyperparametrs that you want in this dic:
```python
hyperparameters = {
    'learning_rate': 0.001,
    'epochs': 1000,
    'batch_size': 2,
    'patience':8
}

```
Besides that you can choose also the data augmentations more suitable for you:

```python
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomApply([ transforms.RandomRotation(5)], p=0.25),
    transforms.RandomApply([transforms.RandomResizedCrop(256, scale=(0.7,0.9))], p=0.25),
    #transforms.GaussianBlur(kernel_size=5, sigma=(1.0,1.2))
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(1.0,1.2))], p=0.25)
    #AddGaussianNoise(mean=0.0, std=0.05)
])

```
And then you just have to run the [train.py](train.py) and the weights of the model will be saved in [models](\3dpericardialsegm\models) (you can change). The name of the model will be the date where the training started.