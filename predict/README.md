# Make predictions with the 3D CNN

## Make predictions and at the same time calculate the Dice metric and see 2d visulatizations

If you have the Ground Truth, I recommend to follow this method.

First of all, ensure that the data is in the correct format, as indicated in the [Data](\3dpericardialsegm\data). Be sure that you have the CSV created and Pytorch installed in your device.

Then you have just to run the [predict_and_dice.py](predict_and_dice.py) script.
Does not forget to select the CSV and load the CSV of your data.

```python
"""Define Test sets""" 
"""Replace 'file_path.csv' with the path to your CSV file"""

#Choose the folds of the test set
folds_test=[4]

file_path = 'data\Abdominal_5.csv'

```
Load the model:
```python
"""Load the model, choose the path"""
model_path='models/Mon_Jul_22_19_40_20_2024/Mon_Jul_22_19_40_20_2024.pth'

model_dic=torch.load(model_path)
state_dic=model_dic['model_state_dict']
```
Choose the output folder:

```python
"""Choose where do you want to save the results"""
dataset='abd_test_csv'
path='data/Results'
path_results=os.path.join(path,dataset,model_path.split('/')[-2],Posprocess)
```
Run and the predictions will be in the "path_results"!

## Make predictions when do you not have the Ground Truth

If you just have the Dicom files, I recommend to run the [predict.py](predict.py) script. You don't need to create a CSV.

Otherwise, this script will create a header for the NRRD with the metadata retrieved directly from the Dicom files. 

You have to choose and load the model like the other method. Instead of using a CSV, now you just need to choose the input folder:

```python
"""Choose input folder"""
img_dir='data/DCM'

```
And of course the output folder like the other method.
Run and the predictions will be in the "path_results"!

