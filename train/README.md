# Train the model

The architecture of the CAN 3D is present in the [can3d_multi_noskip.py](can3d_multi_noskip.py) script. In order to train the model you just have to load your CSV and chose the folds for training and for validation.


```
"""Define train and val sets""" 

# Replace 'file_path.csv' with the path to your CSV file 
file_path_abd = 'C:/Users/RubenSilva/Desktop/CardiacCT_peri/CHVNGE/Abdominal_5.csv'

# Read the CSV file into a DataFrame
csv_file_abd = pd.read_csv(file_path_abd)
  
# Choose fold for train, val and test
folds_train=[1,2,0]
folds_val=[3]
folds_test=[4]

```