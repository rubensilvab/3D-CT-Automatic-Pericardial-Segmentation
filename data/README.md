### Dataset format

In order to run the Can 3D we have to organize the data according to this:

```plaintext
 NameDataset (Data)/
│    │── Dicom (DCM)/
│    │    │── Patient (xxx)
│    │    │    ├── 37559456 (or .dcm)
│    │    │    ├──37559467
│    │    │    ├──...NrSlice
│    │── Mask (Peri_segm)/
│    │    ├── Patient.nrrd (xxx.nrrd)

```

## CSV Creation for the Dataset 

A CSV must be created for both the training and prediction of the model. Example:[Abdominal_5.csv](Abdominal_5.csv)

-  NameDataset.CSV - including all relevant information to facilitate the training process
    - Patient id
    - The fold designation for each patient (if we want a cross validation)
    - Path of the Dicom files (DCM)
    - Path of the masks (Peri_Segm)
    - Image size (x,y)
    - Number of Slices of the CT Scan

The script [create_csv.py](create_csv.py) can help you to create this csv.
                    
    