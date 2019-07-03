## Description

 The python script data_extract.py its extracting the data from the MySQLdb as
python variables.
 The python script data_extract_to_xcels.py its extracting the table of the MySQLdb to an excel file for further processing in development mode.
 The python script data_prediction.py its procesing the data and determine the predictive models.

## Testing environment and scenario

There are two testing scenario's.

## First Scenario

This scenario collects the data from two phones into a MySQLdb of commmon radio parameters. one of the phone (UE1) is left into the best radio channel conditions of the LTE network while in the same time the second one (UE2) is moving between several rooms.


## Conclusion

TBD



## Sets of results:
 A set of results can be found on the src/results.

 The src/ directory strcuture is presented below:
```
.
├── data_extract.py
├── data_prediction.py
├── db_export.csv
├── db_export.xlsx
├── final_model.model
├── README.md
└── results
    ├── Figure_1_eNB.png
    ├── Figure_1.png
    ├── Figure_1_UE.png
    ├── Figure_2.png
    ├── Figure_3.png
    ├── Figure_4.png
    ├── Figure_5.png
    ├── Figure_6.png
    ├── Figure_7.png
    ├── Figure_8.png
    ├── Figure_9.png
    └── param_class_value.txt

1 directory, 18 files
```
 Note that the Figure_1_eNB.png, Figure_1.png and Figure_1_UE.png are representing the results of the corelation matrix. 
