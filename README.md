## BSc Description
B.S.c Gheorghe Denisa

## Directory structure
```
.
├── Databases
│   ├── db_export.csv
│   └── stats_db.sql
├── Documentation
│   ├── 1.ijaerv10n87spl_89-DEC2015.pdf
│   └── mnm2017_paper6.pdf
├── README.md
└── src
    ├── data_extract.py
    ├── data_prediction.py
    ├── db_export.csv
    ├── db_export.xlsx
    ├── final_model.model
    ├── README.md
    └── results
        ├── Figure_1.png
        ├── Figure_2.png
        ├── Figure_3.png
        ├── Figure_4.png
        ├── Figure_5.png
        ├── Figure_6.png
        ├── Figure_7.png
        ├── Figure_8.png
        ├── Figure_9.png
        └── param_class_value.txt

4 directories, 21 files

```
For further development, please keep the directory structure!
On the Databases directory are uploaded just the MySQLDB and the microsoft csv export of itself

In the src/ can be found the python code, with the following attributes:

	- data_extract.py 	it extracts the DB into a csv file
	
	- data_prediction.py 	it computes the prediction for the LTE parameters
	
	- db_export.csv 	its the MYSQLDB extraction in csv format
	
	- db_export.xlsx	its the MYSQLDB extraction in xlsx format this will be used for further development of the code
	
	- final_model.model	its the development export of the prediction

## Description of the src/data_extract.py

The python code needs following extra plugins in order to facilitate the access to the MYSQLDB80:
1. pip install mysql-connector-python

The python code is supporting only Python3.7

## Online Bibliography
An interesting view of the random forest regression algorith choosed to be used in python it is presended on the link below:
1. https://medium.com/datadriveninvestor/random-forest-regression-9871bc9a25eb 

The used plugin sklearn:

2. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html


