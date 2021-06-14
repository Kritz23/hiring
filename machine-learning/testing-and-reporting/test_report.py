import pandas as pd
import pytest
import os
import glob
from sklearn.metrics import classification_report

files = glob.glob("./tests/test_*.csv")
data_inst = []

for i in files:
	data = pd.read_csv(i)
	y_true = data["truth"]
	y_pred = data["prediction"]
	print()
	print("------------------------------------------------------------------")
	print("Classification Report for file ",os.path.basename(i))
	print("------------------------------------------------------------------")
	print(classification_report(y_true, y_pred))
	for index, row in data.iterrows():
		data_inst.append((row["truth"],row["prediction"],os.path.basename(i)))

    
@pytest.mark.parametrize("a, b, c", data_inst)
def test_string(a,b,c):
    assert a == b, c
