import pandas as pd
import numpy as np
import pytest

df1 = pd.read_csv("./tests/test_1.csv")
df2 = pd.read_csv("./tests/test_2.csv")


data_pt = []
for i in range(len(df1)):
    data_pt.append((i+2, df1.loc[i, "truth"], df1.loc[i, "prediction"], df2.loc[i, "prediction"]))


@pytest.mark.parametrize("a, b, c, d", data_pt)
def test_recovered(a,b,c,d):
	assert not(b!=c and b==d), a


@pytest.mark.parametrize("a,b,c,d",data_pt)
def test_failing(a,b,c,d):
	assert not(b==c and b!=d), a

"""
@pytest.mark.parametrize("a,b,c,d",data_pt)
def test_combine(a,b,c,d):
	if b==c and b!=d:
		print("Failing case at index:", a)
	elif b!=c and b==d:
		print("Recovered case at index:", a)
"""