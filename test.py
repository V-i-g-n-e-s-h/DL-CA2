from random import randrange
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

MODEL_PATH   = "final_model.h5"
PREPROC_PATH = "preprocess.pkl"
CSV_PATH     = "train.csv"
ROW_IDX      = 0  

model = load_model(MODEL_PATH)
with open(PREPROC_PATH, "rb") as f:
    prep = pickle.load(f)

num_min          = prep["num_min"]
num_max          = prep["num_max"]
encoders         = prep["encoders"]
numeric_cols     = prep["numeric_cols"]
categorical_cols = prep["categorical_cols"]

df = pd.read_csv(CSV_PATH)

TARGET    = "label"
DROP_COLS = ["fnlwgt", "education"]
df = df.drop(columns=DROP_COLS)
df.replace("?", "unknown", inplace=True)
df[TARGET] = df[TARGET].replace({"<=50K.": "<=50K", ">50K.": ">50K"})

ROW_IDX = randrange(len(df))
if ROW_IDX >= len(df) or ROW_IDX < 0:
    raise IndexError(f"ROW_IDX {ROW_IDX} out of range (dataset length {len(df)})")


raw_row = df.iloc[ROW_IDX].copy()
row     = df.iloc[[ROW_IDX]]

y_true = row[TARGET].values[0]

def scale(vals, vmin, vmax):
    return 2.0 * (vals - vmin) / (vmax - vmin) - 1.0

X_num = scale(row[numeric_cols], num_min, num_max).astype("float32")

X_cat = {}
for col in categorical_cols:
    enc = encoders[col]
    X_cat[col] = enc.transform(row[col].astype(str)).astype("int32")

inputs = {
    "num_in": X_num,
    **{f"{c}_in": X_cat[c] for c in categorical_cols}
}

y_prob = model.predict(inputs, verbose=0)[0, 0]
y_pred = ">50K" if y_prob > 0.5 else "<=50K"
status = "PASSED" if y_pred == y_true else "FAILED"

print("=" * 60)
print(f"Row index:           {ROW_IDX}")
print(f"Actual label:        {y_true}")
print(f"Predicted label:     {y_pred}   (prob={y_prob:.4f})")
print(f"Result:              {status}")
print("-" * 60)
print("Row snapshot:")
print(raw_row)
print("=" * 60)

"""
Output:

============================================================
Row index:           13256
Actual label:        <=50K
Predicted label:     <=50K   (prob=0.0259)
Result:              PASSED
------------------------------------------------------------
Row snapshot:
age                          67
workclass               Private
education_num                 9
marital_status         Divorced
occupation                Sales
relationship      Not-in-family
race                      White
sex                      Female
sex                      Female
capital_gain                  0
capital_loss                  0
hour_per_week                20
native_country    United-States
label                     <=50K
Name: 13256, dtype: object
============================================================
"""