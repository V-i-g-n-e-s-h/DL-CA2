import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras import layers, Model, Input, callbacks
from tensorflow.keras.optimizers import AdamW
import tensorflow as tf
from imblearn.over_sampling import SMOTENC

CSV_PATH          = "train.csv"
MODEL_OUT         = "final_model.h5" 
PREPROCESS_OUT    = "preprocess.pkl"    
MAX_EPOCHS        = 100
BATCH_SIZE        = 128 
LR                = 1e-3
WEIGHT_DECAY      = 1e-3                
SEED              = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

df = pd.read_csv(CSV_PATH)

TARGET       = "label"
DROP_COLS    = ["fnlwgt", "education"]
df = df.drop(columns=DROP_COLS)

df.replace("?", "unknown", inplace=True)
df["label"] = df["label"].replace({
    "<=50K.": "<=50K",
    ">50K.":  ">50K"
})

numeric_cols = ["age", "education_num", "capital_gain",
                "capital_loss", "hour_per_week"]
categorical_cols = [c for c in df.columns
                    if c not in numeric_cols + [TARGET]]

encoders = {}
for col in categorical_cols:
    le          = LabelEncoder()
    le.fit(df[col].astype(str))
    encoders[col] = le

num_min   = df[numeric_cols].min()
num_max   = df[numeric_cols].max()

def scale_numeric(values, vmin, vmax):
    return 2.0 * (values - vmin) / (vmax - vmin) - 1.0


def preprocess_dataframe(frame):
    num_scaled = scale_numeric(frame[numeric_cols], num_min, num_max).astype("float32")
    cat_ints = {}
    for col in categorical_cols:
        cat_ints[col] = encoders[col].transform(frame[col].astype(str)).astype("int32")
    return num_scaled, cat_ints

def build_model():
    num_input = Input(shape=(len(numeric_cols),), name="num_in")

    cat_inputs, emb_layers = [], []
    for col in categorical_cols:
        n_cat   = len(encoders[col].classes_)
        emb_dim = min(6, int(np.ceil(np.log2(n_cat))))

        inp = Input(shape=(1,), name=f"{col}_in")
        cat_inputs.append(inp)

        emb = layers.Embedding(n_cat, emb_dim, name=f"{col}_emb")(inp)
        emb_layers.append(layers.Flatten()(emb))

    x   = layers.Concatenate()( [num_input] + emb_layers )
    x   = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = Model(
            inputs=[num_input] + cat_inputs, 
            outputs=out,
            name="TinyTabularNet"
        )

    assert model.count_params() <= 1000, "Model too large!"

    opt = AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY)
    model.compile(
            optimizer=opt,
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")]
        )
    return model

X_num, X_cat = preprocess_dataframe(df)
y            = (df[TARGET] == ">50K").astype("int32").values

skf          = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
bal_accs     = []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros_like(y), y), 1):
    X_num_tr, X_num_va = X_num.iloc[train_idx], X_num.iloc[val_idx]
    y_tr, y_va = y[train_idx], y[val_idx]

    X_cat_tr = {c: X_cat[c][train_idx] for c in categorical_cols}
    X_cat_va = {c: X_cat[c][val_idx] for c in categorical_cols}

    X_tr_comb = np.hstack(
        [X_num_tr.to_numpy()] + [X_cat_tr[c].reshape(-1, 1) for c in categorical_cols]
    )

    n_num           = X_num_tr.shape[1]
    cat_feature_idx = list(range(n_num, n_num + len(categorical_cols)))

    smote = SMOTENC(
        categorical_features=cat_feature_idx,
        random_state=SEED,
    )
    X_res, y_res = smote.fit_resample(X_tr_comb, y_tr)

    X_num_tr = X_res[:, :n_num].astype("float32")
    X_cat_tr = {
        c: X_res[:, n_num + i].astype("int32")
        for i, c in enumerate(categorical_cols)
    }
    y_tr = y_res

    model = build_model()

    es = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=0
        )

    train_inputs = {"num_in": X_num_tr.astype("float32"), **{f"{c}_in": X_cat_tr[c] for c in categorical_cols}}
    val_inputs   = {"num_in": X_num_va.astype("float32"), **{f"{c}_in": X_cat_va[c] for c in categorical_cols}}

    model.fit(
        train_inputs, y_tr,
        validation_data=(val_inputs, y_va),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[es]
    )

    y_pred = (model.predict(val_inputs, batch_size=BATCH_SIZE, verbose=0) > 0.5).astype("int32")
    bal = balanced_accuracy_score(y_va, y_pred)
    bal_accs.append(bal)
    print(f"[Fold {fold}]  balanced acc = {bal:.4f}")

print("="*60)
print(f"10-fold mean balanced accuracy:  {np.mean(bal_accs):.4f} ± {np.std(bal_accs):.4f}")
print("="*60)

full_inputs = {"num_in": X_num.astype("float32"),
               **{f"{c}_in": X_cat[c] for c in categorical_cols}}
model_final = build_model()
model_final.fit(
        full_inputs, 
        y,
        validation_split=0.1,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=15,
                    restore_best_weights=True,
                    verbose=0
                )
            ],
        verbose=0)

model_final.save(MODEL_OUT) 
print(f" Saved final model ->  {MODEL_OUT}")

with open(PREPROCESS_OUT, "wb") as f:
    pickle.dump(
        {
            "num_min": num_min,
            "num_max": num_max,
            "encoders": encoders,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols
        }, 
        f
    )
print(f"Saved preprocess objects ->  {PREPROCESS_OUT}")

"""
Output:

[Fold 1]  balanced acc = 0.8216
[Fold 2]  balanced acc = 0.8200
[Fold 3]  balanced acc = 0.8262
[Fold 4]  balanced acc = 0.8304
[Fold 5]  balanced acc = 0.8123
[Fold 6]  balanced acc = 0.8234
[Fold 7]  balanced acc = 0.8168
[Fold 8]  balanced acc = 0.8333
[Fold 9]  balanced acc = 0.8140
[Fold 10]  balanced acc = 0.8045
============================================================
10-fold mean balanced accuracy:  0.8202 ± 0.0083
============================================================
Saved final model ->  final_model.h5
Saved preprocess objects ->  preprocess.pkl
"""