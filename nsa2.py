import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report


# 1. Load NSL-KDD Dataset

# Paths to the NSL-KDD train and test files
TRAIN_PATH = "KDDTrain+.txt"
TEST_PATH  = "KDDTest+.txt"

# The dataset files do not come with column names, so we define them here for later use.
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'attack', 'level'
]

# Read the CSV files and convert them to DataFrames
df_train = pd.read_csv(TRAIN_PATH, names=COLUMNS, sep=",")
df_test  = pd.read_csv(TEST_PATH,  names=COLUMNS, sep=",")

def print_dataset_info(df_train=pd.DataFrame, df_test=pd.DataFrame):
    print(f"Train shape: {df_train.shape}")
    print(f"Test  shape: {df_test.shape}")
    print(f"X_train columns: {df_train.columns}")
    print()

# 2. Create Binary Labels (0 = normal, 1 = attack)

# "attack" column contains the type of attack or 'normal' we convert it to a binary label: 0 = normal, 1 = any attack
df_train["binary_label"] = df_train["attack"].apply(lambda x: 0 if x == "normal" else 1)
df_test["binary_label"] = df_test["attack"].apply(lambda x: 0 if x == "normal" else 1)

# 3. Split Features (X) and Labels (y)
DROP_COLS = ["attack", "level"]
X_train = df_train.drop(columns=DROP_COLS + ["binary_label"])
y_train = df_train["binary_label"]
X_test  = df_test.drop(columns=DROP_COLS + ["binary_label"])
y_test  = df_test["binary_label"]

print("Initial dataset info:")
print_dataset_info(X_train, X_test)

# 4. Define Column Types
# Categorical columns that we will encode (embeddings)
EMBED_COLS = ["protocol_type", "service", "flag"]
BINARY_PASS_COLS = ['land', 'logged_in', 'root_shell', 'su_attempted','is_host_login', 'is_guest_login']

# 5. Reduce 'service' Categories
# Keeps the top-n most frequent services, and replaces all the others with a single category 'other'.
def reduce_col_service(df: pd.DataFrame, n: int) -> pd.DataFrame:
    top_services = df['service'].value_counts().index[:n]
    df = df.copy()
    df['service'] = df['service'].apply(lambda x: x if x in top_services else 'other')
    return df

X_train_reduced = reduce_col_service(X_train, n=10)
X_test_reduced  = reduce_col_service(X_test,  n=10)

print("After reducing 'service' categories:")
print_dataset_info(X_train_reduced, X_test_reduced)

# 6. Encode Categorical Columns (Simple Frequency Embedding)
# Encode a categorical column using normalized frequency (0-1).
def embed_categorical(train_df: pd.DataFrame,test_df: pd.DataFrame,col: str):
    """
    Idea:
    - Normalize frequencies to [0, 1].
    - Map each category to its normalized frequency.
    """
    freq = train_df[col].value_counts(normalize=True)
    freq = (freq - freq.min()) / (freq.max() - freq.min() + 1e-9) # Normalize frequencies to [0, 1]

    # Map each category to its frequency
    train_emb = train_df[col].map(freq).fillna(0.0)
    test_emb  = test_df[col].map(freq).fillna(0.0)

    return train_emb, test_emb

for col in EMBED_COLS:
    print(f"Embedding column: {col}")
    train_emb, test_emb = embed_categorical(X_train_reduced, X_test_reduced, col)

    X_train_reduced[col + "_emb"] = train_emb
    X_test_reduced[col + "_emb"]  = test_emb


# We will now drop the original categorical columns and keep only their embedded versions and numerical columns
# Binary colums will be dropped 
DROP_EMBED_COLS = EMBED_COLS + BINARY_PASS_COLS
X_train_embed = X_train_reduced.drop(columns=DROP_EMBED_COLS)
X_test_embed  = X_test_reduced.drop(columns=DROP_EMBED_COLS)

print("After embedding categorical columns:")
print_dataset_info(X_train_embed, X_test_embed)

# Redifining embedded cols
EMBEDDED_COLS = ['protocol_type_emb', 'service_emb', 'flag_emb']
# Columns to exclude when we apply MinMaxScaler
EXCLUDE_FROM_SCALING = BINARY_PASS_COLS + EMBEDDED_COLS
# Identify purely numerical columns to scale
NUMERIC_COLS = [col for col in X_train_embed.columns if col not in EXCLUDE_FROM_SCALING]

# 7. Process Numerical Columns (Min-Max Normalization)
def process_numerical(X_train: pd.DataFrame, X_test: pd.DataFrame, numeric_cols: list):
    """Apply Min-Max normalization to numeric columns."""
    scaler = MinMaxScaler()

    # Fit on training data and transform both train and test
    X_train_num = scaler.fit_transform(X_train[numeric_cols])
    X_test_num  = scaler.transform(X_test[numeric_cols])

    # Convert back to DataFrame with original indices
    X_train_num = pd.DataFrame(X_train_num, columns=numeric_cols, index=X_train.index)
    X_test_num = pd.DataFrame(X_test_num, columns=numeric_cols, index=X_test.index)

    return X_train_num, X_test_num

X_train_norm, X_test_norm = process_numerical(X_train_embed,X_test_embed,NUMERIC_COLS)

print("After normalizing numerical columns:")
print_dataset_info(X_train_norm, X_test_norm)

# 8. Build Final Feature Matrices
X_train_final = pd.concat([
        X_train_norm,
        X_train_embed.drop(columns=NUMERIC_COLS)
    ],axis=1)

X_test_final = pd.concat([
        X_test_norm,
        X_test_embed.drop(columns=NUMERIC_COLS)
    ],axis=1)

print("Final dataset info:")
print_dataset_info(X_train_final, X_test_final)


# 9. Negative Selection Algorithm (NSA) - V-Detector Functions
def generate_vdetectors(self_samples: np.ndarray, num_detectors: int, min_vals: np.ndarray,
    max_vals: np.ndarray, self_radius: float, max_tries: int,):
    """
    Generate V-detectors for the Negative Selection Algorithm (NSA) using Manhattan distance (L1).
    Returns a list of detectors.
        Each detector is a dict with keys: "radius": float, the radius up to the closest self sample (V-detector).
    """
    detectors = []
    dim = self_samples.shape[1]
    tries = 0

    while len(detectors) < num_detectors and tries < max_tries:
        tries += 1

        candidate = np.random.uniform(min_vals, max_vals, size=dim)

        # Compute Manhattan distance to all self samples
        dists = np.sum(np.abs(self_samples - candidate), axis=1)
        min_dist = np.min(dists)

        # Negative selection rule: accept only if far enough from self
        if min_dist > self_radius:
            detector_radius = float(min_dist)  # V-Detector: radius = distance to nearest self
            detectors.append(
                {
                    "center": candidate,
                    "radius": detector_radius
                }
            )

    print(f"[NSA] Detectors generated: {len(detectors)} (tries: {tries})")
    return detectors


def classify_with_detectors( X: np.ndarray, detectors: list) -> np.ndarray:
    """
    Classify samples using a set of V-detectors.
    Returns - np.ndarray, shape (N,)
        Predicted labels:
        - 0 = normal (self)
        - 1 = anomaly/attack (non-self)
    """
    n_samples = X.shape[0]
    y_pred = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        x = X[i]

        # If any detector "covers" the point, mark as attack (1)
        for d in detectors:
            dist = np.sum(np.abs(x - d["center"]))  # Manhattan distance
            if dist <= d["radius"]:
                y_pred[i] = 1
                break  # One activated detector is enough

    return y_pred



# 10. Prepare Data for NSA

# Use only normal class (0) as "self"
self_mask = (y_train == 0)
self_samples = X_train_final[self_mask].values

print(f"[NSA] Self samples (normal): {self_samples.shape[0]} samples,"
      f" {self_samples.shape[1]} dimensions.")

# Compute min and max values per feature (for detector sampling)
min_vals = X_train_final.min(axis=0).values
max_vals = X_train_final.max(axis=0).values
mean_vals = X_train_final.mean(axis=0).values
median_vals = X_train_final.median(axis=0).values

print("[NSA] min_vals (first 10):", min_vals[:10])
print("[NSA] max_vals (first 10):", max_vals[:10])
print("[NSA] mean_vals (first 10):", mean_vals[:10])
print("[NSA] median_vals (first 10):", median_vals[:10])
print()

# 11. Generate Detectors
num_detectors = 7500  # you can tune this (e.g. 500, 1000, 3000, ...)
self_radius  = 1.4   # important hyperparameter to adjust

detectors = generate_vdetectors(
    self_samples=self_samples,
    num_detectors=num_detectors,
    min_vals=min_vals,
    max_vals=max_vals,
    self_radius=self_radius,
    max_tries=160000
)

print(f"[NSA] Total detectors asked: {num_detectors}")
print(f"[NSA] Self Radius: {self_radius}")
print(f"[NSA] Total detectors generated: {len(detectors)}")
if len(detectors) > 0:
    radii = [d["radius"] for d in detectors]
    print(f"[NSA] Radius stats -> mean: {np.mean(radii):.4f},"
          f" min: {np.min(radii):.4f}, max: {np.max(radii):.4f}")
print()



# 12. Classification and Evaluation

# Convert X_test_final to numpy array
X_test_array = X_test_final.values

# Classify test samples using the V-detectors
y_pred = classify_with_detectors(X_test_array, detectors)

# Confusion matrix
print("\n[NSA] Confusion Matrix (y_test vs y_pred):")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Detailed classification report
print("\n[NSA] Classification Report:")
print(classification_report(y_test, y_pred, digits=4))