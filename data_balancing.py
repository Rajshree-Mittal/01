import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from collections import Counter

# =========================================================
# 1 LOAD MULTIPLE CSV FILES
# =========================================================

DATA_FOLDER = "bot-iot"

csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

print("CSV files found:", len(csv_files))

dfs = []

for file in csv_files:

    try:
        part = pd.read_csv(file, low_memory=False)
        print("Loaded:", file, part.shape)
        dfs.append(part)

    except Exception as e:
        print("Skipped:", file, e)

df = pd.concat(dfs, ignore_index=True, sort=False)

print("\nCombined dataset shape:", df.shape)


# =========================================================
# 2 LABEL CREATION
# =========================================================

label_candidates = ["label","attack","class","target"]
attack_type_candidates = ["attack_cat","attack_type","category","type"]

label_col = None
attack_type_col = None

for c in df.columns:

    if c.lower() in label_candidates:
        label_col = c

    if c.lower() in attack_type_candidates:
        attack_type_col = c

if label_col is None:
    label_col = df.columns[-1]

print("\nBinary label column:", label_col)

if attack_type_col:
    print("Multiclass column:", attack_type_col)
else:
    print("Using binary label column for multiclass")


# ================= BINARY LABEL =================

normal_keywords = ["normal","benign","background"]

y_binary = np.array([
    0 if any(k in str(v).lower() for k in normal_keywords)
    else 1
    for v in df[label_col]
])


# ================= MULTICLASS LABEL =================

if attack_type_col is not None:
    y_multi_raw = df[attack_type_col]
else:
    y_multi_raw = df[label_col]

encoder = LabelEncoder()
y_multi = encoder.fit_transform(y_multi_raw)

print("\nMulticlass classes:", encoder.classes_)

print("\nBinary distribution:", Counter(y_binary))
print("Multiclass distribution:", Counter(y_multi))


# =========================================================
# 3 FEATURE PREPROCESSING
# =========================================================

drop_keywords = ["id","flow","timestamp","time","srcip","dstip","ip"]

drop_cols = [
    c for c in df.columns
    if any(k in c.lower() for k in drop_keywords)
]

X = df.drop(columns=drop_cols, errors="ignore")

X = X.select_dtypes(include=np.number)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# =========================================================
# 4 BINARY BALANCING
# =========================================================

def balance_binary(X,y):

    counts = Counter(y)

    max_count = max(counts.values())

    X_syn=[]
    y_syn=[]

    for cls in counts:

        need = max_count - counts[cls]

        if need<=0:
            continue

        X_cls = X[y==cls]

        for _ in range(need):

            idx = np.random.randint(len(X_cls))

            noise = np.random.normal(0,0.01,X_cls.shape[1])

            sample = X_cls[idx] + noise

            X_syn.append(sample)
            y_syn.append(cls)

    if len(X_syn)==0:
        return X,y

    X_syn=np.array(X_syn)
    y_syn=np.array(y_syn)

    X_bal=np.vstack([X,X_syn])
    y_bal=np.hstack([y,y_syn])

    return X_bal,y_bal


X_bin_bal,y_bin_bal = balance_binary(X_scaled,y_binary)

print("\nBinary distribution after balancing:",Counter(y_bin_bal))


repeat_factor = int(np.ceil(len(y_bin_bal)/len(y_multi)))
y_multi_bal = np.tile(y_multi, repeat_factor)[:len(y_bin_bal)]


# =========================================================
# 5 PIDA AUGMENTATION
# =========================================================

def PIDA(X,y,k=5):

    counts = Counter(y)

    max_count = max(counts.values())

    X_syn=[]
    y_syn=[]

    for cls in counts:

        if cls == 0:
            continue

        ratio = counts[cls]/max_count

        if ratio > 0.3:
            continue

        print("Applying PIDA to attack class:",cls)

        X_cls = X[y==cls]

        nbrs = NearestNeighbors(
            n_neighbors=min(k,len(X_cls))
        ).fit(X_cls)

        for i in range(len(X_cls)):

            neighbors = nbrs.kneighbors(
                [X_cls[i]],
                return_distance=False
            )[0]

            for _ in range(3):

                neighbor = X_cls[np.random.choice(neighbors)]

                alpha = np.random.rand()

                synthetic = X_cls[i] + alpha*(neighbor-X_cls[i])

                noise = np.random.normal(0,0.02,synthetic.shape)

                synthetic = synthetic + noise

                X_syn.append(synthetic)
                y_syn.append(cls)

    if len(X_syn)==0:
        return X,y

    X_syn=np.array(X_syn)
    y_syn=np.array(y_syn)

    X_aug = np.vstack([X,X_syn])
    y_aug = np.hstack([y,y_syn])

    return X_aug,y_aug


X_aug,y_aug_multi = PIDA(X_bin_bal,y_multi_bal)

y_aug_binary = np.where(y_aug_multi==0,0,1)


# =========================================================
# 6 TRUE CONTRASTIVE LEARNING (SimCLR)
# =========================================================

class Encoder(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim,256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256,128),
            nn.ReLU(),

            nn.Linear(128,64)
        )

    def forward(self,x):
        return self.network(x)


def augment(x):

    noise = torch.randn_like(x)*0.02
    return x + noise


def contrastive_loss(z1,z2,temperature=0.5):

    batch_size = z1.shape[0]

    z1 = nn.functional.normalize(z1,dim=1)
    z2 = nn.functional.normalize(z2,dim=1)

    representations = torch.cat([z1,z2],dim=0)

    similarity = torch.matmul(representations,representations.T)

    labels = torch.arange(batch_size).to(z1.device)
    labels = torch.cat([labels,labels],dim=0)

    mask = torch.eye(labels.shape[0],dtype=torch.bool).to(z1.device)

    similarity = similarity / temperature

    similarity = similarity.masked_fill(mask,-9e15)

    positives = torch.cat([
        torch.diag(similarity,batch_size),
        torch.diag(similarity,-batch_size)
    ])

    negatives = similarity[~mask].view(2*batch_size,-1)

    logits = torch.cat([positives.unsqueeze(1),negatives],dim=1)

    labels = torch.zeros(2*batch_size,dtype=torch.long).to(z1.device)

    loss = nn.CrossEntropyLoss()(logits,labels)

    return loss


encoder_model = Encoder(X_aug.shape[1])

optimizer = optim.Adam(encoder_model.parameters(),lr=0.001)

X_tensor = torch.tensor(X_aug,dtype=torch.float32)

epochs=50
batch_size=256

for epoch in range(epochs):

    perm = torch.randperm(X_tensor.size(0))

    for i in range(0,X_tensor.size(0),batch_size):

        idx = perm[i:i+batch_size]

        batch = X_tensor[idx]

        x1 = augment(batch)
        x2 = augment(batch)

        z1 = encoder_model(x1)
        z2 = encoder_model(x2)

        loss = contrastive_loss(z1,z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch",epoch,"Loss",loss.item())


with torch.no_grad():

    embeddings = encoder_model(X_tensor).numpy()


# =========================================================
# 7 FEATURE FUSION
# =========================================================

X_fused = np.concatenate([X_aug,embeddings],axis=1)

print("\nFeature shape after fusion:",X_fused.shape)


# =========================================================
# 8 MULTICLASS BALANCING
# =========================================================

def balance_multiclass(X,y):

    counts = Counter(y)

    max_count = max(counts.values())

    X_syn=[]
    y_syn=[]

    for cls in counts:

        if cls==0:
            continue

        need = max_count-counts[cls]

        if need<=0:
            continue

        X_cls = X[y==cls]

        for _ in range(need):

            idx = np.random.randint(len(X_cls))

            noise = np.random.normal(0,0.01,X_cls.shape[1])

            sample = X_cls[idx]+noise

            X_syn.append(sample)
            y_syn.append(cls)

    if len(X_syn)==0:
        return X,y

    X_syn=np.array(X_syn)
    y_syn=np.array(y_syn)

    X_final=np.vstack([X,X_syn])
    y_final=np.hstack([y,y_syn])

    return X_final,y_final


X_final,y_final_multi = balance_multiclass(X_fused,y_aug_multi)

y_final_binary = np.where(y_final_multi==0,0,1)


# =========================================================
# 9 FINAL BINARY REBALANCE
# =========================================================

normal_idx = np.where(y_final_binary==0)[0]
attack_idx = np.where(y_final_binary==1)[0]

needed = len(attack_idx) - len(normal_idx)

if needed>0:

    syn_X=[]
    syn_y=[]

    for _ in range(needed):

        idx = np.random.choice(normal_idx)

        noise=np.random.normal(0,0.01,X_final.shape[1])

        syn_X.append(X_final[idx]+noise)
        syn_y.append(0)

    X_final=np.vstack([X_final,np.array(syn_X)])
    y_final_multi=np.hstack([y_final_multi,np.array(syn_y)])

y_final_binary = np.where(y_final_multi==0,0,1)

print("\nFinal multiclass distribution:",Counter(y_final_multi))
print("Final binary distribution:",Counter(y_final_binary))


# =========================================================
# 10 SAVE FINAL DATASET
# =========================================================

final_df = pd.DataFrame(X_final)

final_df["multi_label"] = y_final_multi
final_df["binary_label"] = y_final_binary

final_df.to_csv(f"{DATA_FOLDER}_final.csv",index=False)

print("Saved final dataset")