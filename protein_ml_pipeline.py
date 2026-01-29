import pandas as pd
import random

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Load dataset
# -----------------------------
DATA_PATH = "/Users/rhealalwani/Downloads/PP1_ProteinFolding_Dataset.xlsx"

df = pd.read_excel(DATA_PATH, sheet_name="S2")
df.columns = ["Protein_ID", "Sequence", "Label"]

# -----------------------------
# Sequence cleaning
# -----------------------------
VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")

def clean_sequence(seq):
    return "".join([aa for aa in str(seq) if aa in VALID_AAS])

df["Sequence"] = df["Sequence"].apply(clean_sequence)
df = df[df["Sequence"].str.len() > 0]

# -----------------------------
# ProtParam feature extraction
# -----------------------------
def extract_protein_features(seq):
    analysis = ProteinAnalysis(seq)
    return pd.Series({
        "Isoelectric_Point": analysis.isoelectric_point(),
        "Aromaticity": analysis.aromaticity(),
        "Instability_Index": analysis.instability_index()
    })

protparam_df = df["Sequence"].apply(extract_protein_features)
df_pp = df.loc[protparam_df.index]
df_pp = pd.concat([df_pp, protparam_df], axis=1)

# -----------------------------
# Train model (ProtParam)
# -----------------------------
X_pp = df_pp[["Isoelectric_Point", "Aromaticity", "Instability_Index"]]
y = df_pp["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X_pp, y, test_size=0.2, random_state=42
)

model_pp = RandomForestClassifier(random_state=42)
model_pp.fit(X_train, y_train)
preds_pp = model_pp.predict(X_test)

pp_accuracy = accuracy_score(y_test, preds_pp)
print("\nProtParam accuracy:", pp_accuracy)

# -----------------------------
# Amino Acid Composition (AAC)
# -----------------------------
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

def compute_aac(seq):
    length = len(seq)
    return pd.Series({aa: seq.count(aa) / length for aa in AA_LIST})

aac_df = df["Sequence"].apply(compute_aac)

X_train, X_test, y_train, y_test = train_test_split(
    import pandas as pd
import random

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Load dataset
# -----------------------------
DATA_PATH = "/Users/rhealalwani/Downloads/PP1_ProteinFolding_Dataset.xlsx"

df = pd.read_excel(DATA_PATH, sheet_name="S2")
df.columns = ["Protein_ID", "Sequence", "Label"]

# -----------------------------
# Sequence cleaning
# -----------------------------
VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")

def clean_sequence(seq):
    return "".join([aa for aa in str(seq) if aa in VALID_AAS])

df["Sequence"] = df["Sequence"].apply(clean_sequence)
df = df[df["Sequence"].str.len() > 0]

# -----------------------------
# ProtParam feature extraction
# -----------------------------
def extract_protein_features(seq):
    analysis = ProteinAnalysis(seq)
    return pd.Series({
        "Isoelectric_Point": analysis.isoelectric_point(),
        "Aromaticity": analysis.aromaticity(),
        "Instability_Index": analysis.instability_index()
    })

protparam_df = df["Sequence"].apply(extract_protein_features)
df_pp = df.loc[protparam_df.index]
df_pp = pd.concat([df_pp, protparam_df], axis=1)

# -----------------------------
# Train model (ProtParam)
# -----------------------------
X_pp = df_pp[["Isoelectric_Point", "Aromaticity", "Instability_Index"]]
y = df_pp["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X_pp, y, test_size=0.2, random_state=42
)

model_pp = RandomForestClassifier(random_state=42)
model_pp.fit(X_train, y_train)
preds_pp = model_pp.predict(X_test)

pp_accuracy = accuracy_score(y_test, preds_pp)
print("\nProtParam accuracy:", pp_accuracy)

# -----------------------------
# Amino Acid Composition (AAC)
# -----------------------------
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

def compute_aac(seq):
    length = len(seq)
    return pd.Series({aa: seq.count(aa) / length for aa in AA_LIST})

aac_df = df["Sequence"].apply(compute_aac)

X_train, X_test, y_train, y_test = train_test_split(
    aac_df, df["Label"], test_size=0.2, random_state=42
)

model_aac = RandomForestClassifier(random_state=42)
model_aac.fit(X_train, y_train)
preds_aac = model_aac.predict(X_test)

aac_accuracy = accuracy_score(y_test, preds_aac)
print("AAC accuracy:", aac_accuracy)

# -----------------------------
# Mutation simulation (C)
# -----------------------------
def mutate_sequence(seq):
    idx = random.randint(0, len(seq) - 1)
    new_aa = random.choice(AA_LIST)
    while new_aa == seq[idx]:
        new_aa = random.choice(AA_LIST)
    return seq[:idx] + new_aa + seq[idx + 1:], idx, new_aa

original_seq = df["Sequence"].iloc[0]
original_features = extract_protein_features(original_seq)

results = []

for _ in range(10):
    mut_seq, pos, aa = mutate_sequence(original_seq)
    mut_feat = extract_protein_features(mut_seq)
    diff = mut_feat - original_features

    results.append({
        "Position": pos,
        "New_AA": aa,
        "Δ_pI": diff["Isoelectric_Point"],
        "Δ_Aromaticity": diff["Aromaticity"],
        "Δ_Instability": diff["Instability_Index"]
    })

mutation_df = pd.DataFrame(results)

print("\nMutation effects:")
print(mutation_df)
