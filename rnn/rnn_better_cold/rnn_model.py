"""
RNN (GRU) model for Core Demand prediction.

Input per customer:
  - nace_code  : industry sector embedding
  - history    : sequence of (eclass, month_offset, log_qty, log_price)
  - fixed_fee  : monthly fee per Core Demand element (€)
  - saving_rate: fraction of spend saved if item is in Core Demand

Output per customer:
  - predicted eclass IDs likely to be bought in next 6 months
  - predicted month (1–6) for each eclass
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = "/home/parsa/Challenge2-20260306T182718Z-1-001/Challenge2"
OUT_DIR      = "/home/parsa/Challenge2-20260306T182718Z-1-001"
HISTORY_END  = pd.Timestamp("2025-07-01")      # split point
FIXED_FEE    = 10.0                             # € per month per element
SAVING_RATE  = 0.10                            # 10 % savings on purchases
MAX_SEQ_LEN  = 150                             # max historical events per customer
BATCH_SIZE   = 128
EPOCHS       = 15
LR           = 1e-3
TOP_K        = 10                              # max predictions per buyer
MIN_PROB     = 0.05                            # minimum sigmoid score to emit
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading training data …")
train = pd.read_csv(
    f"{DATA_DIR}/plis_training.csv",
    sep="\t", encoding="utf-8-sig",
    parse_dates=["orderdate"],
    dtype={"eclass": str, "nace_code": str},
    usecols=["orderdate", "legal_entity_id", "eclass",
             "quantityvalue", "vk_per_item",
             "estimated_number_employees", "nace_code"],
)
train = train.dropna(subset=["eclass", "nace_code"])

customers_test = pd.read_csv(
    f"{DATA_DIR}/customer_test.csv",
    sep="\t", encoding="utf-8-sig",
    dtype={"nace_code": str},
)
print(f"  Train rows : {len(train):,}")
print(f"  Test buyers: {len(customers_test):,}")

# ── 2. Encode categorical IDs ─────────────────────────────────────────────────
eclass_enc = LabelEncoder()
nace_enc   = LabelEncoder()

all_naces = pd.concat([
    train["nace_code"],
    customers_test["nace_code"].dropna()
]).unique()

eclass_enc.fit(train["eclass"].unique())
nace_enc.fit(all_naces)

N_ECLASS = len(eclass_enc.classes_)
N_NACE   = len(nace_enc.classes_)
print(f"  Unique eclasses : {N_ECLASS:,}")
print(f"  Unique NACE codes: {N_NACE:,}")

train["eclass_id"] = eclass_enc.transform(train["eclass"])
train["nace_id"]   = nace_enc.transform(train["nace_code"])

# ── 3. Train / target split ───────────────────────────────────────────────────
history_df = train[train["orderdate"] < HISTORY_END].copy()
target_df  = train[train["orderdate"] >= HISTORY_END].copy()

# Per-customer target: set of eclass_ids bought in last 6 months
target_map = (
    target_df.groupby("legal_entity_id")["eclass_id"]
             .apply(set)
             .to_dict()
)

# Mean price per eclass (for economic filtering at inference)
eclass_price = history_df.groupby("eclass_id")["vk_per_item"].mean().to_dict()

# ── Cold-start: NACE-sector eclass priors ─────────────────────────────────────
# For each NACE code, compute top eclasses ranked by (order_count, total_revenue).
# Used at inference for buyers with no purchase history.
print("Building NACE-sector cold-start priors …")

nace_eclass = (
    history_df.groupby(["nace_code", "eclass"])
              .agg(orders=("eclass_id", "count"), revenue=("vk_per_item", "sum"))
              .reset_index()
)
# Score: orders is the primary signal, revenue breaks ties
nace_eclass["score"] = nace_eclass["orders"] + nace_eclass["revenue"] / 1e6

# Map eclass string → eclass_id for later use
eclass_str_to_id = {v: k for k, v in enumerate(eclass_enc.classes_)}

# nace_prior: nace_code → list of (eclass_str, score) sorted descending
nace_prior: dict[str, list[tuple[str, float]]] = {}
for nace_code, grp in nace_eclass.groupby("nace_code"):
    ranked = grp.nlargest(TOP_K * 3, "score")[["eclass", "score"]].values.tolist()
    nace_prior[nace_code] = ranked   # [(eclass_str, score), ...]

# Global fallback: top eclasses across all training data
global_prior = (
    history_df.groupby("eclass")
              .agg(orders=("eclass_id", "count"), revenue=("vk_per_item", "sum"))
              .assign(score=lambda d: d["orders"] + d["revenue"] / 1e6)
              .nlargest(TOP_K * 3, "score")
              .reset_index()[["eclass", "score"]]
              .values.tolist()
)

print(f"  NACE codes with priors: {len(nace_prior):,}")

# ── 4. Build purchase sequences ───────────────────────────────────────────────
print("Building sequences …")

# Month offset from dataset start (integer months)
min_period = history_df["orderdate"].dt.to_period("M").min()

history_df = history_df.sort_values(["legal_entity_id", "orderdate"])
history_df["month_off"] = (
    history_df["orderdate"].dt.to_period("M") - min_period
).apply(lambda x: float(x.n))
history_df["log_qty"]   = np.log1p(history_df["quantityvalue"].clip(lower=0).fillna(0))
history_df["log_price"] = np.log1p(history_df["vk_per_item"].clip(lower=0).fillna(0))

# Group into sequences per customer (keep last MAX_SEQ_LEN events)
def build_seq(df):
    arr = df[["eclass_id", "month_off", "log_qty", "log_price"]].values
    return arr[-MAX_SEQ_LEN:]

sequences = {
    cid: build_seq(grp)
    for cid, grp in history_df.groupby("legal_entity_id")
}

# Company meta per customer
company_meta = (
    history_df.groupby("legal_entity_id")
              .agg(nace_id=("nace_id", "first"),
                   log_emp=("estimated_number_employees",
                            lambda x: float(np.log1p(x.fillna(1).iloc[0]))))
              .to_dict("index")
)

# ── 5. Dataset ────────────────────────────────────────────────────────────────
warm_customers = [c for c in sequences if c in target_map]
print(f"Warm-start training customers: {len(warm_customers):,}")

class PurchaseDataset(Dataset):
    def __init__(self, customer_ids):
        self.customers = customer_ids

    def __len__(self):
        return len(self.customers)

    def __getitem__(self, idx):
        cid  = self.customers[idx]
        seq  = sequences.get(cid, np.zeros((1, 4), dtype=np.float32))
        meta = company_meta.get(cid, {"nace_id": 0, "log_emp": 0.0})

        # Sequence tensors
        seq_t      = torch.tensor(seq, dtype=torch.float32)
        eclass_ids = seq_t[:, 0].long()
        month_offs = seq_t[:, 1] / 30.0   # normalise
        log_qtys   = seq_t[:, 2]
        log_prices = seq_t[:, 3]

        # Company features
        nace_id     = torch.tensor(meta["nace_id"], dtype=torch.long)
        log_emp     = torch.tensor(meta["log_emp"],  dtype=torch.float32)
        fixed_fee   = torch.tensor(FIXED_FEE,        dtype=torch.float32)
        saving_rate = torch.tensor(SAVING_RATE,      dtype=torch.float32)

        # Target: multi-hot vector over N_ECLASS
        target_vec = torch.zeros(N_ECLASS, dtype=torch.float32)
        for e in target_map.get(cid, set()):
            target_vec[e] = 1.0

        return dict(
            eclass_ids  = eclass_ids,
            month_offs  = month_offs,
            log_qtys    = log_qtys,
            log_prices  = log_prices,
            nace_id     = nace_id,
            log_emp     = log_emp,
            fixed_fee   = fixed_fee,
            saving_rate = saving_rate,
            target      = target_vec,
            seq_len     = len(seq),
        )


def collate_fn(batch):
    max_len = max(b["seq_len"] for b in batch)

    def pad(seqs, dtype, pad=0):
        out = torch.full((len(seqs), max_len), pad, dtype=dtype)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = s
        return out

    return dict(
        eclass_ids  = pad([b["eclass_ids"]  for b in batch], torch.long),
        month_offs  = pad([b["month_offs"]  for b in batch], torch.float32),
        log_qtys    = pad([b["log_qtys"]    for b in batch], torch.float32),
        log_prices  = pad([b["log_prices"]  for b in batch], torch.float32),
        nace_id     = torch.stack([b["nace_id"]     for b in batch]),
        log_emp     = torch.stack([b["log_emp"]     for b in batch]),
        fixed_fee   = torch.stack([b["fixed_fee"]   for b in batch]),
        saving_rate = torch.stack([b["saving_rate"] for b in batch]),
        target      = torch.stack([b["target"]      for b in batch]),
        seq_lens    = torch.tensor([b["seq_len"]    for b in batch]),
    )


# ── 6. Model ──────────────────────────────────────────────────────────────────
class PurchaseRNN(nn.Module):
    """
    GRU-based model.
    Input  : purchase history sequence + company context + economic params
    Output : eclass logits (multi-label) + date logits (6-class)
    """
    def __init__(self, n_eclass, n_nace,
                 eclass_dim=64, nace_dim=32, hidden=256, n_layers=2):
        super().__init__()
        self.eclass_emb = nn.Embedding(n_eclass + 1, eclass_dim, padding_idx=0)
        self.nace_emb   = nn.Embedding(n_nace   + 1, nace_dim,   padding_idx=0)

        # GRU over purchase events
        # each step: eclass_emb (64) + month_off, log_qty, log_price (3)
        self.gru = nn.GRU(
            input_size  = eclass_dim + 3,
            hidden_size = hidden,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = 0.3,
        )

        # Company context branch: nace_emb + log_emp + fixed_fee + saving_rate
        self.company_fc = nn.Sequential(
            nn.Linear(nace_dim + 3, 64),
            nn.ReLU(),
        )

        # Eclass prediction head
        self.eclass_head = nn.Sequential(
            nn.Linear(hidden + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_eclass),
        )

        # Date head: predict which of the 6 months the purchase falls in
        self.date_head = nn.Sequential(
            nn.Linear(hidden + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
        )

    def forward(self, eclass_ids, month_offs, log_qtys, log_prices,
                nace_id, log_emp, fixed_fee, saving_rate, seq_lens):

        # Sequence embedding
        e = self.eclass_emb(eclass_ids)                      # (B, T, 64)
        cont = torch.stack([month_offs, log_qtys, log_prices], dim=-1)  # (B, T, 3)
        x = torch.cat([e, cont], dim=-1)                     # (B, T, 67)

        packed = pack_padded_sequence(
            x, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        h = h_n[-1]                                          # (B, hidden)

        # Company context
        nace_e   = self.nace_emb(nace_id)                   # (B, 32)
        ctx = torch.cat([
            nace_e,
            log_emp.unsqueeze(1),
            fixed_fee.unsqueeze(1),
            saving_rate.unsqueeze(1),
        ], dim=1)                                            # (B, 35)
        company_h = self.company_fc(ctx)                    # (B, 64)

        combined = torch.cat([h, company_h], dim=1)         # (B, hidden+64)

        return self.eclass_head(combined), self.date_head(combined)


# ── 7. Training ───────────────────────────────────────────────────────────────
train_ids, val_ids = train_test_split(warm_customers, test_size=0.1, random_state=42)

train_dl = DataLoader(PurchaseDataset(train_ids), batch_size=BATCH_SIZE,
                      shuffle=True,  collate_fn=collate_fn, num_workers=4)
val_dl   = DataLoader(PurchaseDataset(val_ids),   batch_size=BATCH_SIZE,
                      shuffle=False, collate_fn=collate_fn, num_workers=4)

model     = PurchaseRNN(N_ECLASS, N_NACE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

# Heavy positive class weighting to handle class imbalance
pos_weight  = torch.full((N_ECLASS,), 20.0).to(DEVICE)
bce_loss    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
ce_loss     = nn.CrossEntropyLoss()


def run_epoch(dl, training=True):
    model.train() if training else model.eval()
    total = 0.0
    with torch.set_grad_enabled(training):
        for b in dl:
            to = lambda t: t.to(DEVICE)
            eclass_logits, date_logits = model(
                to(b["eclass_ids"]), to(b["month_offs"]),
                to(b["log_qtys"]),   to(b["log_prices"]),
                to(b["nace_id"]),    to(b["log_emp"]),
                to(b["fixed_fee"]),  to(b["saving_rate"]),
                b["seq_lens"],
            )
            loss = bce_loss(eclass_logits, to(b["target"]))
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item()
    return total / len(dl)


print(f"\nTraining on {DEVICE} …")
best_val = float("inf")
for epoch in range(1, EPOCHS + 1):
    tr = run_epoch(train_dl, training=True)
    vl = run_epoch(val_dl,   training=False)
    scheduler.step(vl)
    tag = " ◄ best" if vl < best_val else ""
    print(f"  Epoch {epoch:2d}/{EPOCHS}  train={tr:.4f}  val={vl:.4f}{tag}")
    if vl < best_val:
        best_val = vl
        torch.save(model.state_dict(), f"{OUT_DIR}/rnn_model.pt")

model.load_state_dict(torch.load(f"{OUT_DIR}/rnn_model.pt", map_location=DEVICE))
print("Best model restored.")

# ── 8. Inference ──────────────────────────────────────────────────────────────
print("\nPredicting for test customers …")
model.eval()

# Encode test customer nace codes (map unknowns to 0)
def encode_nace(code):
    if pd.isna(code) or code not in nace_enc.classes_:
        return 0
    return int(nace_enc.transform([code])[0])

PRED_MONTHS = [
    (HISTORY_END + pd.DateOffset(months=i)).strftime("%Y-%m")
    for i in range(6)
]

rows = []
with torch.no_grad():
    for _, cust in customers_test.iterrows():
        cid       = int(cust["legal_entity_id"])
        task      = cust.get("task", "warm start")
        nace_code = cust.get("nace_code", None)
        has_history = cid in sequences

        # ── Cold-start: use NACE-sector prior directly ────────────────────────
        if not has_history:
            candidates = nace_prior.get(nace_code, global_prior)
            count = 0
            for eclass_str, score in candidates:
                if count >= TOP_K:
                    break
                eid     = eclass_str_to_id.get(eclass_str)
                if eid is None:
                    continue
                price   = eclass_price.get(eid, 10.0)
                # Treat normalised rank score as a proxy probability
                prob    = min(score / (candidates[0][1] + 1e-9), 1.0)
                savings = SAVING_RATE * price * np.sqrt(float(prob) * 12)
                if savings < FIXED_FEE:
                    continue
                rows.append({
                    "buyer_id"            : cid,
                    "predicted_id"        : eclass_str,
                    "predicted_month"     : PRED_MONTHS[0],
                    "probability"         : round(float(prob), 4),
                    "expected_savings_eur": round(float(savings), 2),
                    "task"                : task,
                })
                count += 1
            continue

        # ── Warm-start: run the RNN model ─────────────────────────────────────
        seq  = sequences[cid]
        meta = company_meta.get(cid, {
            "nace_id": encode_nace(nace_code),
            "log_emp": float(np.log1p(cust.get("estimated_number_employees", 1) or 1)),
        })

        seq_arr    = torch.tensor(seq, dtype=torch.float32)
        eclass_ids = seq_arr[:, 0].long().unsqueeze(0).to(DEVICE)
        month_offs = (seq_arr[:, 1] / 30.0).unsqueeze(0).to(DEVICE)
        log_qtys   = seq_arr[:, 2].unsqueeze(0).to(DEVICE)
        log_prices = seq_arr[:, 3].unsqueeze(0).to(DEVICE)

        nace_t = torch.tensor([meta["nace_id"]], dtype=torch.long).to(DEVICE)
        emp_t  = torch.tensor([meta["log_emp"]],  dtype=torch.float32).to(DEVICE)
        fee_t  = torch.tensor([FIXED_FEE],         dtype=torch.float32).to(DEVICE)
        sav_t  = torch.tensor([SAVING_RATE],        dtype=torch.float32).to(DEVICE)
        slen_t = torch.tensor([len(seq)])

        e_logits, d_logits = model(
            eclass_ids, month_offs, log_qtys, log_prices,
            nace_t, emp_t, fee_t, sav_t, slen_t,
        )

        probs      = torch.sigmoid(e_logits)[0].cpu().numpy()
        date_probs = torch.softmax(d_logits, dim=-1)[0].cpu().numpy()

        top_idx = np.argsort(probs)[::-1]

        count = 0
        for idx in top_idx:
            if count >= TOP_K or probs[idx] < MIN_PROB:
                break
            price   = eclass_price.get(int(idx), 10.0)
            savings = SAVING_RATE * price * np.sqrt(float(probs[idx]) * 12)
            if savings < FIXED_FEE:
                continue

            eclass_name = eclass_enc.inverse_transform([idx])[0]
            month_label = PRED_MONTHS[int(np.argmax(date_probs))]

            rows.append({
                "buyer_id"            : cid,
                "predicted_id"        : eclass_name,
                "predicted_month"     : month_label,
                "probability"         : round(float(probs[idx]), 4),
                "expected_savings_eur": round(float(savings), 2),
                "task"                : task,
            })
            count += 1

submission = pd.DataFrame(rows)
out_path = f"{OUT_DIR}/submission_rnn.csv"
submission.to_csv(out_path, index=False)

print(f"\nSaved {out_path}")
print(f"  Rows   : {len(submission)}")
print(f"  Buyers : {submission['buyer_id'].nunique()}")
print(f"  Avg predictions / buyer: {len(submission)/submission['buyer_id'].nunique():.1f}")
print("\nSample output:")
print(submission.head(12).to_string(index=False))
