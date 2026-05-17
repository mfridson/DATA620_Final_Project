"""
Standalone script version of the notebook. Useful for sanity checks and
for re-running the analysis without spinning up Jupyter.
"""
import os, re, json, warnings, random
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim import corpora, models
import community as community_louvain  # python-louvain
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

np.random.seed(42); random.seed(42)

ROOT = "/home/claude/data620_final"
os.makedirs(f"{ROOT}/figs", exist_ok=True)

# ------------------------------------------------------------
# 1. Load
# ------------------------------------------------------------
posts = pd.read_csv(f"{ROOT}/posts.csv")
comments = pd.read_csv(f"{ROOT}/comments.csv")
prices = pd.read_csv(f"{ROOT}/prices.csv")
print(f"comments: {len(comments):,}, posts: {len(posts):,}, prices: {len(prices):,}")

# ------------------------------------------------------------
# 2. Entity resolution: build an alias map for players + sets
# ------------------------------------------------------------
PLAYER_ALIAS = {
    "Cooper Flagg":      ["cooper flagg","flagg","cooper"],
    "Dylan Harper":      ["dylan harper","harper","dylan h"],
    "Victor Wembanyama": ["victor wembanyama","wembanyama","wemby","vw"],
    "Luka Doncic":       ["luka doncic","doncic","luka","lukers"],
    "Jayson Tatum":      ["jayson tatum","tatum"],
    "LeBron James":      ["lebron james","lebron","bron","lbj"],
    "Stephen Curry":     ["stephen curry","curry","steph"],
    "Paul Skenes":       ["paul skenes","skenes"],
    "Roman Anthony":     ["roman anthony","roman a","r anthony"],
    "Ronald Acuna":      ["ronald acuna","acuna"],
    "Shohei Ohtani":     ["shohei ohtani","ohtani","shohei"],
    "Mike Trout":        ["mike trout","trout"],
    "Bobby Witt":        ["bobby witt","witt","bobby w"],
    "Caitlin Clark":     ["caitlin clark","clark","caitlin"],
    "Paige Bueckers":    ["paige bueckers","bueckers","paige"],
}
SET_ALIAS = {
    "Bowman Chrome":   ["bowman chrome","bowman"],
    "Topps Chrome":    ["topps chrome","tc"],
    "Panini Prizm":    ["panini prizm","prizm"],
    "Bowman's Best":   ["bowmans best","bb","bowman's best"],
    "Topps Heritage":  ["topps heritage","heritage"],
    "Bowman U Now":    ["bowman u now","bu now","bowman u"],
    "National Treasures":["national treasures","nt"],
}
def compile_pat(d):
    out = {}
    for canon, al in d.items():
        # word-boundary, case-insensitive, longest alias first to avoid sub-matches
        al_sorted = sorted(al, key=len, reverse=True)
        out[canon] = re.compile(r"\b(" + "|".join(re.escape(a) for a in al_sorted) + r")\b",
                                flags=re.IGNORECASE)
    return out
P_PAT = compile_pat(PLAYER_ALIAS); S_PAT = compile_pat(SET_ALIAS)

def extract(text, pats):
    out = []
    for canon, pat in pats.items():
        if pat.search(text or ""):
            out.append(canon)
    return out

comments["players"] = comments["body"].apply(lambda t: extract(t, P_PAT))
comments["sets"]    = comments["body"].apply(lambda t: extract(t, S_PAT))
mention_rate = (comments["players"].str.len() > 0).mean()
print(f"player-mentioning share of comments: {mention_rate:.1%}")

# ------------------------------------------------------------
# 3. Network analysis: user reply graph (directed)
# ------------------------------------------------------------
G = nx.DiGraph()
edges = comments[["author","parent_author"]].dropna()
edges = edges[edges["author"] != edges["parent_author"]]  # drop self-replies
agg = edges.groupby(["author","parent_author"]).size().reset_index(name="weight")
for r in agg.itertuples(index=False):
    G.add_edge(r.author, r.parent_author, weight=r.weight)
print(f"reply graph: |V|={G.number_of_nodes():,}, |E|={G.number_of_edges():,}")

# Centralities. Eigenvector + betweenness on an undirected projection for stability.
Gu = G.to_undirected()
# Largest connected component for betweenness (faster, standard practice)
giant_nodes = max(nx.connected_components(Gu), key=len)
Gg = Gu.subgraph(giant_nodes).copy()
print(f"giant component: |V|={Gg.number_of_nodes():,}")

deg = dict(Gu.degree(weight="weight"))
# Approx betweenness with sampling for speed
bet = nx.betweenness_centrality(Gg, k=min(300, Gg.number_of_nodes()), seed=42, weight=None)
eig = nx.eigenvector_centrality_numpy(Gg, weight="weight")
cent = pd.DataFrame({
    "user": list(Gu.nodes()),
    "degree_w": [deg.get(u, 0) for u in Gu.nodes()],
    "betweenness": [bet.get(u, 0.0) for u in Gu.nodes()],
    "eigenvector": [eig.get(u, 0.0) for u in Gu.nodes()],
}).sort_values("eigenvector", ascending=False)
cent.to_csv(f"{ROOT}/centrality.csv", index=False)
print("top 5 by eigenvector:")
print(cent.head(5).to_string(index=False))

# Community detection (Louvain on undirected giant component)
part = community_louvain.best_partition(Gg, weight="weight", random_state=42)
mod = community_louvain.modularity(part, Gg, weight="weight")
n_comm = len(set(part.values()))
print(f"Louvain modularity Q={mod:.3f}, communities={n_comm}")

# ------------------------------------------------------------
# 4. Two-mode user x player graph + projection
# ------------------------------------------------------------
rows = []
for r in comments.itertuples(index=False):
    for p in r.players:
        rows.append((r.author, p))
up = pd.DataFrame(rows, columns=["user","player"])
B = nx.Graph()
B.add_nodes_from(up["user"].unique(), bipartite=0)
B.add_nodes_from(up["player"].unique(), bipartite=1)
ec = up.groupby(["user","player"]).size().reset_index(name="w")
for r in ec.itertuples(index=False):
    B.add_edge(r.user, r.player, weight=r.w)
player_nodes = set(up["player"].unique())
# Co-mention projection on players
P_proj = nx.bipartite.weighted_projected_graph(B, player_nodes)
print(f"bipartite: |V|={B.number_of_nodes():,}, |E|={B.number_of_edges():,}")
print(f"player co-mention graph: |V|={P_proj.number_of_nodes()}, |E|={P_proj.number_of_edges()}")

# ------------------------------------------------------------
# 5. Sentiment (VADER) + topic model (LDA, gensim)
# ------------------------------------------------------------
sia = SentimentIntensityAnalyzer()
comments["sent"] = comments["body"].apply(lambda t: sia.polarity_scores(str(t))["compound"])

sw = set(stopwords.words("english"))
def tokenize(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text).lower())
    return [t for t in text.split() if t not in sw and len(t) > 2]

# LDA on a sample for speed (still informative)
sample = comments.sample(n=min(20000, len(comments)), random_state=42)
docs = sample["body"].apply(tokenize).tolist()
dct = corpora.Dictionary(docs)
dct.filter_extremes(no_below=20, no_above=0.5)
corpus = [dct.doc2bow(d) for d in docs]
lda = models.LdaModel(corpus, id2word=dct, num_topics=6, passes=4, random_state=42)
print("LDA topics:")
for i, t in lda.print_topics(num_words=8):
    print(f"  T{i}: {t}")

# ------------------------------------------------------------
# 6. Weekly per-player time series of Reddit signal
# ------------------------------------------------------------
exp = comments.explode("players").dropna(subset=["players"])
weekly = (exp.groupby(["week","players"])
            .agg(mentions=("comment_id","count"),
                 sent_mean=("sent","mean"),
                 sent_pos_share=("sent", lambda s: (s>0.2).mean()))
            .reset_index()
            .rename(columns={"players":"player"}))
weekly.to_csv(f"{ROOT}/weekly_signal.csv", index=False)
print(f"weekly signal rows: {len(weekly):,}")

# ------------------------------------------------------------
# 7. Join with price series and check lead/lag
# ------------------------------------------------------------
df = weekly.merge(prices, on=["player","week"], how="left")
df = df.sort_values(["player","week"])
df["ret"]   = df.groupby("player")["price_index"].pct_change()
df["fwd1"]  = df.groupby("player")["ret"].shift(-1)  # next week return
df["men_z"] = df.groupby("player")["mentions"].transform(lambda x: (x - x.mean())/x.std(ddof=0))
df["sen_z"] = df.groupby("player")["sent_mean"].transform(lambda x: (x - x.mean())/x.std(ddof=0))

# Lead/lag corr at multiple offsets
def lead_lag(g, max_lag=3):
    out = {}
    for k in range(-max_lag, max_lag+1):
        a = g["men_z"]; b = g["ret"].shift(-k)
        out[k] = a.corr(b)
    return pd.Series(out)
ll = df.groupby("player").apply(lead_lag).reset_index().rename(columns={"level_1":"k"})
print("avg lead/lag corr (mentions vs returns), k>0 means Reddit leads price:")
print(df.groupby("player").apply(lead_lag).mean().round(3))

# ------------------------------------------------------------
# 8. Predictive model: does t Reddit signal predict t+1 return?
# ------------------------------------------------------------
mdf = df.dropna(subset=["fwd1","men_z","sen_z","ret"]).copy()
mdf["mom"] = mdf.groupby("player")["ret"].shift(0)  # current week ret as momentum
mdf = mdf.dropna(subset=["mom"])
X = mdf[["men_z","sen_z","mom"]].values
y = mdf["fwd1"].values

# Time-ordered split: train on first 70% of weeks, test on rest
order = mdf["week"].rank(method="dense").values
cut = np.quantile(order, 0.70)
tr = order <= cut; te = order > cut
model = Ridge(alpha=1.0).fit(X[tr], y[tr])
pred = model.predict(X[te])
r2 = r2_score(y[te], pred); mae = mean_absolute_error(y[te], pred)

# Baseline: predict next return = current return (momentum only)
pred_b = mdf.loc[te, "mom"].values
r2_b = r2_score(y[te], pred_b); mae_b = mean_absolute_error(y[te], pred_b)

# Zero baseline: predict 0
r2_0 = r2_score(y[te], np.zeros_like(y[te])); mae_0 = mean_absolute_error(y[te], np.zeros_like(y[te]))

print(f"\nHold-out evaluation (test rows={te.sum()}):")
print(f"  Ridge(men_z, sen_z, mom):   R2={r2:+.3f}  MAE={mae:.4f}")
print(f"  Momentum baseline:          R2={r2_b:+.3f}  MAE={mae_b:.4f}")
print(f"  Zero baseline:              R2={r2_0:+.3f}  MAE={mae_0:.4f}")
print(f"  Coefficients: {dict(zip(['men_z','sen_z','mom'], np.round(model.coef_,4)))}")

# TimeSeriesSplit CV for stability
tss = TimeSeriesSplit(n_splits=4)
cv_r2 = []
order_sorted_idx = np.argsort(order)
Xs, ys = X[order_sorted_idx], y[order_sorted_idx]
for tr_idx, te_idx in tss.split(Xs):
    m = Ridge(alpha=1.0).fit(Xs[tr_idx], ys[tr_idx])
    cv_r2.append(r2_score(ys[te_idx], m.predict(Xs[te_idx])))
print(f"  TimeSeriesSplit R2 across folds: {[round(x,3) for x in cv_r2]}")

# ------------------------------------------------------------
# 9. Save key figures
# ------------------------------------------------------------
# Fig 1: weekly mentions for top 5 players
top5 = (weekly.groupby("player")["mentions"].sum()
        .sort_values(ascending=False).head(5).index.tolist())
plt.figure(figsize=(9,4))
for p in top5:
    s = weekly[weekly["player"]==p].sort_values("week")
    plt.plot(s["week"], s["mentions"], label=p)
plt.xlabel("week"); plt.ylabel("mentions"); plt.title("Weekly mentions, top 5 players")
plt.legend(loc="upper right", fontsize=8); plt.tight_layout()
plt.savefig(f"{ROOT}/figs/weekly_mentions.png", dpi=120); plt.close()

# Fig 2: lead/lag heatmap
pivot = df.groupby("player").apply(lead_lag).round(3)
plt.figure(figsize=(7,5))
plt.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6)
plt.colorbar(label="corr(mentions, ret at k)")
plt.yticks(range(len(pivot)), pivot.index, fontsize=8)
plt.xticks(range(pivot.shape[1]), pivot.columns, fontsize=8)
plt.xlabel("k (>0 = Reddit leads price)"); plt.title("Lead/lag: mentions vs returns")
plt.tight_layout(); plt.savefig(f"{ROOT}/figs/lead_lag.png", dpi=120); plt.close()

# Fig 3: player co-mention graph
plt.figure(figsize=(8,6))
pos = nx.spring_layout(P_proj, seed=42, k=0.8)
weights = [P_proj[u][v]["weight"] for u,v in P_proj.edges()]
nx.draw_networkx_edges(P_proj, pos, alpha=0.3,
                       width=[w/max(weights)*4 for w in weights])
nx.draw_networkx_nodes(P_proj, pos, node_size=350, node_color="#4C72B0")
nx.draw_networkx_labels(P_proj, pos, font_size=8)
plt.title("Player co-mention graph (edge weight = shared commenters)")
plt.axis("off"); plt.tight_layout()
plt.savefig(f"{ROOT}/figs/player_comention.png", dpi=120); plt.close()

print("\nfigures saved to", f"{ROOT}/figs/")
print("done.")
