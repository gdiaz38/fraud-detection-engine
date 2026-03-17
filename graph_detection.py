import pandas as pd
import numpy as np
import networkx as nx
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = "/Users/gabrieldiaz/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/3/creditcard.csv"

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# ── Build transaction graph ───────────────────────────────────────────────────
# Since V1-V28 are anonymized PCA features we simulate graph structure
# using time windows + amount clustering — same approach used in production
# where merchant IDs and card IDs form the graph nodes

print("Building transaction graph...")

# Bin transactions into 10-minute windows
df['time_window'] = (df['Time'] // 600).astype(int)

# Bin amounts into clusters (proxy for merchant categories)
df['amount_bin'] = pd.cut(df['Amount'], bins=20, labels=False).fillna(0).astype(int)

# Create a bipartite-style graph:
# Node type A = time windows
# Node type B = amount bins
# Edge = a transaction connecting them
# Fraud rings show up as dense subgraphs in short time windows

G = nx.Graph()

# Sample 5000 transactions for graph (full graph = too large to visualize)
sample = df.sample(5000, random_state=42)

for _, row in sample.iterrows():
    time_node   = f"T_{int(row['time_window'])}"
    amount_node = f"A_{int(row['amount_bin'])}"
    is_fraud    = row['Class'] == 1

    G.add_node(time_node,   node_type='time',   fraud=is_fraud)
    G.add_node(amount_node, node_type='amount', fraud=is_fraud)
    G.add_edge(time_node, amount_node,
               weight=row['Amount'],
               fraud=int(is_fraud))

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── Graph features per transaction ───────────────────────────────────────────
print("Computing graph anomaly features...")

# For each transaction, compute:
# 1. Degree of its time window node (high = many transactions = suspicious)
# 2. Degree of its amount bin node
# 3. Clustering coefficient (how interconnected is its neighborhood)
# 4. PageRank (how central is this node in the transaction network)

pagerank = nx.pagerank(G, weight='weight')
clustering = nx.clustering(G)

def get_graph_features(row):
    time_node   = f"T_{int(row['time_window'])}"
    amount_node = f"A_{int(row['amount_bin'])}"

    time_degree   = G.degree(time_node)   if G.has_node(time_node)   else 0
    amount_degree = G.degree(amount_node) if G.has_node(amount_node) else 0
    time_pr       = pagerank.get(time_node, 0)
    amount_pr     = pagerank.get(amount_node, 0)
    time_clust    = clustering.get(time_node, 0)

    return pd.Series({
        'graph_time_degree':   time_degree,
        'graph_amount_degree': amount_degree,
        'graph_time_pagerank': time_pr,
        'graph_amt_pagerank':  amount_pr,
        'graph_clustering':    time_clust,
        'graph_anomaly_score': time_degree * time_pr * 1000
    })

print("Extracting graph features for all transactions (takes ~1 min)...")
df['time_window'] = (df['Time'] // 600).astype(int)
df['amount_bin']  = pd.cut(df['Amount'], bins=20,
                           labels=False).fillna(0).astype(int)

graph_features = df.apply(get_graph_features, axis=1)
df_with_graph  = pd.concat([df, graph_features], axis=1)

# ── Analyze graph features by fraud class ────────────────────────────────────
print("\n=== GRAPH FEATURES BY CLASS ===")
graph_cols = ['graph_time_degree','graph_amount_degree',
              'graph_time_pagerank','graph_anomaly_score']

for col in graph_cols:
    legit_mean = df_with_graph[df_with_graph['Class']==0][col].mean()
    fraud_mean = df_with_graph[df_with_graph['Class']==1][col].mean()
    print(f"  {col:<28} legit={legit_mean:.4f}  fraud={fraud_mean:.4f}")

# Save graph features
graph_feature_cols = ['graph_time_degree','graph_amount_degree',
                      'graph_time_pagerank','graph_amt_pagerank',
                      'graph_clustering','graph_anomaly_score']
df_with_graph[graph_feature_cols + ['Class']].to_csv("graph_features.csv", index=False)
joblib.dump(G, "transaction_graph.pkl")
joblib.dump(graph_feature_cols, "graph_feature_cols.pkl")
print("\n✅ Saved graph_features.csv and transaction_graph.pkl")

# ── Visualize a fraud subgraph ────────────────────────────────────────────────
print("Generating graph visualization...")
fraud_edges = [(u,v) for u,v,d in G.edges(data=True) if d.get('fraud')==1]
fraud_nodes = set([n for e in fraud_edges for n in e])

if fraud_nodes:
    subgraph   = G.subgraph(list(fraud_nodes)[:50])
    fig, axes  = plt.subplots(1, 2, figsize=(16, 6))

    pos = nx.spring_layout(subgraph, seed=42)
    node_colors = ['#ff4444' if 'T_' in n else '#ffa500'
                   for n in subgraph.nodes()]
    nx.draw_networkx(subgraph, pos, ax=axes[0],
                     node_color=node_colors,
                     node_size=300, font_size=7,
                     edge_color='#555', width=1.5)
    axes[0].set_title('Fraud Transaction Subgraph\n🔴=Time Window  🟠=Amount Cluster')
    axes[0].axis('off')

    # Graph anomaly score distribution
    axes[1].hist(df_with_graph[df_with_graph['Class']==0]['graph_anomaly_score'],
                 bins=50, alpha=0.6, color='#00d4aa', label='Legitimate')
    axes[1].hist(df_with_graph[df_with_graph['Class']==1]['graph_anomaly_score'],
                 bins=50, alpha=0.6, color='#ff4444', label='Fraud')
    axes[1].set_title('Graph Anomaly Score Distribution')
    axes[1].set_xlabel('Anomaly Score')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("graph_analysis.png", dpi=150)
    print("✅ Saved graph_analysis.png")
