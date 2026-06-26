import argparse
import json
import networkx as nx
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Visualize a KG-RAG knowledge graph JSON file.")
parser.add_argument("graph", help="Path to the graph JSON file")
args = parser.parse_args()

with open(args.graph, "r", encoding="utf-8") as f:
    data = json.load(f)

G = nx.DiGraph()

for node_id, node_info in data["nodes"].items():
    G.add_node(node_id, label=node_info["label"], type=node_info["type"])


for edge in data["edges"]:
    G.add_edge(edge["source"], edge["target"], relation=edge.get("relation", ""), weight=edge.get("weight", 1.0))


pos = nx.spring_layout(G, k=0.5, iterations=50)


color_map = []
for node in G.nodes(data=True):
    if node[1]["type"] == "semantic":
        color_map.append("skyblue")
    else:
        color_map.append("lightgreen")


nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=800)


nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='gray')


labels = {node[0]: node[1]["label"] for node in G.nodes(data=True)}
nx.draw_networkx_labels(G, pos, labels, font_size=8)


edge_labels = {(e['source'], e['target']): e['relation'] for e in data["edges"]}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

plt.title("Knowledge Graph Visualization")
plt.axis('off')
plt.tight_layout()
plt.show()