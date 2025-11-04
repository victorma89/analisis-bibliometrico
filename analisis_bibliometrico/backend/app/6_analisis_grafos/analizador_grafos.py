# analizador_citaciones.py
import bibtexparser
from pathlib import Path
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# ---------- Utilities ----------
def normalize(s):
    return (s or "").strip()

def parse_authors(entry):
    auth = entry.get('author', '') or ''
    if not auth:
        return []
    return [a.strip() for a in auth.replace('\n', ' ').split(' and ') if a.strip()]

def parse_keywords(entry):
    keys = entry.get('keywords', '') or entry.get('keyword', '') or ''
    if not keys:
        return []
    return [k.strip() for k in keys.replace(';', ',').split(',') if k.strip()]

def entry_to_dict(entry):
    return {
        'id': entry.get('ID'),
        'title': normalize(entry.get('title', '')),
        'authors': parse_authors(entry),
        'keywords': parse_keywords(entry),
        'year': entry.get('year', None),
        'raw': entry
    }

# ---------- Load bib ----------
def load_bib_file(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe: {p}")
    with open(p, encoding='utf-8') as f:
        bib_database = bibtexparser.load(f)
    articles = [entry_to_dict(e) for e in bib_database.entries]
    print(f"✅ Cargados {len(articles)} artículos desde {p}")
    return articles

# ---------- Build graph (optimized) ----------
def infer_graph(articles, thr=0.25, verbose=False):
    """
    Construye un grafo dirigido. Para eficiencia:
     - concatenamos title + keywords + authors en un solo texto por artículo
     - TF-IDF sobre esos textos -> matriz de similitud por cosine
     - añadimos arista i->j si sim >= thr
    """
    G = nx.DiGraph()
    if not articles:
        return G

    for a in articles:
        G.add_node(a['id'],
                   title=a.get('title', ''),
                   authors=a.get('authors', []),
                   keywords=a.get('keywords', []),
                   year=a.get('year', None))

    # Prepare combined text corpus
    docs = []
    for a in articles:
        parts = []
        if a.get('title'): parts.append(a['title'])
        if a.get('keywords'): parts.append(" ".join(a['keywords']))
        if a.get('authors'): parts.append(" ".join(a['authors']))
        docs.append(" ||| ".join(parts) if parts else "")

    # TF-IDF + cosine similarity
    try:
        vec = TfidfVectorizer(max_features=20000, stop_words='english')
        X = vec.fit_transform(docs)  # shape (n, m)
        sim_matrix = cosine_similarity(X)  # numpy array
    except Exception as e:
        # Fallback: fuzzy match
        n = len(articles)
        sim_matrix = np.zeros((n, n), dtype=float)
        titles = [a.get('title', '') for a in articles]
        for i in range(n):
            for j in range(n):
                if i == j: continue
                sim_matrix[i, j] = fuzz.token_sort_ratio(titles[i], titles[j]) / 100.0

    # Add edges where similarity >= thr
    n = len(articles)
    ids = [a['id'] for a in articles]
    inds = np.where(sim_matrix >= thr)
    for i, j in zip(*inds):
        if i == j:
            continue
        u = ids[i]; v = ids[j]
        w = float(sim_matrix[i, j])
        G.add_edge(u, v, weight=w)

    # --- 👇 AGREGA ESTE BLOQUE ABAJO (justo antes del return) ---
    print(f"📈 Grafo inferido con {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas (thr={thr})")

    # 🔗 Mostrar ejemplos de conexiones directas
    ejemplos = list(G.edges(data=True))[:10]
    print("\n🔗 Ejemplos de conexiones (primeras 10):")
    for u, v, data in ejemplos:
        print(f"{u} → {v} (peso={data['weight']:.3f})")

    # 🛣️ Buscar un camino indirecto (más de 1 salto)
    for nodo in list(G.nodes())[:50]:
        paths = nx.single_source_shortest_path(G, nodo)
        for destino, camino in paths.items():
            if len(camino) > 2:
                print(f"🛣️ Camino posible {nodo} → {destino}: {camino}")
                break
    # --- 👆 FIN DEL BLOQUE NUEVO ---

    return G


# ---------- Export JSON ----------
def export_graph_to_json(G):
    nodes = [{"id": n, "label": G.nodes[n].get("title", "") or str(n),
              "authors": G.nodes[n].get("authors", []),
              "year": G.nodes[n].get("year", None)}
             for n in G.nodes()]
    edges = [{"from": u, "to": v, "weight": float(d.get("weight", 0.0))}
             for u, v, d in G.edges(data=True)]
    return {"nodes": nodes, "edges": edges}

# ---------- Analysis ----------
def analizar_grafo(G):
    info = {
        "num_nodos": int(G.number_of_nodes()),
        "num_aristas": int(G.number_of_edges()),
        "densidad": float(nx.density(G)),
        "grado_medio": float(sum(dict(G.degree()).values()) / G.number_of_nodes()) if G.number_of_nodes() else 0.0,
        "componentes_fuertes": int(nx.number_strongly_connected_components(G)),
        "componentes_debiles": int(nx.number_weakly_connected_components(G))
    }

    # 🔹 NUEVO BLOQUE: identificar SCC (componentes fuertemente conexas)
    try:
        componentes_fuertes = list(nx.strongly_connected_components(G))
        componentes_fuertes_sorted = sorted(componentes_fuertes, key=len, reverse=True)
        info["componentes_fuertemente_conexas"] = {
            "total": len(componentes_fuertes_sorted),
            "mayores": [list(c) for c in componentes_fuertes_sorted[:3]]
        }
    except Exception as e:
        info["componentes_fuertemente_conexas"] = {"total": 0, "mayores": []}

    # … (resto del código existente)
    try:
        deg_cent = nx.degree_centrality(G)
        if deg_cent:
            top_deg = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:10]
            info["top_degree_centrality"] = [(n, float(v)) for n, v in top_deg]
        else:
            info["top_degree_centrality"] = []
    except Exception:
        info["top_degree_centrality"] = []

    try:
        pr = nx.pagerank(G)
        top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]
        info["top_pagerank"] = [(n, float(v)) for n, v in top_pr]
    except Exception:
        info["top_pagerank"] = []

    return info


def identificar_componentes_fuertemente_conexas(G):
    """
    Retorna las componentes fuertemente conexas (SCC)
    ordenadas por tamaño, con resumen de información.
    """
    sccs = list(nx.strongly_connected_components(G))
    componentes = []

    for i, comp in enumerate(sorted(sccs, key=len, reverse=True), 1):
        subgrafo = G.subgraph(comp)
        num_nodos = len(comp)
        if num_nodos > 0:
            # Muestra de artículos (máximo 5)
            muestra = []
            for n in list(comp)[:5]: # Iterar sobre una porción de los nodos del componente
                node_data = G.nodes[n]
                muestra.append({
                    "id": n,
                    "title": node_data.get("title", "")[:60],
                    "citaciones": int(node_data.get("citaciones", 0)),
                    "year": node_data.get("year")
                })

            componentes.append({
                "id": i,
                "num_nodos": num_nodos,
                "nodos": list(comp),
                "num_aristas": subgrafo.number_of_edges(),
                "densidad": nx.density(subgrafo),
                "grado_promedio": sum(dict(subgrafo.degree()).values()) / num_nodos,
                "articulos_muestra": muestra
            })

    return {
        "total_componentes": len(componentes),
        "mayor_componente": componentes[0] if componentes else {},
        "componentes": componentes[:10]  # solo las 10 mayores para resumen
    }
    
# ---------- Shortest path (Dijkstra using cost = 1 - weight) ----------
def caminos_minimos(G, origen, destino):
    if origen not in G or destino not in G:
        return {"error": "Origen o destino no existen en el grafo."}
    # create cost graph where cost = 1 - weight (larger weight -> small cost)
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        w = float(d.get('weight', 0.0))
        cost = max(1.0 - w, 1e-6)
        H.add_edge(u, v, cost=cost, weight=w)

    try:
        length, path = nx.single_source_dijkstra(H, source=origen, target=destino, weight='cost')
        # compute sum of weights on path
        sim_sum = 0.0
        edges_info = []
        for a, b in zip(path[:-1], path[1:]):
            w = G[a][b].get('weight', 0.0)
            edges_info.append({"from": a, "to": b, "weight": float(w)})
            sim_sum += float(w)
        return {"camino": path, "cost_total": float(length), "sim_total": float(sim_sum), "edges": edges_info}
    except nx.NetworkXNoPath:
        return {"error": "No hay camino entre origen y destino."}