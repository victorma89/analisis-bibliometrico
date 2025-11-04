# analizador_citaciones.py - VERSIÓN MEJORADA CON VISUALIZACIÓN PROFESIONAL
import io
import base64
from pathlib import Path
import bibtexparser
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================== UTILIDADES ====================
def normalize(s):
    """Normaliza strings eliminando espacios extras"""
    return (s or "").strip()

def parse_authors(entry):
    """Extrae lista de autores de una entrada BibTeX"""
    auth = entry.get('author', '') or ''
    if not auth:
        return []
    return [a.strip() for a in auth.replace('\n', ' ').split(' and ') if a.strip()]

def parse_keywords(entry):
    """Extrae lista de palabras clave de una entrada BibTeX"""
    keys = entry.get('keywords', '') or entry.get('keyword', '') or ''
    if not keys:
        return []
    return [k.strip() for k in keys.replace(';', ',').split(',') if k.strip()]

def entry_to_dict(entry):
    """Convierte una entrada BibTeX a diccionario estructurado"""
    return {
        'id': entry.get('ID'),
        'title': normalize(entry.get('title', '')),
        'authors': parse_authors(entry),
        'keywords': parse_keywords(entry),
        'year': entry.get('year', None),
        'abstract': normalize(entry.get('abstract', '')),
        'raw': entry
    }

# ==================== CARGA DE DATOS ====================
def load_bib_file(path):
    """Carga archivo .bib y retorna lista de artículos"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo: {p}")
    
    with open(p, encoding='utf-8') as f:
        bib_database = bibtexparser.load(f)
    
    articles = [entry_to_dict(e) for e in bib_database.entries]
    print(f"✅ Cargados {len(articles)} artículos desde {p.name}")
    return articles

# ==================== CONSTRUCCIÓN DEL GRAFO (PUNTO 1) ====================
def infer_graph(articles, thr=0.25, verbose=True):
    """
    PUNTO 1: Construcción automática del grafo de citaciones
    
    Construye un grafo DIRIGIDO donde:
    - Cada nodo = artículo científico
    - Arista u→v = u "cita" a v (inferido por similitud)
    - Peso = similitud entre artículos (0 a 1)
    
    Args:
        articles: Lista de diccionarios con datos de artículos
        thr: Umbral de similitud mínimo para crear arista (default: 0.25)
        verbose: Si True, imprime estadísticas
    
    Returns:
        nx.DiGraph con nodos y aristas ponderadas
    """
    G = nx.DiGraph()
    
    if not articles:
        print("⚠️ No hay artículos para construir el grafo")
        return G

    # ===== Agregar nodos con atributos =====
    print(f"📊 Agregando {len(articles)} nodos al grafo...")
    for a in articles:
        year_val = None
        if a.get('year'):
            year_str = str(a['year']).strip()
            if year_str.isdigit():
                year_val = int(year_str)
        
        G.add_node(
            a['id'],
            title=a.get('title', ''),
            authors=a.get('authors', []),
            keywords=a.get('keywords', []),
            year=year_val,
            abstract=a.get('abstract', '')
        )

    # ===== Preparar corpus para TF-IDF =====
    print("🔍 Calculando similitud entre artículos...")
    docs = []
    for a in articles:
        parts = []
        if a.get('title'):
            parts.append(a['title'])
        if a.get('abstract'):
            parts.append(a['abstract'])
        if a.get('keywords'):
            parts.append(" ".join(a['keywords']))
        if a.get('authors'):
            parts.append(" ".join(a['authors']))
        
        docs.append(" ||| ".join(parts) if parts else "sin_contenido")

    # ===== Calcular matriz de similitud =====
    try:
        vec = TfidfVectorizer(max_features=5000, stop_words='english', min_df=1, max_df=0.95)
        X = vec.fit_transform(docs)
        sim_matrix = cosine_similarity(X)
        print("   ✅ Similitud calculada con TF-IDF + Coseno")
    except Exception as e:
        print(f"   ⚠️ Error en TF-IDF: {e}")
        print("   🔄 Usando fuzzy matching como alternativa...")
        n = len(articles)
        sim_matrix = np.zeros((n, n), dtype=float)
        titles = [a.get('title', '') for a in articles]
        for i in range(n):
            for j in range(n):
                if i != j:
                    sim_matrix[i, j] = fuzz.token_sort_ratio(titles[i], titles[j]) / 100.0

    # ===== Añadir aristas según umbral =====
    n = len(articles)
    ids = [a['id'] for a in articles]
    edges_added = 0
    
    print(f"🔗 Creando aristas con umbral de similitud ≥ {thr}...")
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            sim = float(sim_matrix[i, j])
            if sim >= thr:
                u, v = ids[i], ids[j]
                G.add_edge(u, v, weight=sim, similarity=sim)
                edges_added += 1

    # ===== Calcular "citaciones" (in-degree) =====
    in_degrees = dict(G.in_degree())
    for node in G.nodes():
        G.nodes[node]['citaciones'] = int(in_degrees.get(node, 0))

    # ===== Estadísticas finales =====
    if verbose:
        print(f"\n{'='*70}")
        print("✅ PUNTO 1: GRAFO DE CITACIONES CONSTRUIDO")
        print(f"{'='*70}")
        print(f"📌 Nodos (artículos):        {G.number_of_nodes()}")
        print(f"🔗 Aristas (citaciones):     {G.number_of_edges()}")
        print(f"📊 Umbral de similitud:      {thr}")
        print(f"📈 Densidad del grafo:       {nx.density(G):.6f}")
        
        cit_vals = [G.nodes[n]['citaciones'] for n in G.nodes()] or [0]
        print(f"📚 Citaciones máximas:       {max(cit_vals)}")
        print(f"📊 Citaciones promedio:      {np.mean(cit_vals):.2f}")
        
        # Top 5 más citados
        top = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        if top and top[0][1] > 0:
            print(f"\n🏆 Top 5 artículos más citados:")
            for idx, (nid, c) in enumerate(top, 1):
                title = G.nodes[nid].get('title', 'Sin título')[:50]
                print(f"   {idx}. [{nid}] {c} citaciones - {title}...")
        print(f"{'='*70}\n")

    return G

# ==================== PUNTO 2: CAMINOS MÍNIMOS ====================
def caminos_minimos(G, origen, destino, algorithm='dijkstra'):
    """
    PUNTO 2: Cálculo de caminos mínimos entre artículos
    
    Calcula el camino más corto entre dos artículos usando:
    - Dijkstra (recomendado para grafos dispersos)
    - Floyd-Warshall (para grafos densos o múltiples consultas)
    
    El "costo" se define como: cost = 1 - similitud
    (Mayor similitud = menor costo = camino preferido)
    
    Args:
        G: Grafo dirigido de citaciones
        origen: ID del artículo origen
        destino: ID del artículo destino
        algorithm: 'dijkstra' o 'floyd-warshall'
    
    Returns:
        Dict con camino, longitud, costos y similitudes
    """
    if origen not in G:
        return {"error": f"El artículo origen '{origen}' no existe en el grafo"}
    if destino not in G:
        return {"error": f"El artículo destino '{destino}' no existe en el grafo"}

    # Crear grafo con atributo 'cost' para algoritmos de camino mínimo
    H = nx.DiGraph()
    for n, d in G.nodes(data=True):
        H.add_node(n, **d)
    
    for u, v, data in G.edges(data=True):
        w = float(data.get('weight', 0.0))
        cost = max(1.0 - w, 1e-6)  # Evitar costos negativos o cero
        H.add_edge(u, v, cost=cost, weight=w)

    # ===== DIJKSTRA =====
    if algorithm.lower() == 'dijkstra':
        try:
            length, path = nx.single_source_dijkstra(H, source=origen, target=destino, weight='cost')
            
            # Reconstruir información de aristas
            edges_info = []
            sim_sum = 0.0
            for a, b in zip(path[:-1], path[1:]):
                w = float(G[a][b].get('weight', 0.0))
                edges_info.append({
                    "from": a,
                    "to": b,
                    "weight": w,
                    "from_title": G.nodes[a].get("title", "")[:60],
                    "to_title": G.nodes[b].get("title", "")[:60]
                })
                sim_sum += w
            
            return {
                "algoritmo": "Dijkstra",
                "camino": path,
                "longitud": len(path),
                "costo_total": float(length),
                "similitud_total": float(sim_sum),
                "similitud_promedio": float(sim_sum / (len(path)-1)) if len(path) > 1 else 0.0,
                "edges": edges_info
            }
        except nx.NetworkXNoPath:
            return {"error": f"No existe camino entre '{origen}' y '{destino}' (Dijkstra)"}

    # ===== FLOYD-WARSHALL =====
    elif algorithm.lower() in ('floyd', 'floyd-warshall', 'floyd_warshall'):
        # --- INICIO DE LA MODIFICACIÓN ---
        # Añadir un chequeo para el tamaño del grafo
        num_nodos = G.number_of_nodes()
        if num_nodos > 200:
            return {
                "error": f"El algoritmo Floyd-Warshall no es recomendado para grafos con más de 200 nodos (actual: {num_nodos}). Su complejidad O(V^3) puede causar tiempos de espera muy largos. Por favor, use Dijkstra para grafos grandes."
            }
        # --- FIN DE LA MODIFICACIÓN ---
        try:
            pred, dist = nx.floyd_warshall_predecessor_and_distance(H, weight='cost')
            
            # Verificar si existe camino
            if origen not in dist or destino not in dist[origen]:
                return {"error": f"No existe camino entre '{origen}' y '{destino}' (Floyd-Warshall)"}
            
            # Reconstruir camino desde pred
            camino = [destino]
            cur = destino
            while cur != origen:
                if origen not in pred or cur not in pred[origen]:
                    return {"error": f"No se pudo reconstruir el camino (Floyd-Warshall)"}
                cur = pred[origen][cur]
                camino.append(cur)
            camino.reverse()
            
            # Calcular métricas
            costo_total = float(dist[origen][destino])
            sim_sum = 0.0
            edges_info = []
            for a, b in zip(camino[:-1], camino[1:]):
                w = float(G[a][b].get('weight', 0.0))
                sim_sum += w
                edges_info.append({"from": a, "to": b, "weight": w})
            
            return {
                "algoritmo": "Floyd-Warshall",
                "camino": camino,
                "longitud": len(camino),
                "costo_total": costo_total,
                "similitud_total": float(sim_sum),
                "similitud_promedio": float(sim_sum / (len(camino)-1)) if len(camino) > 1 else 0.0,
                "edges": edges_info
            }
        except Exception as e:
            return {"error": f"Error en Floyd-Warshall: {str(e)}"}
    
    else:
        return {"error": f"Algoritmo desconocido: '{algorithm}'. Use 'dijkstra' o 'floyd-warshall'"}

# ==================== PUNTO 3: COMPONENTES FUERTEMENTE CONEXAS ====================
def identificar_componentes_fuertemente_conexas(G):
    """
    PUNTO 3: Identificación de componentes fuertemente conexas
    
    Detecta grupos de artículos donde existe un camino dirigido
    entre cualquier par de artículos dentro del grupo.
    
    Usa el algoritmo de Tarjan (implementado en NetworkX).
    
    Args:
        G: Grafo dirigido de citaciones
    
    Returns:
        Dict con estadísticas y lista de componentes
    """
    sccs = list(nx.strongly_connected_components(G))
    
    componentes = []
    for i, comp in enumerate(sorted(sccs, key=len, reverse=True), 1):
        sub = G.subgraph(comp)
        nodos = list(comp)
        
        # Estadísticas del componente
        grados = dict(sub.degree())
        grado_prom = sum(grados.values()) / len(nodos) if nodos else 0
        
        # Muestra de artículos (máximo 5)
        muestra = []
        for n in nodos[:5]:
            muestra.append({
                "id": n,
                "title": G.nodes[n].get("title", "")[:60],
                "citaciones": int(G.nodes[n].get("citaciones", 0)),
                "year": G.nodes[n].get("year")
            })
        
        componentes.append({
            "id": i,
            "num_nodos": len(nodos),
            "nodos": nodos,
            "grado_promedio": float(grado_prom),
            "articulos_muestra": muestra,
            "num_aristas": sub.number_of_edges(),
            "densidad": float(nx.density(sub))
        })

    # Resumen general
    resultado = {
        "total_componentes": len(componentes),
        "mayor_componente": componentes[0] if componentes else {},
        "componentes": componentes[:10],  # Primeros 10 componentes
        "algoritmo": "Tarjan (NetworkX - strongly_connected_components)"
    }
    
    print(f"\n{'='*70}")
    print("✅ PUNTO 3: COMPONENTES FUERTEMENTE CONEXAS IDENTIFICADAS")
    print(f"{'='*70}")
    print(f"🔴 Total de componentes:     {len(componentes)}")
    if componentes:
        print(f"🏆 Componente mayor:         {componentes[0]['num_nodos']} nodos")
        print(f"📊 Distribución de tamaños:  {[c['num_nodos'] for c in componentes[:5]]}")
    print(f"{'='*70}\n")
    
    return resultado

# ==================== VISUALIZACIÓN PROFESIONAL CLARA Y EXPLICATIVA ====================
def graficar_grafo_citaciones(G, titulo="Red de Citaciones Inferida por Similitud"):
    """
    Visualización clara y profesional del grafo de citaciones.
    - Flechas visibles y proporcionales (dirección y peso)
    - Paleta interpretativa: azul → baja citación, naranja/rojo → alta citación
    - Leyenda visual de colores
    - Fondo limpio con contraste
    """
    import matplotlib.patches as mpatches

    if not G or G.number_of_nodes() == 0:
        print("⚠️ Grafo vacío, no se puede generar visualización")
        return None

    print("🎨 Generando visualización clara y profesional del grafo...")

    # --- Citaciones y pesos ---
    citaciones = [G.nodes[n].get('citaciones', 0) for n in G.nodes()]
    max_citas = max(citaciones) if citaciones else 1

    # --- Tamaños proporcionales ---
    sizes = [80 + (c / max_citas) * 300 for c in citaciones]

    # --- Colores informativos (azul → bajo, naranja → alto) ---
    def color_map(c):
        if c < max_citas * 0.2:
            return "#A9CCE3"  # azul claro
        elif c < max_citas * 0.4:
            return "#5DADE2"  # azul medio
        elif c < max_citas * 0.6:
            return "#3498DB"  # azul fuerte
        elif c < max_citas * 0.8:
            return "#F4A261"  # naranja
        else:
            return "#E74C3C"  # rojo suave (altamente citado)
    colors = [color_map(c) for c in citaciones]

    # --- Layout más amplio ---
    try:
        pos = nx.kamada_kawai_layout(G, scale=3.5)
    except:
        pos = nx.spring_layout(G, k=3.0 / np.sqrt(G.number_of_nodes()), iterations=150, seed=42)

    # --- Crear figura ---
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor("#F9FAFB")

    # --- Dibujar aristas dirigidas ---
    print("   Dibujando aristas...")
    edge_weights = [d.get("weight", 0.2) for _, _, d in G.edges(data=True)]
    nx.draw_networkx_edges(
        G, pos,
        edge_color="#95A5A6",
        alpha=0.25,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=8,
        width=[0.5 + 1.5 * w for w in edge_weights],
        connectionstyle='arc3,rad=0.05',
        ax=ax
    )

    # --- Dibujar nodos ---
    print("   Dibujando nodos...")
    nx.draw_networkx_nodes(
        G, pos,
        node_size=sizes,
        node_color=colors,
        edgecolors='#2C3E50',
        linewidths=0.8,
        alpha=0.95,
        ax=ax
    )

    # --- Etiquetas solo si el grafo es pequeño ---
    if G.number_of_nodes() <= 250:
        labels = {}
        top_citas = sorted(citaciones, reverse=True)[:15]
        threshold = min(top_citas) if top_citas else 0
        for n in G.nodes():
            cit = G.nodes[n].get('citaciones', 0)
            if cit >= threshold and cit > 0:
                titulo = G.nodes[n].get('title', '')
                palabras = titulo.split()[:4]
                labels[n] = " ".join(palabras) + ("..." if len(titulo.split()) > 4 else "")
        nx.draw_networkx_labels(
            G, pos, labels=labels,
            font_size=6.5, font_color="#1B2631", font_family="sans-serif", ax=ax
        )

    # --- Título profesional ---
    plt.title(
        titulo,
        fontsize=15,
        fontweight='bold',
        color='#2C3E50',
        pad=25
    )

    # --- Leyenda interpretativa ---
    legend_elements = [
        mpatches.Patch(color="#A9CCE3", label="Bajas citaciones"),
        mpatches.Patch(color="#3498DB", label="Citaciones medias"),
        mpatches.Patch(color="#F4A261", label="Altas citaciones"),
        mpatches.Patch(color="#E74C3C", label="Muy altas citaciones (artículos influyentes)")
    ]
    plt.legend(
        handles=legend_elements,
        loc='lower right',
        fontsize=9,
        frameon=True,
        edgecolor='gray'
    )

    plt.axis('off')
    plt.tight_layout()

    # --- Guardar en buffer (base64) ---
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    print("✅ Visualización clara generada correctamente.")
    return img_b64


# ==================== EXPORTACIÓN JSON ====================
def export_graph_to_json(G):
    """
    Exporta el grafo a formato JSON compatible con vis-network
    
    Returns:
        Dict con 'nodes' y 'edges' en formato vis.js
    """
    nodes = []
    for n in G.nodes():
        node_data = G.nodes[n]
        nodes.append({
            "id": n,
            "label": (node_data.get("title", "") or str(n))[:50],
            "title": node_data.get("title", ""),
            "authors": node_data.get("authors", []),
            "year": node_data.get("year"),
            "citaciones": int(node_data.get("citaciones", 0))
        })
    
    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({
            "from": u,
            "to": v,
            "weight": float(d.get("weight", 0.0))
        })
    
    return {"nodes": nodes, "edges": edges}

# ==================== ANÁLISIS COMPLETO ====================
def analizar_grafo(G):
    """
    Genera estadísticas completas del grafo
    
    Returns:
        Dict con métricas y análisis
    """
    if G.number_of_nodes() == 0:
        return {
            "num_nodos": 0,
            "num_aristas": 0,
            "densidad": 0.0,
            "grado_medio": 0.0
        }
    
    info = {
        "num_nodos": int(G.number_of_nodes()),
        "num_aristas": int(G.number_of_edges()),
        "densidad": float(nx.density(G)),
        "grado_medio": float(sum(dict(G.degree()).values()) / G.number_of_nodes())
    }
    
    # Centralidad de grado
    try:
        deg_cent = nx.degree_centrality(G)
        top_deg = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:10]
        info["top_degree_centrality"] = [
            {"node": n, "centrality": float(v), "title": G.nodes[n].get('title', '')[:60]}
            for n, v in top_deg
        ]
    except:
        info["top_degree_centrality"] = []
    
    # PageRank
    try:
        pr = nx.pagerank(G, max_iter=200)
        top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]
        info["top_pagerank"] = [
            {"node": n, "pagerank": float(v), "title": G.nodes[n].get('title', '')[:60]}
            for n, v in top_pr
        ]
    except:
        info["top_pagerank"] = []
    
    # Estadísticas de citaciones
    citaciones = [G.nodes[n].get('citaciones', 0) for n in G.nodes()]
    if citaciones:
        info["estadisticas_citaciones"] = {
            "max": int(max(citaciones)),
            "min": int(min(citaciones)),
            "promedio": float(np.mean(citaciones)),
            "mediana": float(np.median(citaciones)),
            "total": int(sum(citaciones))
        }
    
    return info

# ==================== FUNCIÓN PRINCIPAL ====================
def analizar_citaciones_completo(bib_path, sim_threshold=0.25, verbose=True):
    """
    Ejecuta el análisis completo del Requerimiento 1 - Seguimiento 2
    
    Incluye:
    1. Construcción del grafo de citaciones
    2. Cálculo de métricas y estadísticas
    3. Identificación de componentes fuertemente conexas
    4. Generación de visualización base64 PROFESIONAL
    5. Exportación a JSON
    
    Args:
        bib_path: Ruta al archivo .bib
        sim_threshold: Umbral de similitud (default: 0.25)
        verbose: Imprimir información detallada
    
    Returns:
        Dict con todos los resultados del análisis
    """
    try:
        # 1. Cargar artículos
        articles = load_bib_file(bib_path)
        if not articles:
            return {"error": "No hay artículos en el archivo .bib"}

        # 2. Construir grafo (PUNTO 1)
        G = infer_graph(articles, thr=sim_threshold, verbose=verbose)

        # 3. Análisis general
        info = analizar_grafo(G)

        # 4. Componentes fuertemente conexas (PUNTO 3)
        componentes_info = identificar_componentes_fuertemente_conexas(G)

        # 5. Visualización base64 PROFESIONAL
        imagen_base64 = graficar_grafo_citaciones(G)

        # 6. Export JSON
        grafo_json = export_graph_to_json(G)

        resultado = {
            "analisis": info,
            "componentes": componentes_info,
            "grafico_base64": imagen_base64,
            "grafo_json": grafo_json,
            "params": {
                "sim_threshold": sim_threshold
            }
        }

        return resultado

    except FileNotFoundError as e:
        return {"error": f"Archivo no encontrado: {str(e)}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Error interno: {str(e)}"}