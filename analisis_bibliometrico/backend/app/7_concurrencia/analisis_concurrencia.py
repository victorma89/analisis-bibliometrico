import os
import itertools
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64
import bibtexparser
import importlib
from collections import Counter
import re

# --- Importar Requerimientos 3 y 4 ---
analizador_frecuencias = importlib.import_module("app.3_frecuencia_palabras.analizador_frecuencias")
analizador_cluster = importlib.import_module("app.4_agrupamiento_jerarquico.analizador_cluster")

# === Ruta de datos ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BIB_PATH = os.path.join(ROOT_DIR, 'datos', 'procesados', 'articulos_unicos.bib')

# === Función auxiliar: cargar abstracts ===
def _cargar_abstracts():
    if not os.path.exists(BIB_PATH):
        raise FileNotFoundError(f"No se encontró el archivo: {BIB_PATH}")
    with open(BIB_PATH, 'r', encoding='utf-8') as f:
        parser = bibtexparser.bparser.BibTexParser(common_strings=False)
        parser.ignore_errors = True
        db = bibtexparser.load(f, parser=parser)
    return [entry['abstract'] for entry in db.entries if 'abstract' in entry]

# === Construcción del grafo de coocurrencia ===
def construir_grafo_coocurrencia(abstracts, terminos):
    """
    Construye un grafo de coocurrencia usando búsqueda de palabras completas.
    """
    G = nx.Graph()
    
    print(f"\n🔍 Construyendo grafo con {len(terminos)} términos...")
    print(f"   Términos: {terminos[:10]}...")  # Mostrar primeros 10
    
    # 📊 Contador de apariciones por término
    contador_terminos = Counter()
    
    # 📊 Contador de coocurrencias
    coocurrencias_debug = []
    
    for idx, abstract in enumerate(abstracts):
        if idx % 100 == 0:
            print(f"   Procesando abstract {idx}/{len(abstracts)}...")
        
        texto = abstract.lower()
        
        # ✅ Búsqueda de palabras completas usando regex
        presentes = []
        for termino in terminos:
            pattern = r'\b' + re.escape(termino.lower()) + r'\b'
            if re.search(pattern, texto):
                presentes.append(termino)
                contador_terminos[termino] += 1
        
        # 🔍 DEBUG: Mostrar cuántos términos aparecen juntos
        if idx < 5:  # Primeros 5 abstracts
            print(f"      Abstract {idx}: {len(presentes)} términos presentes: {presentes[:5]}...")
        
        # Crear aristas entre términos que coocurren
        combinaciones = list(itertools.combinations(set(presentes), 2))
        
        for w1, w2 in combinaciones:
            if G.has_edge(w1, w2):
                G[w1][w2]['weight'] += 1
            else:
                G.add_edge(w1, w2, weight=1)
                
            # DEBUG: Guardar las primeras coocurrencias
            if len(coocurrencias_debug) < 10:
                coocurrencias_debug.append((w1, w2))
    
    # 📊 MOSTRAR ESTADÍSTICAS DE DEBUG
    print(f"\n📊 ESTADÍSTICAS DE CONSTRUCCIÓN:")
    print(f"   Grafo: {len(G.nodes())} nodos, {len(G.edges())} aristas")
    print(f"\n   🔝 Top 10 términos más frecuentes:")
    for termino, freq in contador_terminos.most_common(10):
        print(f"      {termino}: {freq} apariciones")
    
    print(f"\n   🔗 Primeras 10 coocurrencias detectadas:")
    for w1, w2 in coocurrencias_debug[:10]:
        peso = G[w1][w2]['weight'] if G.has_edge(w1, w2) else 0
        print(f"      {w1} <-> {w2}: peso {peso}")
    
    print(f"\n   📈 Distribución de grados:")
    grados = dict(G.degree())
    if grados:
        print(f"      Mínimo: {min(grados.values())}")
        print(f"      Máximo: {max(grados.values())}")
        print(f"      Promedio: {sum(grados.values()) / len(grados):.2f}")
    
    return G

# === Graficar el grafo ===
def _graficar_grafo(G):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=800)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title("Grafo de Coocurrencia de Términos", fontsize=13)
    plt.axis('off')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

# === Función principal ===
def analizar_grafo_coocurrencia():
    """
    Análisis mejorado con filtrado de términos muy comunes.
    """
    try:
        print("\n" + "="*60)
        print("🔬 INICIANDO ANÁLISIS DE GRAFO DE COOCURRENCIA")
        print("="*60)
        
        # --- 1. Cargar abstracts ---
        abstracts = _cargar_abstracts()
        if not abstracts:
            return {
                "Construcción automática del grafo": "⚠️ No se encontraron abstracts.",
                "Cálculo del grado (términos más conectados)": [],
                "Detección de componentes conexas": {
                    "num_componentes": 0, 
                    "tamano_componentes": []
                },
                "grafico_base64": None
            }

        # --- 2. Definir términos MÁS ESPECÍFICOS ---
        palabras_clave_base = [
            # Modelos generativos
            "gpt", "bert", "transformer", "gan", "vae", "diffusion",
            
            # Técnicas específicas
            "fine-tuning", "prompt engineering", "few-shot", "zero-shot",
            "transfer learning", "attention mechanism",
            
            # Aplicaciones
            "text generation", "image generation", "chatbot",
            "natural language processing", "computer vision",
            
            # Métricas
            "accuracy", "precision", "recall", "f1-score",
            
            # Conceptos éticos
            "bias", "fairness", "explainability", "transparency",
            "privacy", "interpretability"
        ]
        
        terminos_finales = palabras_clave_base.copy()
        
        # --- 3. Obtener términos de otros requerimientos ---
        if analizador_frecuencias:
            try:
                print("📊 Obteniendo términos del Requerimiento 3...")
                resultado_req3 = analizador_frecuencias.analizar_frecuencias_completo(
                    abstracts, 
                    palabras_clave_base
                )
                # Solo tomar términos con frecuencia moderada (no muy comunes ni muy raros)
                terminos_req3 = list(resultado_req3.get("frecuencias_generadas", {}).keys())[:20]
                terminos_finales.extend(terminos_req3)
                print(f"   ✅ Agregados {len(terminos_req3)} términos del Req 3")
            except Exception as e:
                print(f"   ⚠️ Error obteniendo términos del Req 3: {e}")

        # --- 4. FILTRAR TÉRMINOS DEMASIADO COMUNES ---
        print("\n🔍 Filtrando términos muy comunes...")
        
        # Contar frecuencia de cada término
        frecuencias = Counter()
        for abstract in abstracts:
            texto = abstract.lower()
            for termino in terminos_finales:
                pattern = r'\b' + re.escape(termino.lower()) + r'\b'
                if re.search(pattern, texto):
                    frecuencias[termino] += 1
        
        total_abstracts = len(abstracts)
        
        # Filtrar términos que aparecen en más del 80% de los abstracts (muy comunes)
        # o en menos del 2% (muy raros)
        terminos_filtrados = []
        for termino, freq in frecuencias.items():
            porcentaje = (freq / total_abstracts) * 100
            if 2 <= porcentaje <= 80:  # Entre 2% y 80%
                terminos_filtrados.append(termino)
            else:
                print(f"   ❌ Descartado '{termino}': {porcentaje:.1f}% apariciones")
        
        print(f"\n   ✅ Términos filtrados: {len(terminos_filtrados)} de {len(terminos_finales)}")
        
        if len(terminos_filtrados) < 5:
            print("   ⚠️ Muy pocos términos después del filtrado, usando términos originales")
            terminos_filtrados = terminos_finales[:30]  # Tomar los primeros 30
        
        terminos_finales = list(set(terminos_filtrados))
        print(f"\n📋 Total de términos únicos después de filtrado: {len(terminos_finales)}")
        
        if not terminos_finales:
            return {
                "Construcción automática del grafo": "⚠️ No hay términos para construir el grafo.",
                "Cálculo del grado (términos más conectados)": [],
                "Detección de componentes conexas": {
                    "num_componentes": 0, 
                    "tamano_componentes": []
                },
                "grafico_base64": None
            }

        # --- 5. Construcción del grafo ---
        print("\n🔨 Construyendo grafo de coocurrencia...")
        G = construir_grafo_coocurrencia(abstracts, terminos_finales)
        
        if not G.nodes():
            return {
                "Construcción automática del grafo": "⚠️ Grafo vacío (sin coocurrencias detectadas).",
                "Cálculo del grado (términos más conectados)": [],
                "Detección de componentes conexas": {
                    "num_componentes": 0, 
                    "tamano_componentes": []
                },
                "grafico_base64": None
            }

        # --- 6. FILTRAR ARISTAS CON POCO PESO ---
        print("\n🔍 Filtrando aristas con poco peso...")
        aristas_originales = len(G.edges())
        
        # Calcular peso mínimo (por ejemplo, 5% del máximo)
        pesos = [data['weight'] for _, _, data in G.edges(data=True)]
        if pesos:
            peso_max = max(pesos)
            peso_min_threshold = max(2, peso_max * 0.05)  # Al menos 2 o 5% del máximo
            
            # Crear nuevo grafo solo con aristas significativas
            G_filtrado = nx.Graph()
            G_filtrado.add_nodes_from(G.nodes(data=True))
            
            for u, v, data in G.edges(data=True):
                if data['weight'] >= peso_min_threshold:
                    G_filtrado.add_edge(u, v, weight=data['weight'])
            
            # Eliminar nodos aislados
            nodos_aislados = list(nx.isolates(G_filtrado))
            G_filtrado.remove_nodes_from(nodos_aislados)
            
            print(f"   Aristas antes: {aristas_originales}")
            print(f"   Aristas después: {len(G_filtrado.edges())}")
            print(f"   Nodos eliminados (aislados): {len(nodos_aislados)}")
            
            G = G_filtrado

        # --- 7. Generar gráfico ---
        print("\n🎨 Generando visualización...")
        grafico_base64 = _graficar_grafo(G)

        # --- 8. Cálculo de grado ---
        print("\n📊 Calculando estadísticas del grafo...")
        grados = dict(G.degree())
        top_grado = sorted(grados.items(), key=lambda x: x[1], reverse=True)[:15]

        # --- 9. Componentes conexas ---
        componentes = list(nx.connected_components(G))
        tamanos_componentes = sorted([len(c) for c in componentes], reverse=True)

        # --- 10. Estadísticas adicionales ---
        densidad = nx.density(G)
        grado_medio = sum(dict(G.degree()).values()) / len(G.nodes()) if G.nodes() else 0

        print("\n" + "="*60)
        print("✅ ANÁLISIS COMPLETADO")
        print(f"   📍 Nodos: {len(G.nodes())}")
        print(f"   🔗 Aristas: {len(G.edges())}")
        print(f"   📊 Densidad: {densidad:.4f}")
        print(f"   📈 Grado medio: {grado_medio:.2f}")
        print(f"   🔴 Componentes: {len(componentes)}")
        print("="*60 + "\n")

        return {
            "Construcción automática del grafo": f"✅ Grafo creado con {len(G.nodes())} términos y {len(G.edges())} relaciones.",
            "Cálculo del grado (términos más conectados)": top_grado,
            "Detección de componentes conexas": {
                "num_componentes": len(componentes),
                "tamano_componentes": tamanos_componentes
            },
            "estadisticas_adicionales": {
                "densidad": round(densidad, 4),
                "grado_medio": round(grado_medio, 2),
                "nodos": len(G.nodes()),
                "aristas": len(G.edges())
            },
            "grafico_base64": grafico_base64
        }

    except FileNotFoundError as e:
        print(f"❌ Archivo no encontrado: {e}")
        return {"error": f"Archivo no encontrado: {str(e)}"}
    except Exception as e:
        print(f"❌ Error interno en analizar_grafo_coocurrencia: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error interno: {str(e)}"}