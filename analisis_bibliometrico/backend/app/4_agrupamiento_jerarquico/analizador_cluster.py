import os
import bibtexparser
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# --- Constantes y Rutas ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BIB_FILE_PATH = os.path.join(ROOT_DIR, 'datos', 'procesados', 'articulos_unicos.bib')

EXPLICACIONES = {
    "ward": """
**Método de Agrupamiento: Ward**

1.  **¿Qué es?**: Es un método de agrupamiento jerárquico que busca minimizar la varianza total dentro de cada clúster. En cada paso, fusiona los dos clústeres cuya unión resulta en el menor aumento de la \"suma de cuadrados de las diferencias\" total.

2.  **Características**:
    *   Tiende a producir clústeres de tamaño similar y forma esférica (compactos).
    *   Es sensible a los valores atípicos (outliers).
    *   Generalmente se considera muy eficiente y produce agrupamientos fáciles de interpretar.

3.  **Ideal para**: Descubrir clústeres bien definidos y separados en los datos.
""",
    "complete": """
**Método de Agrupamiento: Complete Linkage (Enlace Completo)**

1.  **¿Qué es?**: En este método, la distancia entre dos clústeres se define como la distancia **máxima** entre un punto en el primer clúster y un punto en el segundo clúster.

2.  **Características**:
    *   Tiende a producir clústeres más compactos y apretados.
    *   Es menos sensible a los valores atípicos que el método de Ward.
    *   Puede tener dificultades con clústeres de formas no esféricas.

3.  **Ideal para**: Encontrar los grupos más compactos posibles, donde todos los miembros de un clúster están relativamente cerca entre sí.
""",
    "average": """
**Método de Agrupamiento: Average Linkage (Enlace Promedio - UPGMA)**

1.  **¿Qué es?**: La distancia entre dos clústeres se calcula como la distancia **promedio** entre cada punto de un clúster y cada punto del otro clúster.

2.  **Características**:
    *   Es un punto intermedio entre \"Single Linkage\" (que es muy sensible a outliers) y \"Complete Linkage\".
    *   Es menos propenso a ser afectado por valores atípicos en comparación con Ward y Complete Linkage.
    *   Tiende a fusionar clústeres con baja varianza y a producir clústeres con varianzas similares.

3.  **Ideal para**: Cuando se espera que los clústeres no sean necesariamente compactos o esféricos, y se busca un balance general.
"""
}

def _cargar_y_muestrear_articulos(num_articulos: int):
    """Carga artículos y toma una muestra aleatoria si es necesario."""
    if not os.path.exists(BIB_FILE_PATH):
        raise FileNotFoundError(f"El archivo .bib no se encontró en {BIB_FILE_PATH}")

    with open(BIB_FILE_PATH, 'r', encoding='utf-8') as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)
    
    entries = [e for e in bib_database.entries if 'abstract' in e and e['abstract'].strip()]
    
    if len(entries) < 2:
        return None, "No hay suficientes artículos con abstract para analizar."

    if len(entries) > num_articulos:
        sample_entries = random.sample(entries, num_articulos)
    else:
        sample_entries = entries
        
    return sample_entries, None

def _generar_dendrograma_base64(linkage_matrix, labels):
    """Genera un dendrograma y lo devuelve como una imagen en base64."""
    plt.figure(figsize=(15, max(10, len(labels) * 0.5)))
    plt.title("Dendrograma de Similitud de Artículos")
    plt.xlabel("Distancia")
    plt.ylabel("Artículos")
    
    dendrogram(
        linkage_matrix,
        labels=labels,
        orientation='right',
        leaf_font_size=10
    )
    
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return img_base64

# --- Funciones de Lógica Principal ---

def analizar_agrupamiento_completo(metodo: str, num_articulos: int = 30):
    """
    Orquesta el análisis de agrupamiento jerárquico completo.
    """
    try:
        # 1. Cargar y muestrear artículos
        articulos, error = _cargar_y_muestrear_articulos(num_articulos)
        if error:
            return {"error": error}

        # Extraer abstracts y etiquetas (títulos)
        abstracts = [art['abstract'] for art in articulos]
        labels = [art.get('title', art.get('ID', 'ID Desconocido')) for art in articulos]
        # Truncar etiquetas largas para mejor visualización
        labels = [label[:75] + '...' if len(label) > 75 else label for label in labels]

        # 2. Vectorizar los abstracts con TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(abstracts)

        # 3. Calcular la matriz de distancia coseno
        # La distancia es 1 - similitud. Es lo que necesita el linkage.
        dist_matrix = cosine_distances(tfidf_matrix)

        # 4. Aplicar el algoritmo de `linkage`
        # linkage_matrix es la estructura jerárquica de los clústeres
        linkage_matrix = linkage(dist_matrix, method=metodo)

        # 5. Generar el dendrograma
        grafico_base64 = _generar_dendrograma_base64(linkage_matrix, labels)

        # 6. Generar explicaciones detalladas
        dendrograma_explicacion = f"""
**Cómo Interpretar el Dendrograma:**

1.  **Eje Y (Artículos):** Cada etiqueta en el eje vertical representa un único artículo científico de la muestra, extraído de `articulos_unicos.bib`. Está identificado por su título.

2.  **Eje X (Escala de Distancia):** El eje horizontal mide la **distancia** o **disimilitud** entre los abstracts de los artículos. La distancia se calculó usando 'Distancia Coseno' sobre la representación vectorial (TF-IDF) de los abstracts.
    *   Una **distancia de 0.0** significaría que los abstracts son idénticos en su contenido de palabras clave.
    *   Una **distancia mayor** indica que son más diferentes.

3.  **Líneas y Uniones (Clusters):** El dendrograma muestra cómo se unen los artículos para formar clústeres.
    *   Las líneas verticales conectan artículos (hojas) o grupos de artículos (ramas) que son considerados un clúster.
    *   La posición de la línea horizontal que une dos ramas indica la distancia a la que se realizó esa fusión. Uniones que ocurren más a la derecha (menor distancia) representan artículos o grupos que son **muy similares**. Uniones más a la izquierda (mayor distancia) agrupan clústeres que son **menos parecidos** entre sí.
"""

        # 7. Devolver el gráfico y la explicación
        return {
            "grafico_dendrograma_base64": grafico_base64,
            "explicacion_metodo": EXPLICACIONES.get(metodo, "No hay explicación disponible para este método."),
            "dendrograma_explicacion": dendrograma_explicacion,
            "metodo_usado": metodo,
            "num_articulos_analizados": len(articulos)
        }

    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": f"Ocurrió un error inesperado: {e}"}


if __name__ == '__main__':
    # Pequeña prueba para ejecutar el módulo de forma independiente
    print("--- Probando Módulo de Análisis de Agrupamiento Jerárquico ---")
    
    # Prueba con el método 'ward'
    print("\nEjecutando con método: 'ward'")
    resultado_ward = analizar_agrupamiento_completo(metodo='ward', num_articulos=25)
    if "error" in resultado_ward:
        print(f"  Error: {resultado_ward['error']}")
    else:
        print(f"  Análisis completado para {resultado_ward['num_articulos_analizados']} artículos.")
        print(f"  Gráfico generado: {resultado_ward['grafico_dendrograma_base64' is not None]}")

    # Prueba con el método 'average'
    print("\nEjecutando con método: 'average'")
    resultado_avg = analizar_agrupamiento_completo(metodo='average', num_articulos=25)
    if "error" in resultado_avg:
        print(f"  Error: {resultado_avg['error']}")
    else:
        print(f"  Análisis completado para {resultado_avg['num_articulos_analizados']} artículos.")
        print(f"  Gráfico generado: {resultado_avg['grafico_dendrograma_base64' is not None]}")

    print("\n--- Pruebas finalizadas ---")