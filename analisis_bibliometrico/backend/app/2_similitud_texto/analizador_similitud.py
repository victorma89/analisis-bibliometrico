import bibtexparser
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# --- Constantes y Modelos de IA cacheados ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BIB_FILE_PATH = os.path.join(ROOT_DIR, 'datos', 'procesados', 'articulos_unicos.bib')

# Caché para los modelos de IA. Se cargarán una sola vez.
model_cache = {}

EXPLICACIONES = {
    "levenshtein": """
**Algoritmo: Distancia de Levenshtein**

1.  **¿Qué es?**: Mide el número mínimo de ediciones (inserciones, eliminaciones o sustituciones) de un solo carácter necesarias para cambiar una cadena de texto en otra.

2.  **Funcionamiento Matemático**:
    *   Se construye una matriz donde la fila `i` y la columna `j` representan el costo de convertir los primeros `i` caracteres del texto 1 en los primeros `j` caracteres del texto 2.
    *   El costo de cada celda `(i, j)` se calcula como el mínimo de:
        a) Costo de inserción: `matriz[i-1][j] + 1`
        b) Costo de eliminación: `matriz[i][j-1] + 1`
        c) Costo de sustitución: `matriz[i-1][j-1] + (1 si char1 != char2, 0 si son iguales)`
    *   El valor en la esquina inferior derecha de la matriz es la distancia total.

3.  **Cálculo de Similitud**:
    *   `Similitud = 1 - (Distancia / Longitud del texto más largo)`
    *   Un resultado de 1.0 significa que los textos son idénticos.
""",
    "coseno": """
**Algoritmo: Similitud de Coseno con TF-IDF**

1.  **¿Qué es?**: Mide el coseno del ángulo entre dos vectores en un espacio multidimensional. En este caso, los vectores representan los abstracts de los artículos.

2.  **Funcionamiento (Paso a Paso)**:
    a) **TF-IDF (Term Frequency-Inverse Document Frequency)**: Primero, cada abstract se convierte en un vector numérico.
        *   **TF (Frecuencia del Término)**: ¿Qué tan a menudo aparece una palabra en el abstract? `(Nº de veces que aparece la palabra / Total de palabras en el abstract)`
        *   **IDF (Frecuencia Inversa del Documento)**: Mide la importancia de una palabra. Palabras comunes como "el" o "y" tienen un IDF bajo. `log(Nº total de abstracts / Nº de abstracts que contienen la palabra)`
        *   El valor final para cada palabra en el vector es `TF * IDF`.

    b) **Similitud de Coseno**: Una vez que tenemos los dos vectores (V1, V2) que representan los abstracts, se aplica la fórmula:
        *   `Similitud = (V1 · V2) / (||V1|| * ||V2||)`
        *   Donde `·` es el producto punto de los vectores y `||V||` es su magnitud (longitud).

3.  **Interpretación**:
    *   Un resultado de 1.0 significa que los abstracts son idénticos en términos de su contenido y frecuencia de palabras.
    *   Un resultado de 0.0 significa que no comparten ninguna palabra (después de aplicar TF-IDF).
""",
    "jaccard": """
**Algoritmo: Índice de Jaccard**

1.  **¿Qué es?**: Mide la similitud entre dos conjuntos finitos. Se define como el tamaño de la intersección dividido por el tamaño de la unión de los conjuntos.

2.  **Funcionamiento (Paso a Paso)**:
    a) **Creación de Conjuntos**: Cada abstract se convierte en un conjunto de palabras únicas (tokens).
    b) **Cálculo**: Se aplica la fórmula:
        *   `J(A, B) = |A ∩ B| / |A ∪ B|`
        *   `|A ∩ B|`: Número de palabras que aparecen en AMBOS abstracts.
        *   `|A ∪ B|`: Número total de palabras únicas en los dos abstracts combinados.

3.  **Interpretación**:
    *   El resultado es un valor entre 0 y 1.
    *   1.0 significa que los abstracts tienen exactamente el mismo conjunto de palabras.
    *   0.0 significa que no comparten ninguna palabra.
""",
    "dice": """
**Algoritmo: Coeficiente de Sørensen-Dice**

1.  **¿Qué es?**: Similar a Jaccard, mide la similitud entre dos conjuntos. Es más sensible a la intersección que Jaccard.

2.  **Funcionamiento (Paso a Paso)**:
    a) **Creación de Conjuntos**: Cada abstract se convierte en un conjunto de palabras únicas.
    b) **Cálculo**: Se aplica la fórmula:
        *   `DSC(A, B) = 2 * |A ∩ B| / (|A| + |B|)`
        *   `|A ∩ B|`: Número de palabras que aparecen en AMBOS abstracts.
        *   `|A| + |B|`: Suma del número de palabras en cada conjunto.

3.  **Interpretación**:
    *   El resultado está entre 0 y 1. Un valor de 1.0 indica que los conjuntos son idénticos.
""",
    "ia_mini_lm": """
**Algoritmo: IA - Sentence Transformer (all-MiniLM-L6-v2)**

1.  **¿Qué es?**: Utiliza un modelo de IA (una red neuronal profunda, tipo BERT) para convertir cada abstract en un vector numérico de alta calidad llamado "embedding". Este embedding captura el significado semántico del texto.

2.  **Funcionamiento (Paso a Paso)**:
    a) **Carga del Modelo**: Se carga el modelo pre-entado 'all-MiniLM-L6-v2', especializado en generar embeddings para frases y párrafos.
    b) **Codificación (Encoding)**: El modelo procesa el texto de cada abstract y lo convierte en un vector de 384 dimensiones. A diferencia de TF-IDF, este vector representa el "significado" del texto, no solo las palabras que contiene.
    c) **Cálculo de Similitud**: Se calcula la Similitud de Coseno entre los dos vectores generados. `Similitud = (V1 · V2) / (||V1|| * ||V2||)`

3.  **Interpretación**:
    *   El resultado está entre -1 y 1. Un valor cercano a 1.0 indica una alta similitud semántica (los textos significan lo mismo), incluso si usan palabras diferentes.
""",
    "ia_paraphrase": """
**Algoritmo: IA - Sentence Transformer (paraphrase-mpnet-base-v2)**

1.  **¿Qué es?**: Similar al otro modelo de IA, pero utiliza una arquitectura diferente ('paraphrase-mpnet-base-v2') que es especialmente buena para detectar si dos textos son paráfrasis uno del otro.

2.  **Funcionamiento**: El proceso es idéntico al del otro modelo de IA:
    a) Carga del modelo pre-entado.
    b) Codificación de cada abstract en un vector de alta dimensión (embedding).
    c) Cálculo de la Similitud de Coseno entre los dos vectores.

3.  **Interpretación**:
    *   Este modelo es generalmente más robusto y preciso que MiniLM para tareas de similitud semántica, aunque puede ser un poco más lento.
    *   Un valor cercano a 1.0 indica que los textos son muy probablemente una paráfrasis el uno del otro.
"""
}

# --- Funciones de Lógica Principal ---

def cargar_articulos():
    if not os.path.exists(BIB_FILE_PATH):
        return []
    with open(BIB_FILE_PATH, 'r', encoding='utf-8') as bibtex_file:
        parser = bibtexparser.bparser.BibTexParser(common_strings=True)
        bib_database = bibtexparser.load(bibtex_file, parser=parser)
    return bib_database.entries

def _get_abstracts(articulos, id1, id2):
    articulo1 = next((a for a in articulos if a.get('ID') == id1), None)
    articulo2 = next((a for a in articulos if a.get('ID') == id2), None)
    if not articulo1 or not articulo2:
        return None, None, {"error": "No se encontró uno o ambos artículos."}
    abstract1 = articulo1.get('abstract', '')
    abstract2 = articulo2.get('abstract', '')
    if not abstract1 or not abstract2:
        return None, None, {"error": "Uno o ambos artículos no tienen abstract."}
    return abstract1, abstract2, None

def calcular_distancia_levenshtein(s1, s2):
    if len(s1) < len(s2):
        return calcular_distancia_levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def analizar_similitud_levenshtein(articulos, id1, id2):
    abstract1, abstract2, error = _get_abstracts(articulos, id1, id2)
    if error:
        return error

    distancia = calcular_distancia_levenshtein(abstract1, abstract2)
    longitud_max = max(len(abstract1), len(abstract2))
    similitud = 1 - (distancia / longitud_max) if longitud_max > 0 else 1.0

    return {
        "articulo1_id": id1,
        "articulo2_id": id2,
        "algoritmo": "Distancia de Levenshtein",
        "similitud": round(similitud, 4),
        "explicacion": EXPLICACIONES["levenshtein"]
    }

def analizar_similitud_coseno(articulos, id1, id2):
    abstract1, abstract2, error = _get_abstracts(articulos, id1, id2)
    if error:
        return error

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([abstract1, abstract2])
    similitud = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return {
        "articulo1_id": id1,
        "articulo2_id": id2,
        "algoritmo": "Similitud de Coseno (TF-IDF)",
        "similitud": round(similitud[0][0], 4),
        "explicacion": EXPLICACIONES["coseno"]
    }

def analizar_similitud_jaccard(articulos, id1, id2):
    abstract1, abstract2, error = _get_abstracts(articulos, id1, id2)
    if error:
        return error

    a = set(abstract1.lower().split())
    b = set(abstract2.lower().split())
    interseccion = len(a.intersection(b))
    union = len(a.union(b))
    similitud = interseccion / union if union != 0 else 0
    
    return {
        "articulo1_id": id1,
        "articulo2_id": id2,
        "algoritmo": "Índice de Jaccard",
        "similitud": round(similitud, 4),
        "explicacion": EXPLICACIONES["jaccard"]
    }

def analizar_similitud_dice(articulos, id1, id2):
    abstract1, abstract2, error = _get_abstracts(articulos, id1, id2)
    if error:
        return error

    a = set(abstract1.lower().split())
    b = set(abstract2.lower().split())
    interseccion = len(a.intersection(b))
    similitud = (2 * interseccion) / (len(a) + len(b)) if (len(a) + len(b)) > 0 else 0

    return {
        "articulo1_id": id1,
        "articulo2_id": id2,
        "algoritmo": "Coeficiente de Sørensen-Dice",
        "similitud": round(similitud, 4),
        "explicacion": EXPLICACIONES["dice"]
    }

def analizar_similitud_ia(articulos, id1, id2, model_name):
    abstract1, abstract2, error = _get_abstracts(articulos, id1, id2)
    if error:
        return error

    # Cargar modelo desde la caché o descargarlo si es la primera vez
    if model_name not in model_cache:
        print(f"[INFO] Cargando modelo de IA: {model_name}. Esto puede tardar un momento...")
        model_cache[model_name] = SentenceTransformer(model_name)
        print(f"[INFO] Modelo {model_name} cargado.")
    
    model = model_cache[model_name]
    
    # Codificar los abstracts a embeddings
    embedding1 = model.encode(abstract1, convert_to_tensor=True)
    embedding2 = model.encode(abstract2, convert_to_tensor=True)
    
    # Calcular similitud de coseno entre los embeddings
    similitud = util.pytorch_cos_sim(embedding1, embedding2)

    explicacion_key = "ia_mini_lm" if "MiniLM" in model_name else "ia_paraphrase"

    return {
        "articulo1_id": id1,
        "articulo2_id": id2,
        "algoritmo": f"IA - Sentence Transformer ({model_name})",
        "similitud": round(similitud.item(), 4),
        "explicacion": EXPLICACIONES[explicacion_key]
    }

# --- Ejemplo de uso (para pruebas) ---
if __name__ == '__main__':
    lista_articulos = cargar_articulos()
    if lista_articulos and len(lista_articulos) >= 2:
        id_articulo_1 = lista_articulos[0].get('ID')
        id_articulo_2 = lista_articulos[1].get('ID')

        print(f"\nComparando artículo {id_articulo_1} y {id_articulo_2}")

        print("\n--- Levenshtein ---")
        print(analizar_similitud_levenshtein(lista_articulos, id_articulo_1, id_articulo_2))

        print("\n--- Coseno (TF-IDF) ---")
        print(analizar_similitud_coseno(lista_articulos, id_articulo_1, id_articulo_2))

        print("\n--- Jaccard ---")
        print(analizar_similitud_jaccard(lista_articulos, id_articulo_1, id_articulo_2))

        print("\n--- Dice ---")
        print(analizar_similitud_dice(lista_articulos, id_articulo_1, id_articulo_2))

        print("\n--- IA (MiniLM) ---")
        print(analizar_similitud_ia(lista_articulos, id_articulo_1, id_articulo_2, 'all-MiniLM-L6-v2'))

        print("\n--- IA (Paraphrase) ---")
        print(analizar_similitud_ia(lista_articulos, id_articulo_1, id_articulo_2, 'paraphrase-mpnet-base-v2'))