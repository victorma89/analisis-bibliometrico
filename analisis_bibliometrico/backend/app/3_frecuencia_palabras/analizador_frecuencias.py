import os
import bibtexparser
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from bibtexparser.bparser import BibTexParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# --- Funciones de Ploteo ---
def _plot_frequencies_to_base64(frequencies: dict, title: str, x_label: str, y_label: str) -> str:
    """Genera un gráfico de barras a partir de un diccionario de frecuencias y lo devuelve como base64."""
    # Ordenar por frecuencia para una mejor visualización
    sorted_items = sorted(frequencies.items(), key=lambda item: item[1])
    words = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    plt.figure(figsize=(12, 8))
    bars = plt.bar(words, counts, color='skyblue')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Añadir etiquetas de valor en las barras
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center')

    plt.tight_layout()

    # Guardar en buffer y convertir a base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

# --- Funciones Auxiliares ---
def cargar_base_de_datos(ruta_archivo_bib):
    """Carga una base de datos BibTeX desde un archivo."""
    if not os.path.exists(ruta_archivo_bib):
        return None
    with open(ruta_archivo_bib, 'r', encoding='utf-8') as bibtex_file:
        parser = BibTexParser(common_strings=False)
        parser.ignore_errors = True
        return bibtexparser.load(bibtex_file, parser=parser)

def encontrar_articulos_con_abstract(db):
    """Encuentra y devuelve una lista de artículos que tienen un abstract."""
    return [entrada for entrada in db.entries if 'abstract' in entrada and entrada['abstract'].strip()]

# --- Lógica Principal del Análisis ---
def calcular_frecuencia_palabras_dadas(abstracts, palabras_clave):
    """Calcula la frecuencia de aparición de una lista dada de palabras clave."""
    texto_completo = ' '.join(abstracts).lower()
    frecuencias = {}
    for palabra in palabras_clave:
        frecuencias[palabra] = len(re.findall(r'\b' + re.escape(palabra.lower()) + r'\b', texto_completo))
    return frecuencias

def generar_y_contar_nuevas_palabras(abstracts, num_palabras=15):
    """Genera nuevas palabras clave con TF-IDF y calcula su frecuencia."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=num_palabras)
    vectorizer.fit_transform(abstracts)
    palabras_generadas = vectorizer.get_feature_names_out()
    
    # Ahora, calcula la frecuencia de estas palabras generadas en el corpus
    frecuencias_generadas = calcular_frecuencia_palabras_dadas(abstracts, palabras_generadas)
    
    return frecuencias_generadas

def calcular_precision_nuevas_palabras(palabras_generadas, palabras_originales):
    """Calcula la precisión de las palabras generadas en comparación con una lista original."""
    set_generadas = set(palabras_generadas)
    set_originales = set(palabra.lower() for palabra in palabras_originales)
    palabras_comunes = set_generadas.intersection(set_originales)
    precision = len(palabras_comunes) / len(palabras_generadas) if len(palabras_generadas) > 0 else 0
    return precision, list(palabras_comunes)

# --- Función Orquestadora ---
def analizar_frecuencias_completo(abstracts: list, palabras_clave_dadas: list):
    """
    Ejecuta el flujo completo del análisis de frecuencia y devuelve todos los resultados.
    """
    # 1. Frecuencia de palabras dadas
    frecuencias_dadas = calcular_frecuencia_palabras_dadas(abstracts, palabras_clave_dadas)
    grafico_dadas = _plot_frequencies_to_base64(
        frecuencias_dadas,
        title="Frecuencia de Palabras Clave Predefinidas",
        x_label="Palabras Clave",
        y_label="Frecuencia de Aparición"
    )

    # 2. Generación y frecuencia de nuevas palabras
    frecuencias_generadas = generar_y_contar_nuevas_palabras(abstracts, num_palabras=15)
    grafico_generadas = _plot_frequencies_to_base64(
        frecuencias_generadas,
        title="Frecuencia de Nuevas Palabras Clave Generadas (TF-IDF)",
        x_label="Palabras Clave Generadas",
        y_label="Frecuencia de Aparición"
    )

    # 3. Precisión
    palabras_generadas_lista = list(frecuencias_generadas.keys())
    precision, comunes = calcular_precision_nuevas_palabras(palabras_generadas_lista, palabras_clave_dadas)

    return {
        "frecuencias_dadas": frecuencias_dadas,
        "grafico_frecuencias_dadas": grafico_dadas,
        "frecuencias_generadas": frecuencias_generadas,
        "grafico_frecuencias_generadas": grafico_generadas,
        "precision": precision,
        "palabras_comunes": comunes
    }