import os
import bibtexparser
import pandas as pd
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import pycountry
import re
import io
import base64
from fpdf import FPDF
from collections import Counter
from datetime import datetime

# --- Constantes y Rutas ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BIB_FILE_PATH = os.path.join(ROOT_DIR, 'datos', 'procesados', 'articulos_unicos.bib')

# --- Funciones Auxiliares ---

def _cargar_datos_en_dataframe():
    """Carga el archivo .bib y lo convierte en un DataFrame de pandas."""
    if not os.path.exists(BIB_FILE_PATH):
        raise FileNotFoundError(f"El archivo .bib no se encontró en {BIB_FILE_PATH}")

    with open(BIB_FILE_PATH, 'r', encoding='utf-8') as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)
    
    df = pd.DataFrame(bib_database.entries)
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    return df

def _extraer_pais(texto: str):
    """Intenta extraer un nombre de país de una cadena de texto (afiliación)."""
    if not isinstance(texto, str):
        return None
    for country in pycountry.countries:
        if re.search(r'\b' + re.escape(country.name) + r'\b', texto, re.IGNORECASE):
            return country.alpha_3
        if hasattr(country, 'common_name') and re.search(r'\b' + re.escape(country.common_name) + r'\b', texto, re.IGNORECASE):
            return country.alpha_3
    if re.search(r'\bUSA\b', texto, re.IGNORECASE) or re.search(r'United States', texto, re.IGNORECASE):
        return 'USA'
    if re.search(r'\bUK\b', texto, re.IGNORECASE) or re.search(r'United Kingdom', texto, re.IGNORECASE):
        return 'GBR'
    if re.search(r'\bChina\b', texto, re.IGNORECASE):
        return 'CHN'
    return None

# --- Generadores de Gráficos ---

def _generar_nube_palabras(df: pd.DataFrame):
    text = ' '.join(df['abstract'].dropna().astype(str)) + ' ' + ' '.join(df['keywords'].dropna().astype(str))
    if not text.strip():
        return None, "No hay texto para generar la nube."
    wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(text)
    buf = io.BytesIO()
    wordcloud.to_image().save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8'), None

def _generar_linea_tiempo(df: pd.DataFrame):
    if 'year' not in df.columns or df['year'].isnull().all():
        return None, "No hay datos de año."
    df['year'] = pd.to_numeric(df['year'], errors='coerce').dropna()
    yearly_counts = df['year'].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    plt.plot(yearly_counts.index, yearly_counts.values, marker='o', linestyle='-')
    plt.title('Número de Publicaciones por Año')
    plt.xlabel('Año')
    plt.ylabel('Número de Publicaciones')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(yearly_counts.index.astype(int), rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_base64, None

def _generar_linea_tiempo_por_revista(df: pd.DataFrame):
    if 'year' not in df.columns or 'journal' not in df.columns or df['year'].isnull().all() or df['journal'].isnull().all():
        return None, "No hay datos de año o revista."
    counts = df.groupby(['year', 'journal']).size().reset_index(name='count')
    top_journals = df['journal'].value_counts().nlargest(10).index
    counts_top = counts[counts['journal'].isin(top_journals)]
    if counts_top.empty:
        return None, "No hay suficientes datos para el gráfico por revista."
    fig = px.line(counts_top, x='year', y='count', color='journal', title='Publicaciones por Año y Revista (Top 10)', labels={'year': 'Año', 'count': '# Publicaciones', 'journal': 'Revista'})
    fig.update_layout(legend_title_text='Revista')
    buf = io.BytesIO()
    pio.write_image(fig, buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8'), None

def _generar_mapa_calor(df: pd.DataFrame):
    if 'author' not in df.columns:
        return None, None, None, "No hay datos de autor."

    author_records = []
    for _, row in df.iterrows():
        if isinstance(row.get('author'), str):
            authors = [a.strip() for a in row['author'].split(' and ')]
            country_iso = _extraer_pais(row['author'])
            if country_iso:
                for author_name in authors:
                    if author_name:
                        author_records.append({'author_name': author_name, 'country_iso': country_iso})

    if not author_records:
        return None, None, None, "No se pudo extraer información de país y autor."

    author_df = pd.DataFrame(author_records)
    country_counts = author_df.groupby('country_iso').size().reset_index(name='count')
    
    top_authors_per_country = author_df.groupby('country_iso')['author_name'].apply(
        lambda x: ', '.join(x.value_counts().nlargest(3).index) + ('...' if x.nunique() > 3 else '')
    ).reset_index()
    top_authors_per_country.rename(columns={'author_name': 'top_authors'}, inplace=True)

    plot_df = pd.merge(country_counts, top_authors_per_country, on='country_iso')

    fig = px.choropleth(
        plot_df,
        locations="country_iso",
        color="count",
        hover_name="country_iso",
        custom_data=['top_authors', 'count']
    )
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        coloraxis_colorbar={'title':'Publicaciones'},
        title="Distribución Geográfica de Publicaciones"
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{hover_name}</b><br><br>" +
            "Publicaciones: %{customdata[1]}<br>" +
            "Autores: %{customdata[0]}" +
            "<extra></extra>"
        )
    )

    mapa_json = fig.to_json()
    mapa_base64 = pio.to_image(fig, format='png', width=800, height=600, scale=2)
    pdf_data = plot_df[['country_iso', 'count', 'top_authors']].to_dict('records')

    return mapa_json, base64.b64encode(mapa_base64).decode('utf-8'), pdf_data, None

# --- Función Orquestadora ---

def generar_visualizaciones_completo():
    try:
        df = _cargar_datos_en_dataframe()
        if df.empty:
            return {"error": "No se encontraron artículos."}

        nube_palabras_base64, error_nube = _generar_nube_palabras(df)
        linea_tiempo_base64, error_linea = _generar_linea_tiempo(df)
        linea_tiempo_revista_base64, error_linea_revista = _generar_linea_tiempo_por_revista(df)
        mapa_calor_json, mapa_calor_base64, mapa_pdf_data, error_mapa = _generar_mapa_calor(df)

        explicaciones = {
            "mapa_calor": "Este mapa coroplético muestra la distribución geográfica de las publicaciones. Cada país está coloreado según el número de artículos cuya afiliación se encuentra en ese país. Al pasar el mouse sobre un país, se puede ver el recuento y los autores más frecuentes de esa región.",
            "nube_palabras": "La nube de palabras visualiza la frecuencia de los términos más comunes encontrados en títulos, palabras clave y resúmenes. El tamaño de cada palabra es proporcional a su frecuencia.",
            "linea_tiempo": "Este gráfico de líneas muestra la evolución del número de publicaciones a lo largo del tiempo, permitiendo identificar tendencias en la producción científica.",
            "linea_tiempo_revista": "Este gráfico desglosa las publicaciones por año para las 10 revistas con más artículos, mostrando dónde se concentra la investigación."
        }

        return {
            "nube_palabras_base64": nube_palabras_base64,
            "linea_tiempo_base64": linea_tiempo_base64,
            "linea_tiempo_revista_base64": linea_tiempo_revista_base64,
            "mapa_calor_json": mapa_calor_json,
            "mapa_calor_base64": mapa_calor_base64,
            "mapa_calor_pdf_data": mapa_pdf_data,
            "explicaciones": explicaciones,
            "errors": {
                "nube_palabras": error_nube,
                "linea_tiempo": error_linea,
                "linea_tiempo_revista": error_linea_revista,
                "mapa_calor": error_mapa
            }
        }

    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        import traceback
        return {"error": f"Ocurrió un error inesperado: {e}\n{traceback.format_exc()}"}

# --- Función de Exportación a PDF ---

def exportar_visualizaciones_pdf(imagenes_base64: dict, datos_mapa: list):
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Reporte de Análisis Visual Bibliométrico', 0, 1, 'C')
        pdf.ln(10)

        if imagenes_base64.get('mapa_calor'):
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, '1. Distribución Geográfica de Publicaciones', 0, 1)
            img_bytes = base64.b64decode(imagenes_base64['mapa_calor'])
            img_file = io.BytesIO(img_bytes)
            pdf.image(img_file, x=10, y=pdf.get_y(), w=190)
            pdf.ln(120)

            if datos_mapa:
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 10, 'Datos del Mapa (Top 10 Países)', 0, 1)
                pdf.set_font('Arial', '', 8)
                pdf.cell(20, 6, 'País', 1)
                pdf.cell(20, 6, 'Artículos', 1)
                pdf.cell(150, 6, 'Autores Principales', 1)
                pdf.ln()
                for row in sorted(datos_mapa, key=lambda x: x['count'], reverse=True)[:10]:
                    pdf.cell(20, 6, row['country_iso'], 1)
                    pdf.cell(20, 6, str(row['count']), 1)
                    x = pdf.get_x()
                    y = pdf.get_y()
                    pdf.multi_cell(150, 6, row['top_authors'].encode('latin-1', 'replace').decode('latin-1'), 1, 'L')
                    pdf.set_xy(x + 150, y)
                    pdf.ln()
                pdf.ln(5)

        if imagenes_base64.get('nube_palabras'):
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, '2. Nube de Palabras', 0, 1)
            img_bytes = base64.b64decode(imagenes_base64['nube_palabras'])
            img_file = io.BytesIO(img_bytes)
            pdf.image(img_file, x=10, y=pdf.get_y(), w=190)
            pdf.ln(105)

        if imagenes_base64.get('linea_tiempo'):
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, '3. Publicaciones por Año', 0, 1)
            img_bytes = base64.b64decode(imagenes_base64['linea_tiempo'])
            img_file = io.BytesIO(img_bytes)
            pdf.image(img_file, x=10, y=pdf.get_y(), w=190)
            pdf.ln(100)

        if imagenes_base64.get('linea_tiempo_revista'):
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, '4. Publicaciones por Revista (Top 10)', 0, 1)
            img_bytes = base64.b64decode(imagenes_base64['linea_tiempo_revista'])
            img_file = io.BytesIO(img_bytes)
            pdf.image(img_file, x=10, y=pdf.get_y(), w=190)

        output_dir = os.path.join(ROOT_DIR, 'datos', 'reportes_visuales')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"reporte_visual_{timestamp}.pdf"
        output_filepath = os.path.join(output_dir, output_filename)
        
        pdf.output(output_filepath)

        return {"output_file": output_filename, "error": None}

    except Exception as e:
        import traceback
        return {"error": f"Ocurrió un error al generar el PDF: {e}\n{traceback.format_exc()}"}