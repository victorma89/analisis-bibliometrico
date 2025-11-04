import asyncio
import subprocess
import os
import sys
import traceback
from pathlib import Path
from typing import List
from queue import Queue
from threading import Thread
from queue import Queue
import networkx as nx
from fastapi import FastAPI, Query

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import importlib

# Importar la lógica de los requerimientos de forma dinámica
analizador_similitud = importlib.import_module("app.2_similitud_texto.analizador_similitud")
analizador_ordenamiento = importlib.import_module("app.3_analisis_ordenamiento.analizador_ordenamiento")
analizador_frecuencias = importlib.import_module("app.3_frecuencia_palabras.analizador_frecuencias")
analizador_cluster = importlib.import_module("app.4_agrupamiento_jerarquico.analizador_cluster")
generador_visualizaciones = importlib.import_module("app.5_analisis_visual.generador_visualizaciones")
analizador_grafos = importlib.import_module("app.6_analisis_grafos.analizador_grafos")
analizador_citaciones = importlib.import_module("app.1_procesamiento_datos.analizador_citaciones")
analisis_concurrencia = importlib.import_module("app.7_concurrencia.analisis_concurrencia")


# Fix para NotImplementedError en Windows con asyncio.subprocess
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

app = FastAPI()

# --- Montar directorio estático ---
static_dir = Path(__file__).resolve().parent.parent / "frontend" / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# --- Configuración de rutas ---
BASE_DIR = Path(__file__).resolve().parent.parent
templates_dir = os.path.join(str(BASE_DIR), "frontend", "templates")
print(f"DEBUG: La ruta de los templates es: {templates_dir}")
templates = Jinja2Templates(directory=templates_dir)

PYTHON_EXECUTABLE = sys.executable

# PROJECT_ROOT → carpeta analisis_bibliometrico
PROJECT_ROOT = BASE_DIR

# Ruta al script de scraping
SCRAPER_SCRIPT_PATH = PROJECT_ROOT / "backend" / "app" / "1_procesamiento_datos" / "web_scraper.py"
SCRAPER_WORKING_DIR = SCRAPER_SCRIPT_PATH.parent

# --- Modelos de Datos (Pydantic) ---
class AnalisisSimilitudRequest(BaseModel):
    article_ids: List[str]
    algoritmo: str

# --- Rutas y Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- Endpoints para Requerimiento 1 ---
DEDUPLICATOR_SCRIPT_PATH = PROJECT_ROOT / "backend" / "app" / "1_procesamiento_datos" / "unificador_deduplicador.py"

def _run_in_thread_and_stream_to_queue(command: List[str], working_dir: str, q: Queue, input_data: str = None):
    """
    Ejecuta un comando en un hilo y pone su salida en una cola.
    Esta función es SÍNCRONA y está diseñada para ser ejecutada en un hilo separado.
    """
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Combina stderr con stdout
            cwd=working_dir,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )

        if input_data:
            process.stdin.write(input_data)
            process.stdin.close()

        # Leer la salida línea por línea en tiempo real
        for line in iter(process.stdout.readline, ''):
            q.put(line)
        
        process.stdout.close()
        process.wait()

    except Exception:
        tb_str = traceback.format_exc()
        q.put(f"\n--- ERROR CRÍTICO EN EL HILO DE EJECUCIÓN ---\n{tb_str}")
    finally:
        q.put(None) # Centinela para indicar que el proceso ha terminado

async def streamer(q: Queue):
    """
    Generador asíncrono que lee de una cola hasta que encuentra el centinela (None).
    """
    loop = asyncio.get_running_loop()
    while True:
        # Ejecuta la operación de lectura de la cola (que es bloqueante) en un hilo
        line = await loop.run_in_executor(None, q.get)
        if line is None:
            break
        yield line

async def run_scraper_and_deduplicator(database: str, email: str, password: str):
    """
    Orquesta la ejecución de los scripts y transmite su salida.
    """
    q = Queue()

    # --- 1. Ejecutar Scraper ---
    yield "--- INICIANDO PROCESO DE SCRAPING ---\n"
    scraper_command = [
        PYTHON_EXECUTABLE, "-u", str(SCRAPER_SCRIPT_PATH),
        "--database", database, "--email", email
    ]
    scraper_thread = Thread(
        target=_run_in_thread_and_stream_to_queue,
        args=(scraper_command, str(SCRAPER_WORKING_DIR), q, password)
    )
    scraper_thread.start()
    async for line in streamer(q):
        yield line
    scraper_thread.join()

    # --- 2. Ejecutar Deduplicador ---
    yield "\n--- PROCESO DE SCRAPING FINALIZADO ---\n"
    yield "\n--- INICIANDO PROCESO DE UNIFICACIÓN Y DEDUPLICACIÓN ---\n"
    deduplicator_q = Queue()
    deduplicator_command = [PYTHON_EXECUTABLE, "-u", str(DEDUPLICATOR_SCRIPT_PATH)]
    deduplicator_thread = Thread(
        target=_run_in_thread_and_stream_to_queue,
        args=(deduplicator_command, str(PROJECT_ROOT), deduplicator_q)
    )
    deduplicator_thread.start()
    async for line in streamer(deduplicator_q):
        yield line
    deduplicator_thread.join()

    yield "\n--- PROCESO DE UNIFICACIÓN Y DEDUPLICACIÓN FINALIZADO ---\n"

@app.post("/run-scraper")
async def run_scraper(
    database: str = Form(...),
    email: str = Form(...),
    password: str = Form(...)
):
    """
    Ejecuta el scraper y luego el unificador, transmitiendo la salida en tiempo real.
    """
    return StreamingResponse(run_scraper_and_deduplicator(database, email, password), media_type="text/plain")

# --- Endpoints para Requerimiento 2 ---
@app.get("/articulos")
async def get_articulos():
    """
    Carga y devuelve la lista de artículos desde el archivo .bib.
    """
    articulos = analizador_similitud.cargar_articulos()
    if not articulos:
        return JSONResponse(content={"error": "No se pudo cargar la lista de artículos. Asegúrate de haber generado el archivo 'articulos_unicos.bib' primero."}, status_code=404)
    
    articulos_simplificados = [
        {"id": articulo.get('ID', ''), "title": articulo.get('title', 'Sin título')}
        for articulo in articulos
    ]
    return JSONResponse(content=articulos_simplificados)

@app.post("/analizar-similitud")
async def analizar_similitud(request_data: AnalisisSimilitudRequest):
    """
    Recibe dos IDs de artículos y el algoritmo a usar, y calcula la similitud.
    """
    if len(request_data.article_ids) != 2:
        return JSONResponse(content={"error": "Por favor, selecciona exactamente dos artículos para comparar."}, status_code=400)

    id1 = request_data.article_ids[0]
    id2 = request_data.article_ids[1]
    algoritmo = request_data.algoritmo

    articulos = analizador_similitud.cargar_articulos()
    if not articulos:
        return JSONResponse(content={"error": "No se pudo cargar la lista de artículos."}, status_code=500)

    if algoritmo == "levenshtein":
        resultado = analizador_similitud.analizar_similitud_levenshtein(articulos, id1, id2)
    elif algoritmo == "coseno":
        resultado = analizador_similitud.analizar_similitud_coseno(articulos, id1, id2)
    elif algoritmo == "jaccard":
        resultado = analizador_similitud.analizar_similitud_jaccard(articulos, id1, id2)
    elif algoritmo == "dice":
        resultado = analizador_similitud.analizar_similitud_dice(articulos, id1, id2)
    elif algoritmo == "ia_mini_lm":
        resultado = analizador_similitud.analizar_similitud_ia(articulos, id1, id2, 'all-MiniLM-L6-v2')
    elif algoritmo == "ia_paraphrase":
        resultado = analizador_similitud.analizar_similitud_ia(articulos, id1, id2, 'paraphrase-mpnet-base-v2')
    else:
        return JSONResponse(content={"error": f"Algoritmo '{algoritmo}' no reconocido."}, status_code=400)

    if "error" in resultado:
        return JSONResponse(content=resultado, status_code=404)

    return JSONResponse(content=resultado)

# --- Endpoints para Requerimiento 3 ---
BIB_FILE_PATH = PROJECT_ROOT / "datos" / "procesados" / "articulos_unicos.bib"

@app.post("/analizar-frecuencia")
async def analizar_frecuencia():
    """
    Ejecuta el análisis de frecuencia de palabras, incluyendo la generación de gráficos.
    """
    if not BIB_FILE_PATH.exists():
        return JSONResponse(
            content={"error": f"El archivo '{BIB_FILE_PATH.name}' no existe. Asegúrate de generarlo primero con el Requerimiento 1."}, 
            status_code=404
        )

    try:
        db = analizador_frecuencias.cargar_base_de_datos(str(BIB_FILE_PATH))
        if not db:
            return JSONResponse(content={"error": "No se pudo cargar la base de datos BibTeX."}, status_code=500)

        articulos_con_abstract = analizador_frecuencias.encontrar_articulos_con_abstract(db)
        if not articulos_con_abstract:
            return JSONResponse(content={"error": "No se encontraron artículos con abstract en el archivo."}, status_code=404)

        abstracts = [entry['abstract'] for entry in articulos_con_abstract]

        palabras_clave_dadas = [
            "Generative models", "Prompting", "Machine learning", "Multimodality",
            "Fine-tuning", "Training data", "Algorithmic bias", "Explainability",
            "Transparency", "Ethics", "Privacy", "Personalization",
            "Human-AI interaction", "AI literacy", "Co-creation"
        ]

        loop = asyncio.get_running_loop()
        
        # Ejecutar el análisis completo en un hilo
        resultado_completo = await loop.run_in_executor(
            None, 
            analizador_frecuencias.analizar_frecuencias_completo, 
            abstracts, 
            palabras_clave_dadas
        )

        return JSONResponse(content=resultado_completo)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"Ocurrió un error inesperado durante el análisis de frecuencia: {e}"},
            status_code=500
        )

# --- Endpoints para Requerimiento 4 ---
@app.post("/analizar-agrupamiento")
async def analizar_agrupamiento(metodo: str = Form(...), num_articulos: int = Form(...)):
    """
    Ejecuta el análisis de agrupamiento jerárquico.
    """
    if not BIB_FILE_PATH.exists():
        return JSONResponse(
            content={"error": f"El archivo '{BIB_FILE_PATH.name}' no existe. Genéralo primero con el Requerimiento 1."}, 
            status_code=404
        )

    loop = asyncio.get_running_loop()
    try:
        resultado = await loop.run_in_executor(
            None,
            analizador_cluster.analizar_agrupamiento_completo,
            metodo,
            num_articulos
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"Ocurrió un error inesperado durante el análisis de agrupamiento: {e}"},
            status_code=500
        )

    if "error" in resultado:
        return JSONResponse(content=resultado, status_code=500)
        
    return JSONResponse(content=resultado)

# --- Endpoints para Requerimiento 5 ---
class VisualizacionesPDFRequest(BaseModel):
    mapa_calor: str
    nube_palabras: str
    linea_tiempo: str
    linea_tiempo_revista: str
    mapa_calor_pdf_data: list

@app.post("/generar-visualizaciones")
async def generar_visualizaciones():
    """
    Ejecuta la generación de todas las visualizaciones del Requerimiento 5.
    """
    if not BIB_FILE_PATH.exists():
        return JSONResponse(
            content={"error": f"El archivo '{BIB_FILE_PATH.name}' no existe. Genéralo primero con el Requerimiento 1."}, 
            status_code=404
        )

    loop = asyncio.get_running_loop()
    try:
        resultado = await loop.run_in_executor(
            None,
            generador_visualizaciones.generar_visualizaciones_completo
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"Ocurrió un error inesperado durante la generación de visualizaciones: {e}"},
            status_code=500
        )

    if "error" in resultado and resultado["error"]:
        return JSONResponse(content=resultado, status_code=500)
        
    return JSONResponse(content=resultado)

@app.post("/exportar-visualizaciones-pdf")
async def exportar_visualizaciones_pdf(request_data: VisualizacionesPDFRequest):
    """
    Recibe las imágenes en base64 y las exporta a un archivo PDF.
    """
    loop = asyncio.get_running_loop()
    try:
        imagenes_dict = {
            "mapa_calor": request_data.mapa_calor,
            "nube_palabras": request_data.nube_palabras,
            "linea_tiempo": request_data.linea_tiempo,
            "linea_tiempo_revista": request_data.linea_tiempo_revista,
        }
        datos_mapa = request_data.mapa_calor_pdf_data

        resultado = await loop.run_in_executor(
            None,
            generador_visualizaciones.exportar_visualizaciones_pdf,
            imagenes_dict,
            datos_mapa
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"Ocurrió un error inesperado durante la exportación a PDF: {e}"},
            status_code=500
        )

    if "error" in resultado and resultado["error"] is not None:
        return JSONResponse(content=resultado, status_code=500)
        
    return JSONResponse(content=resultado)

# --- Endpoints para Seguimiento 2 (Análisis de Grafos/Citaciones) ---
@app.post("/analizar-citaciones")
async def analizar_citaciones_endpoint(
    tipo_analisis: str = Form(...),
    sim_threshold: float = Form(0.25),
    origen: str = Form(None),
    destino: str = Form(None)
):
    """
    Ejecuta el análisis de grafos de citaciones.
    """
    if not BIB_FILE_PATH.exists():
        return JSONResponse(
            content={"error": f"El archivo '{BIB_FILE_PATH.name}' no existe. Genéralo primero."}, 
            status_code=404
        )

    try:
        loop = asyncio.get_running_loop()
        
        # Cargar artículos y construir el grafo (pasos comunes)
        articles = await loop.run_in_executor(None, analizador_grafos.load_bib_file, str(BIB_FILE_PATH))
        G = await loop.run_in_executor(None, analizador_grafos.infer_graph, articles, sim_threshold)

        if tipo_analisis == "resumen_grafo":
            resultado = await loop.run_in_executor(None, analizador_grafos.analizar_grafo, G)
        
        elif tipo_analisis == "exportar_json":
            resultado = await loop.run_in_executor(None, analizador_grafos.export_graph_to_json, G)

        elif tipo_analisis == "componentes_conexas":
            resultado = await loop.run_in_executor(None, analizador_grafos.identificar_componentes_fuertemente_conexas, G)

        elif tipo_analisis == "camino_minimo":
            if not origen or not destino:
                return JSONResponse(content={"error": "Se requieren nodos de origen y destino para calcular el camino mínimo."}, status_code=400)
            resultado = await loop.run_in_executor(None, analizador_grafos.caminos_minimos, G, origen, destino)
        
        else:
            return JSONResponse(content={"error": f"Tipo de análisis no soportado: {tipo_analisis}"}, status_code=400)

        return JSONResponse(content=resultado)

    except FileNotFoundError as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"Ocurrió un error inesperado durante el análisis de grafos: {e}"},
            status_code=500
        )

# --- Endpoints para Seguimiento 1 ---
@app.post("/analizar-ordenamiento")
async def post_analizar_ordenamiento(size: int = Form(...), algorithm: str = Form(...)):
    """
    Ejecuta el análisis completo de los algoritmos de ordenamiento.
    """
    if not BIB_FILE_PATH.exists():
        return JSONResponse(
            content={"error": f"El archivo '{BIB_FILE_PATH.name}' no existe. Asegúrate de generarlo primero con el Requerimiento 1."}, 
            status_code=404
        )

    # La ejecución puede ser pesada, se ejecuta en un hilo para no bloquear
    loop = asyncio.get_running_loop()
    try:
        resultado = await loop.run_in_executor(
            None,  # Usa el executor por defecto (ThreadPoolExecutor)
            analizador_ordenamiento.analizar_ordenamiento_completo,
            str(BIB_FILE_PATH),
            size,
            algorithm
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"Ocurrió un error inesperado durante el análisis: {e}"},
            status_code=500
        )

    if "error" in resultado:
        return JSONResponse(content=resultado, status_code=500)
        
    return JSONResponse(content=resultado)


RESULTS_DIR = PROJECT_ROOT / "datos" / "resultados_ordenados"
@app.post("/grafo-citaciones")
async def grafo_citaciones(
    min_citaciones: int | None = Query(None, description="Mínimo de citaciones para filtrar")
):
    """
    PUNTO 1: Construcción automática del grafo de citaciones
    
    Genera y devuelve el grafo completo con:
    - Análisis estadístico
    - Visualización base64 (imagen estática)
    - JSON para vis-network (interactivo)
    
    Query params:
        - min_citaciones: Filtrar artículos con al menos N citaciones
    """
    try:
        bib_path = PROJECT_ROOT / "datos" / "procesados" / "articulos_unicos.bib"
        if not bib_path.exists():
            return JSONResponse(
                content={"error": f"No existe el archivo: {bib_path}"},
                status_code=404
            )

        print(f"\n{'='*70}")
        print("🚀 GENERANDO GRAFO DE CITACIONES")
        print(f"{'='*70}")

        # 1️⃣ Cargar artículos y construir grafo
        articulos = analizador_citaciones.load_bib_file(str(bib_path))
        G = analizador_citaciones.infer_graph(articulos, thr=0.25, verbose=True)

        # 2️⃣ Filtrar nodos según min_citaciones si aplica
        if min_citaciones is not None and min_citaciones > 0:
            nodos_validos = [
                n for n in G.nodes()
                if G.nodes[n].get('citaciones', 0) >= min_citaciones
            ]
            G = G.subgraph(nodos_validos).copy()
            print(f"🔍 Grafo filtrado: {G.number_of_nodes()} nodos con ≥{min_citaciones} citaciones")

        # 3️⃣ Exportar grafo y análisis
        grafo_json = analizador_citaciones.export_graph_to_json(G)
        analisis = analizador_citaciones.analizar_grafo(G)

        # 4️⃣ Generar imagen base64 (PRINCIPAL)
        grafico_base64 = None
        try:
            grafico_base64 = analizador_citaciones.graficar_grafo_citaciones(
                G,
                titulo="Red de Citaciones Inferida por Similitud"
            )
            print("✅ Visualización base64 generada")
        except Exception as e:
            print(f"⚠️ Error generando visualización: {e}")

        print(f"{'='*70}\n")

        # 5️⃣ Devolver JSON unificado
        return JSONResponse(content={
            "grafo": grafo_json,
            "analisis": analisis,
            "grafico_base64": grafico_base64,
            "success": True
        })

    except FileNotFoundError as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"Error al generar el grafo: {str(e)}"},
            status_code=500
        )

@app.get("/grafo-citaciones/valores")
async def valores_citaciones():
    """
    GET /grafo-citaciones/valores
    
    Devuelve lista de valores sugeridos para el filtro de citaciones
    basado en los grados de entrada (in-degree) del grafo.
    """
    try:
        bib_path = PROJECT_ROOT / "datos" / "procesados" / "articulos_unicos.bib"
        if not bib_path.exists():
            return JSONResponse(
                content={"error": f"No existe {bib_path}"}, 
                status_code=404
            )

        articulos = analizador_citaciones.load_bib_file(str(bib_path))
        G = analizador_citaciones.infer_graph(articulos, thr=0.25, verbose=False)

        # Obtener citaciones de todos los nodos
        citaciones = [G.nodes[n].get('citaciones', 0) for n in G.nodes()]
        citaciones_validas = [c for c in citaciones if c > 0]

        if not citaciones_validas:
            return JSONResponse(content={"valores": [10, 5, 2, 1]})

        # Calcular valores representativos
        max_cit = max(citaciones_validas)
        
        valores = []
        if max_cit >= 4:
            valores.extend([
                max_cit,
                max_cit // 2,
                max_cit // 4,
                max(1, max_cit // 10)
            ])
        else:
            valores.extend(sorted(set(citaciones_validas), reverse=True)[:5])

        # Remover duplicados y ordenar
        valores = sorted(set(valores), reverse=True)

        print(f"📊 Valores de citaciones calculados: {valores}")
        return JSONResponse(content={"valores": valores[:6]})

    except Exception as e:
        print(f"❌ Error en valores_citaciones: {e}")
        traceback.print_exc()
        return JSONResponse(content={"valores": [10, 5, 2, 1]})


@app.post("/analizar-grafo")
async def post_analizar_grafo(request: Request):
    """
    POST que admite:
      { "thr": 0.25 }  -> devuelve grafo + analisis
      { "thr":0.25, "origen": "ID1", "destino":"ID2" } -> además devuelve camino mínimo
    """
    try:
        body = await request.json()
        thr = float(body.get("thr", 0.25))
        origen = body.get("origen", None)
        destino = body.get("destino", None)

        bib_path = PROJECT_ROOT / "datos" / "procesados" / "articulos_unicos.bib"
        if not bib_path.exists():
            return JSONResponse(content={"error": f"No existe {bib_path}"}, status_code=404)

        articulos = analizador_citaciones.load_bib_file(str(bib_path))
        G = analizador_citaciones.infer_graph(articulos, thr=thr)
        grafo_json = analizador_citaciones.export_graph_to_json(G)
        analisis = analizador_citaciones.analizar_grafo(G)
        respuesta = {"grafo": grafo_json, "analisis": analisis}

        if origen is not None and destino is not None:
            # asegurar tipo string (IDs en .bib suelen ser strings)
            camino = analizador_citaciones.caminos_minimos(G, str(origen), str(destino))
            respuesta["camino_minimo"] = camino

        return JSONResponse(content=respuesta)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)
@app.get("/camino-minimo")
async def get_camino_minimo(
    origen: str = Query(..., description="ID del artículo origen"),
    destino: str = Query(..., description="ID del artículo destino"),
    algoritmo: str = Query("dijkstra", description="dijkstra o floyd-warshall")
):
    """
    PUNTO 2: Cálculo de caminos mínimos (Dijkstra / Floyd-Warshall)
    
    Calcula el camino más corto entre dos artículos usando:
    - Dijkstra (recomendado para grafos dispersos)
    - Floyd-Warshall (para análisis global)
    
    Query params:
        - origen: ID del artículo inicial
        - destino: ID del artículo final
        - algoritmo: "dijkstra" (default) o "floyd-warshall"
    """
    try:
        bib_path = PROJECT_ROOT / "datos" / "procesados" / "articulos_unicos.bib"
        if not bib_path.exists():
            return JSONResponse(
                content={"error": f"No existe {bib_path}"}, 
                status_code=404
            )

        print(f"\n{'='*70}")
        print(f"🔍 CALCULANDO CAMINO MÍNIMO: {origen} → {destino}")
        print(f"   Algoritmo: {algoritmo.upper()}")
        print(f"{'='*70}")

        # Cargar y construir grafo
        articulos = analizador_citaciones.load_bib_file(str(bib_path))
        G = analizador_citaciones.infer_graph(articulos, thr=0.25, verbose=False)

        if origen not in G:
            return JSONResponse(
                content={"error": f"El artículo origen '{origen}' no existe en el grafo."},
                status_code=404
            )
        
        if destino not in G:
            return JSONResponse(
                content={"error": f"El artículo destino '{destino}' no existe en el grafo."},
                status_code=404
            )

        # Calcular camino mínimo
        resultado = analizador_citaciones.caminos_minimos(G, origen, destino, algorithm=algoritmo)
        
        if "error" in resultado:
            print(f"⚠️ {resultado['error']}")
        else:
            print(f"✅ Camino encontrado: {len(resultado['camino'])} nodos")
            print(f"   Similitud promedio: {resultado['similitud_promedio']*100:.2f}%")
        
        print(f"{'='*70}\n")
        
        return JSONResponse(content=resultado)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"Error calculando camino: {str(e)}"}, 
            status_code=500
        )


@app.post("/analizar-citaciones")
async def analizar_citaciones_endpoint(
    tipo_analisis: str = Form(...),
    sim_threshold: float = Form(0.25),
    origen: str = Form(None),
    destino: str = Form(None)
):
    """
    PUNTO 3: Análisis de componentes fuertemente conexas
    
    POST /analizar-citaciones
    
    Tipos soportados:
    - componentes_conexas: Identifica componentes fuertemente conexas (Tarjan)
    - resumen_grafo: Estadísticas generales del grafo
    - exportar_json: Exporta el grafo en formato JSON
    - camino_minimo: Calcula camino entre origen y destino
    
    Form params:
        - tipo_analisis: Tipo de análisis a realizar
        - sim_threshold: Umbral de similitud (default: 0.25)
        - origen: ID origen (solo para camino_minimo)
        - destino: ID destino (solo para camino_minimo)
    """
    try:
        bib_path = PROJECT_ROOT / "datos" / "procesados" / "articulos_unicos.bib"
        if not bib_path.exists():
            return JSONResponse(
                content={"error": f"No existe {bib_path}"}, 
                status_code=404
            )

        print(f"\n{'='*70}")
        print(f"🔬 ANÁLISIS DE CITACIONES: {tipo_analisis.upper()}")
        print(f"{'='*70}")

        loop = asyncio.get_running_loop()
        
        # Cargar artículos y construir grafo
        articles = await loop.run_in_executor(
            None, 
            analizador_citaciones.load_bib_file, 
            str(bib_path)
        )
        G = await loop.run_in_executor(
            None, 
            analizador_citaciones.infer_graph, 
            articles, 
            sim_threshold,
            False  # verbose=False para análisis
        )

        # =====================================================
        # 🔹 COMPONENTES FUERTEMENTE CONEXAS (TARJAN)
        # =====================================================
        if tipo_analisis == "componentes_conexas":
            try:
                componentes_info = await loop.run_in_executor(
                    None,
                    analizador_citaciones.identificar_componentes_fuertemente_conexas,
                    G
                )

                # 🔧 Armar respuesta con todos los valores esperados por el frontend
                resultado = {
                    "total_componentes": componentes_info.get("total_componentes", 0),
                    "mayor_componente": {
                        "num_nodos": componentes_info.get("mayor_componente", {}).get("num_nodos", 0),
                        "num_aristas": componentes_info.get("mayor_componente", {}).get("num_aristas", 0),
                        "densidad": componentes_info.get("mayor_componente", {}).get("densidad", 0.0),
                        "grado_promedio": componentes_info.get("mayor_componente", {}).get("grado_promedio", 0.0)
                    },
                    "componentes": componentes_info.get("componentes", []),
                    "algoritmo": componentes_info.get("algoritmo", "Tarjan (NetworkX)")
                }

                print(f"✅ Componentes identificadas: {resultado['total_componentes']}")
            except Exception as e:
                traceback.print_exc()
                return JSONResponse(
                    content={"error": f"Error analizando componentes: {str(e)}"},
                    status_code=500
                )

        # =====================================================
        # 🔹 RESUMEN GENERAL DEL GRAFO
        # =====================================================
        elif tipo_analisis == "resumen_grafo":
            resultado = await loop.run_in_executor(
                None, 
                analizador_citaciones.analizar_grafo, 
                G
            )
            print(f"✅ Análisis completado: {resultado['num_nodos']} nodos")
        
        # =====================================================
        # 🔹 EXPORTAR JSON
        # =====================================================
        elif tipo_analisis == "exportar_json":
            resultado = await loop.run_in_executor(
                None, 
                analizador_citaciones.export_graph_to_json, 
                G
            )
            print(f"✅ JSON exportado")

        # =====================================================
        # 🔹 CAMINO MÍNIMO ENTRE ARTÍCULOS
        # =====================================================
        elif tipo_analisis == "camino_minimo":
            if not origen or not destino:
                return JSONResponse(
                    content={"error": "Se requieren origen y destino para calcular camino mínimo."},
                    status_code=400
                )
            resultado = await loop.run_in_executor(
                None, 
                analizador_citaciones.caminos_minimos, 
                G, 
                origen, 
                destino
            )
            print(f"✅ Camino calculado")
        
        else:
            return JSONResponse(
                content={"error": f"Tipo de análisis no soportado: {tipo_analisis}"},
                status_code=400
            )

        print(f"{'='*70}\n")
        return JSONResponse(content=resultado)

    except FileNotFoundError as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"Error en análisis de citaciones: {str(e)}"},
            status_code=500
        )


@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Permite descargar un archivo previamente generado en la carpeta de resultados.
    """
    file_path = RESULTS_DIR / filename
    if not file_path.is_file():
        return JSONResponse(content={"error": "Archivo no encontrado."}, status_code=404)
    
    return FileResponse(path=file_path, media_type='application/octet-stream', filename=filename)

REPORTS_DIR = PROJECT_ROOT / "datos" / "reportes_visuales"

@app.get("/download/reporte/{filename}")
async def download_reporte(filename: str):
    """
    Permite descargar un reporte PDF previamente generado.
    """
    file_path = REPORTS_DIR / filename
    if not file_path.is_file():
        return JSONResponse(content={"error": "Archivo de reporte no encontrado."}, status_code=404)
    
    return FileResponse(path=file_path, media_type='application/pdf', filename=filename)

@app.get("/grafo-coocurrencia")
async def grafo_coocurrencia():
    try:
        resultado = analisis_concurrencia.analizar_grafo_coocurrencia()

        return resultado  
    except Exception as e:
        print("Error en backend:", str(e))
        return {"error": f"Ocurrió un error al generar el grafo: {str(e)}"}




# Para ejecutar la aplicación:
# uvicorn main:app --reload
