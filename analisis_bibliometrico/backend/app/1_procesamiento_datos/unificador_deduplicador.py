import os
import unicodedata
import csv
import bibtexparser
from collections import defaultdict
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bparser import BibTexParser

# ---------- Normalización ----------
def limpiar_texto(texto):
    """
    Normaliza texto: quita llaves, baja a minúsculas, quita acentos y dobles espacios.
    """
    if not texto:
        return ''
    limpio = texto.replace('{', '').replace('}', '').lower().strip()
    limpio = unicodedata.normalize('NFKD', limpio).encode('ascii', 'ignore').decode('utf-8')
    limpio = ' '.join(limpio.split())
    return limpio

def clave_unica(entrada):
    """
    Genera una clave de unicidad.
    Prioridad:
      1. DOI (si existe).
      2. Autor + Año + Título.
      3. ID interno.
    """
    if 'doi' in entrada and entrada['doi']:
        return limpiar_texto(entrada['doi'])
    
    partes = []
    if 'author' in entrada:
        partes.append(limpiar_texto(entrada['author']))
    if 'year' in entrada:
        partes.append(limpiar_texto(entrada['year']))
    if 'title' in entrada:
        partes.append(limpiar_texto(entrada['title']))
    
    if partes:
        return '||'.join(partes)
    
    return limpiar_texto(entrada.get('ID', ''))

# ---------- Proceso principal ----------
def unificar_y_deduplicar(directorio_descargas, archivo_unicos, archivo_duplicados_csv):
    """
    Lee todos los .bib en el directorio y subdirectorios, unifica, elimina duplicados
    y guarda:
        - Únicos en un archivo .bib
        - Duplicados resumidos en un CSV
    """
    parser = BibTexParser(common_strings=False)
    parser.ignore_errors = True
    db_combinada = bibtexparser.bibdatabase.BibDatabase()
    
    # Leer todos los .bib
    for dirpath, _, filenames in os.walk(directorio_descargas):
        for nombre_archivo in filenames:
            if nombre_archivo.endswith(".bib"):
                ruta_archivo = os.path.join(dirpath, nombre_archivo)
                print(f"Procesando archivo: {ruta_archivo}...")
                
                try:
                    with open(ruta_archivo, 'r', encoding='utf-8') as bibtex_file:
                        db = bibtexparser.load(bibtex_file, parser=parser)
                        db_combinada.entries.extend(db.entries)
                except Exception as e:
                    print(f"  Error al procesar {ruta_archivo}: {e}")

    print(f"\nTotal de entradas encontradas: {len(db_combinada.entries)}")

    # Contadores por clave
    contador = defaultdict(int)
    entradas_representativas = {}

    for entrada in db_combinada.entries:
        clave = clave_unica(entrada)
        contador[clave] += 1
        if clave not in entradas_representativas:
            entradas_representativas[clave] = entrada

    # Siempre guardamos una copia representativa en únicos
    entradas_unicas = list(entradas_representativas.values())

    # Guardamos solo las que tienen más de una repetición en el CSV
    duplicados_resumen = []
    for clave, entrada in entradas_representativas.items():
        if contador[clave] > 1:
            duplicados_resumen.append({
                "clave": clave,
                "titulo": entrada.get("title", ""),
                "autor": entrada.get("author", ""),
                "anio": entrada.get("year", ""),
                "repeticiones": contador[clave]
            })

    print("Proceso de deduplicación completado.")
    print(f" - Únicas (guardadas en .bib): {len(entradas_unicas)}")
    print(f" - Duplicados distintos (en CSV): {len(duplicados_resumen)}")

    # Guardar únicas en .bib
    writer = BibTexWriter()
    writer.indent = '    '
    writer.comma_first = False

    db_unicas = bibtexparser.bibdatabase.BibDatabase()
    db_unicas.entries = entradas_unicas
    with open(archivo_unicos, 'w', encoding='utf-8') as bibfile:
        bibfile.write(writer.write(db_unicas))
    print(f"\nArchivo 'únicos' guardado en: {archivo_unicos}")

    # Guardar duplicados en CSV resumido
    with open(archivo_duplicados_csv, 'w', encoding='utf-8', newline='') as csvfile:
        writer_csv = csv.DictWriter(csvfile, fieldnames=["clave", "titulo", "autor", "anio", "repeticiones"])
        writer_csv.writeheader()
        for fila in duplicados_resumen:
            writer_csv.writerow(fila)
    print(f"Archivo de resumen de duplicados guardado en: {archivo_duplicados_csv}")


# ---------- Main ----------
if __name__ == '__main__':
    directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    
    directorio_descargas = os.path.join(directorio_raiz, 'datos', 'descargas')
    directorio_procesados = os.path.join(directorio_raiz, 'datos', 'procesados')
    
    os.makedirs(directorio_procesados, exist_ok=True)
    
    archivo_unicos = os.path.join(directorio_procesados, 'articulos_unicos.bib')
    archivo_duplicados_csv = os.path.join(directorio_procesados, 'articulos_duplicados.csv')
    
    if not os.path.isdir(directorio_descargas):
        print(f"Error: no se encontró el directorio de descargas en '{directorio_descargas}'")
    else:
        unificar_y_deduplicar(directorio_descargas, archivo_unicos, archivo_duplicados_csv)
        print("\n--- PROCESO DE UNIFICACIÓN Y DEDUPLICACIÓN FINALIZADO ---")
        print("--- PROCESO COMPLETADO ---")