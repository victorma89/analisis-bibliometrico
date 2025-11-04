import time
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple
import numpy as np
import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase
from collections import Counter
import io
import base64
import os
from datetime import datetime

# ==================== ALGORITMOS DE ORDENAMIENTO ====================

def tim_sort(arr):
    arr.sort()
    return arr

def comb_sort(arr):
    gap = len(arr)
    shrink = 1.3
    sorted_flag = False
    while not sorted_flag:
        gap = int(gap / shrink)
        if gap < 1:
            gap = 1
        sorted_flag = True
        for i in range(len(arr) - gap):
            if arr[i] > arr[i + gap]:
                arr[i], arr[i + gap] = arr[i + gap], arr[i]
                sorted_flag = False
    return arr

def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

def tree_sort(arr):
    class Node:
        def __init__(self, val):
            self.val = val
            self.left = None
            self.right = None
    def insert(root, val):
        if not root:
            return Node(val)
        if val < root.val:
            root.left = insert(root.left, val)
        else:
            root.right = insert(root.right, val)
        return root
    def inorder(root, result):
        if root:
            inorder(root.left, result)
            result.append(root.val)
            inorder(root.right, result)
    if not arr:
        return arr
    root = None
    for val in arr:
        root = insert(root, val)
    result = []
    inorder(root, result)
    return result

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

def heapsort(arr):
    def heapify(arr, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and arr[i] < arr[left]:
            largest = left
        if right < n and arr[largest] < arr[right]:
            largest = right
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr

def bitonic_sort(arr):
    def bitonic_merge(arr, low, cnt, direction):
        if cnt > 1:
            k = cnt // 2
            for i in range(low, low + k):
                if (arr[i] > arr[i + k]) == direction:
                    arr[i], arr[i + k] = arr[i + k], arr[i]
            bitonic_merge(arr, low, k, direction)
            bitonic_merge(arr, low + k, k, direction)
    def bitonic_sort_rec(arr, low, cnt, direction):
        if cnt > 1:
            k = cnt // 2
            bitonic_sort_rec(arr, low, k, 1)
            bitonic_sort_rec(arr, low + k, k, 0)
            bitonic_merge(arr, low, cnt, direction)
    n = len(arr)
    next_power = 1
    while next_power < n:
        next_power *= 2
    dummy_entry = (float('inf'), "")
    arr.extend([dummy_entry] * (next_power - n))
    bitonic_sort_rec(arr, 0, next_power, 1)
    return [x for x in arr if x != dummy_entry]

def gnome_sort(arr):
    i = 0
    while i < len(arr):
        if i == 0 or arr[i] >= arr[i - 1]:
            i += 1
        else:
            arr[i], arr[i - 1] = arr[i - 1], arr[i]
            i -= 1
    return arr

def binary_insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        left, right = 0, i
        while left < right:
            mid = (left + right) // 2
            if arr[mid] > key:
                right = mid
            else:
                left = mid + 1
        for j in range(i - 1, left - 1, -1):
            arr[j + 1] = arr[j]
        arr[left] = key
    return arr

def pigeonhole_sort_adapted(entries):
    if not entries:
        return entries
    min_year = min(entry[0] for entry in entries)
    max_year = max(entry[0] for entry in entries)
    range_years = max_year - min_year + 1 # aca se calcula el numero total de casilleros
    holes = [[] for _ in range(range_years)] # aca se crea el casillero con varias listas vacias
    for entry in entries: # en este ciclo se ordena por año, inicia recorriendo la lista
        year_index = entry[0] - min_year
        holes[year_index].append(entry)# el articulo se guarda en el casillero correspondiente
    sorted_entries = []
    for year_entries in holes: # aca se ordena por titulo recorriendo el casillero uno por uno
        if year_entries:
            year_entries.sort(key=lambda x: x[1].lower()) # dentro del mismo año se ordena esa pequeña lista por orden alfabetico
            sorted_entries.extend(year_entries)
    return sorted_entries

def bucket_sort_adapted(entries):
    return pigeonhole_sort_adapted(entries)

def radix_sort_adapted(entries):
    if not entries:
        return entries
    def get_digit(number, position):
        return (number // (10 ** position)) % 10
        # Este bloque de codigo se encarga de ordenar la lista de articulos teniendo en cuenta el año
    max_year = max(entry[0] for entry in entries)
    max_digits = len(str(max_year)) if max_year > 0 else 1
    result = entries[:]
    for digit_position in range(max_digits):
        counts = [[] for _ in range(10)]
        for entry in result:
            digit = get_digit(entry[0], digit_position)
            counts[digit].append(entry)
        result = []
        for digit_entries in counts:
            result.extend(digit_entries) #result solo contiene la lista de articulos ordenada por año
    final_result = []
    if result:
        from itertools import groupby 
        # Agrupamos los articulos por año
        grouped_by_year = groupby(result, key=lambda x: x[0])
        # por cada grupo de articulos del mismo año
        for _, group in grouped_by_year:
            #se ordena el grupo por titulo y se añade al resultado final
            final_result.extend(sorted(list(group), key=lambda x: x[1].lower()))# metodo lower contiene todas las cadenas de texto en python convierte mayuscula a minus
    return final_result

# ==================== FUNCIONES DE ANÁLISIS ====================

def _parse_bib_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)
    return bib_database.entries

def _measure_time(sort_func, arr):
    arr_copy = arr[:]
    start_time = time.time()
    sort_func(arr_copy)
    end_time = time.time()
    return end_time - start_time

def _plot_algorithms_to_base64(results):
    results.sort(key=lambda x: x[3])
    names = [res[0] for res in results]
    times = [res[3] for res in results]
    plt.figure(figsize=(12, 8))
    bars = plt.bar(names, times, color='steelblue')
    plt.ylabel('Tiempo de Ejecución (segundos)')
    plt.xlabel('Algoritmo de Ordenamiento')
    plt.title('Comparación de Tiempos de Ejecución de Algoritmos de Ordenamiento')
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7, color='lightgray')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.6f}s', va='bottom', ha='center')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

def _plot_authors_to_base64(authors_data):
    authors_data.sort(key=lambda x: x[1])
    authors = [item[0] for item in authors_data]
    counts = [item[1] for item in authors_data]
    plt.figure(figsize=(12, 8))
    bars = plt.bar(authors, counts, color='steelblue')
    plt.ylabel('Número de Apariciones')
    plt.xlabel('Autor')
    plt.title('Top 15 Autores con más Apariciones')
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7, color='lightgray')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval}', va='bottom', ha='center')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

def analizar_ordenamiento_completo(bib_file_path: str, size: int, algorithm: str):
    try:
        articles = _parse_bib_file(bib_file_path)
    except Exception as e:
        return {"error": f"No se pudo parsear el archivo .bib: {e}"}

    articulos_para_ordenar = []
    for art in articles:
        year = art.get('year', '0').strip()
        title = art.get('title', 'Sin Título').strip()
        try:
            year_int = int(year)
        except (ValueError, TypeError):
            year_int = 0
        articulos_para_ordenar.append({'year': year_int, 'title': title})
    
    articulos_ordenados_display = sorted(articulos_para_ordenar, key=lambda x: (x['year'], x['title']))

    data_to_sort = [(d['year'], d['title']) for d in articulos_para_ordenar]

    if not data_to_sort:
        return {"error": "No se encontraron artículos válidos para analizar."}

    if len(data_to_sort) > size:
        sample_data = random.sample(data_to_sort, size)
    else:
        sample_data = data_to_sort
        size = len(sample_data)

    all_algorithms = [
        ("TimSort", tim_sort, "O(n log n)"),
        ("Comb Sort", comb_sort, "O(n²)"),
        ("Selection Sort", selection_sort, "O(n²)"),
        ("Tree Sort", tree_sort, "O(n log n) avg"),
        ("Pigeonhole Sort", pigeonhole_sort_adapted, "O(n + range)"),
        ("BucketSort", bucket_sort_adapted, "O(n + k) avg"),
        ("QuickSort", quicksort, "O(n log n) avg"),
        ("HeapSort", heapsort, "O(n log n)"),
        ("Bitonic Sort", bitonic_sort, "O(n log² n)"),
        ("Gnome Sort", gnome_sort, "O(n²)"),
        ("Binary Insertion Sort", binary_insertion_sort, "O(n²)"),
        ("RadixSort", radix_sort_adapted, "O(d(n + k))")
    ]

    algorithms_to_run = []
    if algorithm == "all":
        algorithms_to_run = all_algorithms
    else:
        for alg in all_algorithms:
            if alg[0] == algorithm:
                algorithms_to_run.append(alg)
                break
    
    results_table = []
    for name, func, complexity in algorithms_to_run:
        try:
            exec_time = _measure_time(func, sample_data)
            results_table.append((name, complexity, size, exec_time))
        except Exception as e:
            print(f"Error ejecutando {name}: {e}")
            results_table.append((name, complexity, size, -1))

    grafico_algoritmos_base64 = None
    if algorithm == "all" and results_table:
        valid_results = [r for r in results_table if r[3] != -1]
        if valid_results:
            grafico_algoritmos_base64 = _plot_algorithms_to_base64(valid_results)

    all_authors = []
    for art in articles:
        if 'author' in art:
            authors = art['author'].split(' and ')
            all_authors.extend([author.strip() for author in authors if author.strip()])
            
    author_counts = Counter(all_authors)
    top_15_authors = author_counts.most_common(15)
    
    autores_ordenados_para_grafico = sorted(top_15_authors, key=lambda x: (x[1], x[0]))

    grafico_autores_base64 = None
    if autores_ordenados_para_grafico:
        grafico_autores_base64 = _plot_authors_to_base64(autores_ordenados_para_grafico)

    def sort_key(entry):
        try:
            year = int(entry.get('year', 0))
        except (ValueError, TypeError):
            year = 0
        title = entry.get('title', 'Sin Título').lower()
        return (year, title)

    full_sorted_articles = sorted(articles, key=sort_key)
    
    output_dir = os.path.join(os.path.dirname(bib_file_path), '..', 'resultados_ordenados')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"ordenados_{timestamp}.bib"
    output_filepath = os.path.join(output_dir, output_filename)

    db = BibDatabase()
    db.entries = full_sorted_articles
    writer = BibTexWriter()
    with open(output_filepath, 'w', encoding='utf-8') as bibfile:
        bibfile.write(writer.write(db))

    debug_info = {
        "is_all_algorithms": algorithm == "all",
        "top_author_count": len(top_15_authors),
        "performance_graph_generated": grafico_algoritmos_base64 is not None,
        "author_graph_generated": grafico_autores_base64 is not None
    }

    return {
        "articulos_ordenados": articulos_ordenados_display[:100],
        "tabla_algoritmos": results_table,
        "grafico_algoritmos_base64": grafico_algoritmos_base64,
        "autores_ordenados": top_15_authors,
        "grafico_autores_base64": grafico_autores_base64,
        "output_file": output_filename,
        "debug_info": debug_info
    }