import time
import random
import os
import json
from datetime import datetime

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException

# --- Constantes específicas de ScienceDirect ---
BASE_URL = "https://www-sciencedirect-com.crai.referencistas.com/"
SEARCH_TERM = "generative artificial intelligence"

# --- Funciones de ayuda ---

def type_like_human(element, text: str):
    """
    Simula la escritura humana en un elemento web.
    """
    element.clear()
    time.sleep(0.5)
    
    for char in text:
        element.send_keys(char)
        time.sleep(random.uniform(0.1, 0.2))

def wait_for_new_file(download_path, timeout=20):
    """
    Espera a que se complete una nueva descarga.
    """
    print(f"[SD-LOG] Esperando nueva descarga en: {download_path}")
    start_time = time.time()
    initial_files = set(os.listdir(download_path)) if os.path.exists(download_path) else set()
    
    while time.time() - start_time < timeout:
        time.sleep(1)
        try:
            current_files = set(os.listdir(download_path))
            new_files = current_files - initial_files
            if new_files:
                newest_file = max([os.path.join(download_path, f) for f in new_files], key=os.path.getctime)
                # Esperar a que el archivo termine de escribirse
                last_size = -1
                while last_size != os.path.getsize(newest_file):
                    last_size = os.path.getsize(newest_file)
                    time.sleep(0.5)
                print(f"[SD-LOG] Nuevo archivo detectado: {new_files}")
                return True
        except Exception as e:
            print(f"[SD-LOG] Error verificando descargas: {e}")
    
    return False

# --- Lógica principal del Scraper de ScienceDirect ---

def perform_login(driver, email: str, password: str):
    """
    Realiza el proceso de login en la plataforma ScienceDirect a través del proxy.
    """
    print("[SD-LOG] Iniciando login...")
    print(f"[SD-LOG] Navegando a ScienceDirect: {BASE_URL}")
    driver.get(BASE_URL)

    print("[SD-LOG] Esperando la redirección al portal de login de la universidad...")
    try:
        print("[SD-LOG] Buscando botón de Google...")
        google_button = WebDriverWait(driver, 25).until(
            EC.element_to_be_clickable((By.ID, "btn-google"))
        )
        google_button.click()
        print("[SD-LOG] Clic en el botón de Google.")

        print("[SD-LOG] Ingresando correo electrónico...")
        username_field = WebDriverWait(driver, 15).until(
            EC.visibility_of_element_located((By.ID, "identifierId"))
        )
        type_like_human(username_field, email)
        username_field.send_keys(Keys.RETURN)
        
        time.sleep(random.uniform(2, 4))

        print("[SD-LOG] Ingresando contraseña...")
        password_field = WebDriverWait(driver, 15).until(
            EC.visibility_of_element_located((By.NAME, "Passwd"))
        )
        type_like_human(password_field, password)
        password_field.send_keys(Keys.RETURN)

        print("[SD-LOG] Login exitoso. Esperando a la redirección a ScienceDirect...")
        
        # Solo verificar con el selector que funciona según el log
        try:
            search_field = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.NAME, "qs"))
            )
            print("[SD-LOG] Redirección a ScienceDirect completada.")
        except:
            print("[SD-LOG] No se encontró el campo de búsqueda, pero continuando...")

    except TimeoutException:
        print("[SD-LOG] No se completó el flujo de login o ya estabas logueado.")
        # Verificación rápida
        try:
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.NAME, "qs"))
            )
            print("[SD-LOG] Ya estamos en ScienceDirect.")
        except:
            print("[SD-LOG] Error en verificación, continuando...")
    except Exception as e:
        print(f"[SD-LOG] Error durante el login: {e}")
        raise

def search_and_download(driver, max_pages: int, download_path: str, continue_last: bool):
    """
    Realiza la búsqueda y descarga las citas página por página en ScienceDirect.
    """
    # Cargar el estado de la paginación
    progress_file = os.path.join(download_path, 'sd_scraping_progress.json')
    start_page = 1
    if continue_last and os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            data = json.load(f)
            start_page = data.get('last_page', 1) + 1
        print(f"[SD-LOG] Reanudando desde la página {start_page}.")

    if start_page > max_pages:
        print("[SD-LOG] Todas las páginas ya han sido procesadas.")
        return

    try:
        print(f'[SD-LOG] Realizando búsqueda del término: "{SEARCH_TERM}"')
        
        # Usar directamente el selector que funciona según el log
        search_box = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.NAME, "qs"))
        )
        print("[SD-LOG] Campo de búsqueda encontrado")
        
        # Hacer scroll y clic en el campo
        driver.execute_script("arguments[0].scrollIntoView(true);", search_box)
        time.sleep(1)
        
        try:
            search_box.click()
        except:
            driver.execute_script("arguments[0].click();", search_box)
        
        print("[SD-LOG] Escribiendo término de búsqueda...")
        type_like_human(search_box, SEARCH_TERM)
        
        # Ejecutar búsqueda
        search_box.send_keys(Keys.ENTER)
        print("[SD-LOG] Búsqueda ejecutada")

        print("[SD-LOG] Esperando que la página de resultados cargue...")
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.ID, "srp-results-list"))
        )
        print("[SD-LOG] Búsqueda completada.")

    except Exception as e:
        print(f"[SD-LOG] Error al realizar la búsqueda: {e}")
        raise

    for page_num in range(start_page, max_pages + 1):
        print(f"--- [SD-LOG] Procesando página {page_num} de {max_pages} ---")
        try:
            # Esperar que los resultados se carguen
            print("[SD-LOG] Esperando que los resultados se carguen...")
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "li.ResultItem"))
            )
            
            # Selección manual de todos los resultados (ya que el checkbox select-all no existe)
            print("[SD-LOG] Seleccionando todos los resultados...")
            try:
                result_items = driver.find_elements(By.CSS_SELECTOR, "li.ResultItem")
                print(f"[SD-LOG] Encontrados {len(result_items)} resultados")
                
                selected_count = 0
                for i, item in enumerate(result_items[:25]):  # Máximo 25 items por página
                    try:
                        item_checkbox = item.find_element(By.CSS_SELECTOR, "input[type='checkbox']")
                        
                        if not item_checkbox.is_selected():
                            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", item_checkbox)
                            time.sleep(0.1)
                            
                            try:
                                item_checkbox.click()
                            except:
                                driver.execute_script("arguments[0].click();", item_checkbox)
                            
                            selected_count += 1
                            print(f"[SD-LOG] ✓ Seleccionado resultado {i+1}")
                            time.sleep(0.2)
                        else:
                            selected_count += 1
                            print(f"[SD-LOG] ✓ Resultado {i+1} ya estaba seleccionado")
                    except Exception as item_error:
                        print(f"[SD-LOG] Error procesando resultado {i+1}: {item_error}")
                        continue
                
                print(f"[SD-LOG] ✓ Total seleccionados: {selected_count} elementos")
                
                if selected_count == 0:
                    raise Exception("No se pudo seleccionar ningún elemento")
                    
            except Exception as e:
                print(f"[SD-LOG] Error en selección: {e}")
                raise

            time.sleep(random.uniform(2, 4))

            # Buscar y hacer clic en el botón de exportar (usar el selector que funciona)
            print("[SD-LOG] Buscando botón de exportar...")
            export_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Export')]"))
            )
            export_button.click()
            print("[SD-LOG] Botón de exportar presionado.")

            # Esperar modal y seleccionar BibTeX
            print("[SD-LOG] Esperando modal de exportación...")
            WebDriverWait(driver, 15).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".ReactModal__Body--open"))
            )
            print("[SD-LOG] Modal de exportación visible.")
            
            print("[SD-LOG] Seleccionando opción BibTeX...")
            bibtex_export_button = WebDriverWait(driver, 15).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Export citation to BibTeX')]"))
            )
            bibtex_export_button.click()
            print("[SD-LOG] Descarga iniciada.")

            # Esperar la descarga
            if not wait_for_new_file(download_path):
                print("[SD-LOG] No se detectó la descarga del archivo.")
                raise Exception("Fallo en la descarga del archivo.")
            
            print(f"[SD-LOG] Descarga de la página {page_num} completada.")
            
            # Guardar progreso
            with open(progress_file, 'w') as f:
                json.dump({'last_page': page_num, 'timestamp': str(datetime.now())}, f)

            # Navegar a la siguiente página si no es la última
            if page_num < max_pages:
                print(f"[SD-LOG] Navegando a la página {page_num + 1}...")
                try:
                    # Usar el selector específico que funciona según el log
                    next_page_button = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "li.pagination-link.next-link"))
                    )
                    
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_page_button)
                    time.sleep(1)
                    
                    current_url = driver.current_url
                    
                    try:
                        next_page_button.click()
                    except:
                        driver.execute_script("arguments[0].click();", next_page_button)
                    
                    print("[SD-LOG] Clic en botón de siguiente página")
                    
                    # Esperar que la página cambie
                    WebDriverWait(driver, 10).until(
                        lambda x: x.current_url != current_url
                    )
                    
                    # Esperar que los nuevos resultados aparezcan
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "li.ResultItem"))
                    )
                    
                    print(f"[SD-LOG] Navegación a página {page_num + 1} completada")
                    time.sleep(random.uniform(4, 6))
                    
                except Exception as nav_error:
                    print(f"[SD-LOG] Error en navegación: {nav_error}")
                    print("[SD-LOG] No se pudo navegar a la siguiente página.")
                    break

        except Exception as e:
            print(f"[SD-LOG] Error procesando página {page_num}: {e}")
            # Intentar refrescar y continuar
            driver.refresh()
            time.sleep(5)
            continue