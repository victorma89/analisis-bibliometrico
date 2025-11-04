import time
import random
import os
import json
from datetime import datetime

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains

# --- Constantes específicas de IEEE ---
BASE_URL = "https://ieeexplore-ieee-org.crai.referencistas.com/Xplore/home.jsp"
SEARCH_TERM = "generative artificial intelligence"
PROGRESS_FILE = "ieee_scraping_progress.json"

# --- Funciones de ayuda ---
def type_like_human(element, text: str):
    """
    Simula la escritura humana en un elemento web.
    """
    print(f"[IEEE-LOG] Escribiendo texto: '{text[:20]}...'")
    for char in text:
        element.send_keys(char)
        time.sleep(random.uniform(0.08, 0.2))

def handle_cookies(driver):
    """
    Busca y hace clic en el botón para aceptar cookies de forma más robusta.
    """
    possible_texts = ['Aceptar todo', 'Accept all', 'Aceptar', 'Accept']
    for text in possible_texts:
        try:
            print(f"[IEEE-LOG] Buscando banner de cookies con texto: '{text}'...")
            cookie_button_xpath = f"//button[contains(., '{text}')]"
            cookie_button = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.XPATH, cookie_button_xpath))
            )
            print(f"[IEEE-LOG] Botón de cookies encontrado con texto '{text}'. Haciendo clic...")
            driver.execute_script("arguments[0].click();", cookie_button)
            print("[IEEE-LOG] Cookies aceptadas.")
            time.sleep(2)
            return
        except TimeoutException:
            print(f"[IEEE-LOG] No se encontró botón de cookies con texto: '{text}'.")
            continue

def wait_for_new_file(download_path, timeout=60):
    """
    Espera a que aparezca un nuevo archivo en el directorio de descargas.
    """
    print(f"[IEEE-LOG] Esperando nueva descarga en: {download_path}")
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
                print(f"[IEEE-LOG] Nuevo archivo detectado y estable: {new_files}")
                return True
        except Exception as e:
            print(f"[IEEE-LOG] Error verificando descargas: {e}")
    
    return False

def realizar_busqueda(driver, termino):
    """
    Realiza la búsqueda usando el selector y la lógica del usuario.
    """
    try:
        print("[IEEE-LOG] Esperando el campo de búsqueda...")
        search_input = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.global-search-bar input[type='search']"))
        )
        print("[IEEE-LOG] Campo de búsqueda encontrado. Limpiando y escribiendo...")
        search_input.clear()
        search_input.send_keys(termino)
        search_input.send_keys(Keys.RETURN)
        print(f"[IEEE-LOG] Búsqueda de '{termino}' enviada.")
    except Exception as e:
        print(f"[IEEE-LOG] Error al realizar la búsqueda: {e}")
        raise

def save_progress(page_num, download_path, status="completed"):
    """
    Guarda el progreso del scraping en un archivo JSON.
    """
    progress_file = os.path.join(download_path, PROGRESS_FILE)
    progress_data = {
        "last_page_processed": page_num,
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "search_term": SEARCH_TERM
    }
    
    try:
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                existing_data = json.load(f)
                if "pages_completed" not in existing_data:
                    existing_data["pages_completed"] = []
        else:
            existing_data = {"pages_completed": []}
        
        existing_data.update(progress_data)
        if status == "completed" and page_num not in existing_data["pages_completed"]:
            existing_data["pages_completed"].append(page_num)
        
        with open(progress_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
    except Exception as e:
        print(f"[IEEE-LOG] Error guardando progreso: {e}")

def load_progress(download_path):
    """
    Carga el progreso previo si existe.
    """
    progress_file = os.path.join(download_path, PROGRESS_FILE)
    try:
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                data = json.load(f)
                return data.get("last_page_processed", 0), data.get("pages_completed", [])
    except Exception as e:
        print(f"[IEEE-LOG] Error cargando progreso: {e}")
    return 0, []
def close_modal_if_exists(driver):
    """
    Intenta cerrar el modal de exportación si está abierto.
    """
    try:
        cancel_button = driver.find_element(By.CLASS_NAME, "stats-download-citations-button-cancel")
        cancel_button.click()
        print("[IEEE-LOG] Modal cerrado.")
        time.sleep(1)
    except NoSuchElementException:
        pass

def perform_login(driver, email: str, password: str):
    """
    Realiza el proceso de login en la plataforma IEEE a través del proxy.
    """
    print("[IEEE-LOG] Iniciando login...")
    print(f"[IEEE-LOG] Navegando a IEEE Xplore: {BASE_URL}")
    driver.get(BASE_URL)

    try:
        print("[IEEE-LOG] Esperando la redirección al portal de login de la universidad...")
        print("[IEEE-LOG] Buscando botón de Google...")
        google_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.ID, "btn-google"))
        )
        print("[IEEE-LOG] Botón de Google encontrado. Haciendo clic...")
        google_button.click()

        print("[IEEE-LOG] Buscando el campo de correo electrónico...")
        username_field = WebDriverWait(driver, 15).until(
            EC.visibility_of_element_located((By.XPATH, "//input[@type='email']"))
        )
        username_field.send_keys(email)
        username_field.send_keys(Keys.RETURN)
        print("[IEEE-LOG] Email enviado.")

        print("[IEEE-LOG] Buscando el campo de contraseña...")
        password_field = WebDriverWait(driver, 15).until(
            EC.visibility_of_element_located((By.XPATH, "//input[@type='password']"))
        )
        password_field.send_keys(password)
        password_field.send_keys(Keys.RETURN)
        print("[IEEE-LOG] Contraseña enviada.")

        print("[IEEE-LOG] Login enviado. Esperando redirección a IEEE (max 40s)...")
        print("[IEEE-LOG] Redirección a IEEE completada.")
        print("[IEEE-LOG] Proceso de login finalizado. Aceptando cookies de la página de IEEE...")
        WebDriverWait(driver, 40).until(EC.url_contains("ieee"))
        print("[IEEE-LOG] Redirección a IEEE completada.")

    except TimeoutException:
        print("[IEEE-LOG] No se completó el flujo de login de Google. Verificando si ya estamos en IEEE...")
        try:
            WebDriverWait(driver, 10).until(EC.url_contains("ieee"))
            print("[IEEE-LOG] Verificación exitosa. Ya estábamos en IEEE.")
        except TimeoutException:
            final_url = driver.current_url
            print(f"[IEEE-LOG] ERROR: La URL final ({final_url}) no parece ser de IEEE.")
            raise
    
    finally:
        print("[IEEE-LOG] Proceso de login finalizado. Aceptando cookies de la página de IEEE...")
        handle_cookies(driver)

def download_page_citations(driver, page_num, download_path):
    """
    Descarga las citas de una página específica.
    """
    print("[IEEE-LOG] Esperando que los resultados de la página se carguen...")
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "xpl-results-item"))
    )
    print("[IEEE-LOG] Resultados cargados.")
    time.sleep(3)

    print("[IEEE-LOG] Buscando checkbox 'Select All on Page'...")
    select_all_element = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'Select All on Page')]"))
    )
    driver.execute_script("arguments[0].click();", select_all_element)
    print("[IEEE-LOG] Elemento 'Select All on Page' encontrado y clickeado.")
    time.sleep(3)

    print("[IEEE-LOG] Buscando y haciendo clic en el botón de Exportar...")
    export_button = WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(normalize-space(.), 'Export')]"))
    )
    export_button.click()
    time.sleep(2)

    print("[IEEE-LOG] Esperando modal de exportación...")
    modal_selector = "//div[contains(@class, 'modal-dialog')]"
    WebDriverWait(driver, 15).until(EC.visibility_of_element_located((By.XPATH, modal_selector)))
    print("[IEEE-LOG] Modal de exportación encontrado.")
    time.sleep(1)

    print("[IEEE-LOG] Haciendo clic en la pestaña 'Citations'...")
    citations_tab = WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable((By.XPATH, "//a[contains(normalize-space(.), 'Citations')]"))
    )
    driver.execute_script("arguments[0].click();", citations_tab)
    time.sleep(1)

    print("[IEEE-LOG] Seleccionando formato 'BibTeX' por posición...")
    radio_buttons_format = driver.find_elements(By.NAME, "download-format")
    if len(radio_buttons_format) > 1:
        radio_buttons_format[1].click()
        print("[IEEE-LOG] Opción 'BibTeX' seleccionada.")
    time.sleep(1)

    print("[IEEE-LOG] Seleccionando 'Citation and Abstract' por posición...")
    radio_buttons_content = driver.find_elements(By.NAME, "citations-format")
    if len(radio_buttons_content) > 1:
        radio_buttons_content[1].click()
        print("[IEEE-LOG] Opción 'Citation and Abstract' seleccionada.")
    time.sleep(1)

    print("[IEEE-LOG] Buscando y haciendo clic en el botón final de Descarga...")
    download_button = WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable((By.CLASS_NAME, "stats-SearchResults_Citation_Download"))
    )
    driver.execute_script("arguments[0].scrollIntoView();", download_button)
    download_button.click()
    print("[IEEE-LOG] Botón 'Download' clickeado.")

    download_success = wait_for_new_file(download_path, timeout=60)
    
    print("[IEEE-LOG] Cerrando el modal de exportación...")
    cancel_button = WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable((By.CLASS_NAME, "stats-download-citations-button-cancel"))
    )
    cancel_button.click()
    print("[IEEE-LOG] Modal cerrado.")
    time.sleep(2)
    
    return download_success

def search_and_download(driver, max_pages: int, download_path: str, continue_last: bool = True, max_retries: int = 3):
    """
    Realiza la búsqueda y descarga las citas de hasta `max_pages` páginas con reintentos y guardado de progreso.
    """
    # Cargar progreso previo
    last_processed, completed_pages = load_progress(download_path)
    
    if last_processed > 0 and continue_last:
        print(f"[IEEE-LOG] Se detectó progreso previo. Última página procesada: {last_processed}")
        print(f"[IEEE-LOG] Páginas completadas previamente: {completed_pages}")
        print("[IEEE-LOG] Continuando desde la última página procesada...")
    else:
        print("[IEEE-LOG] Comenzando desde la primera página...")
        last_processed = 0
        completed_pages = []
    
    realizar_busqueda(driver, SEARCH_TERM)
    
    print("[IEEE-LOG] Búsqueda enviada. Esperando que la página de resultados cargue...")
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "xpl-results-list"))
    )
    print("[IEEE-LOG] Página de resultados cargada.")
    
    # Si estamos reanudando, navegar a la página correcta
    if last_processed > 0:
        print(f"[IEEE-LOG] Navegando a la página {last_processed + 1}...")
        for i in range(last_processed):
            try:
                next_button = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "stats-Pagination_arrow_next_2"))
                )
                driver.execute_script("arguments[0].scrollIntoView();", next_button)
                next_button.click()
                time.sleep(3)
            except Exception as e:
                print(f"[IEEE-LOG] Error navegando a la página de reanudación: {e}")
                break

    consecutive_failures = 0
    
    for page_num in range(last_processed + 1, max_pages + 1):
        # Saltar páginas ya completadas
        if page_num in completed_pages:
            print(f"\n--- [IEEE-LOG] Página {page_num} ya fue procesada anteriormente. Saltando... ---")
            continue
            
        print(f"\n--- [IEEE-LOG] Procesando página {page_num} de {max_pages} ---")
        
        retry_count = 0
        page_success = False
        
        while retry_count < max_retries and not page_success:
            try:
                # Intentar descargar las citas de la página actual
                download_success = download_page_citations(driver, page_num, download_path)
                
                if download_success:
                    print(f"[IEEE-LOG] Descarga de la página {page_num} completada con éxito.")
                    save_progress(page_num, download_path, status="completed")
                    page_success = True
                    consecutive_failures = 0
                else:
                    print(f"[WARN] No se detectó descarga para la página {page_num}.")
                    raise Exception("Descarga no detectada")
                
            except Exception as e:
                retry_count += 1
                print(f"!! [IEEE-LOG] ERROR en la página {page_num} (intento {retry_count}/{max_retries}): {e}")
                
                # Intentar cerrar el modal si quedó abierto
                close_modal_if_exists(driver)
                
                if retry_count < max_retries:
                    print(f"[IEEE-LOG] Reintentando página {page_num} en 5 segundos...")
                    time.sleep(5)
                else:
                    consecutive_failures += 1
                    print(f"[IEEE-LOG] Página {page_num} falló después de {max_retries} intentos.")
                    save_progress(page_num - 1, download_path, status="failed")
                    
                    if consecutive_failures >= 3:
                        print("[IEEE-LOG] Demasiados errores consecutivos. Deteniendo el proceso.")
                        print(f"[IEEE-LOG] Último progreso guardado: página {page_num - 1}")
                        return

        # --- Paginación ---
        if page_num < max_pages:
            try:
                # Tiempo de espera aleatorio entre páginas
                wait_time = random.uniform(3, 7)
                print(f"[IEEE-LOG] Esperando {wait_time:.1f} segundos antes de continuar...")
                time.sleep(wait_time)
                
                print("[IEEE-LOG] Buscando botón 'Next' para paginación...")
                
                # Intentar diferentes selectores para el botón Next
                next_button = None
                selectors_to_try = [
                    (By.CLASS_NAME, "stats-Pagination_arrow_next_2"),
                    (By.XPATH, "//button[contains(@class, 'stats-Pagination_arrow_next')]"),
                    (By.XPATH, "//div[contains(@class, 'pagination-next')]//button"),
                    (By.XPATH, "//button[contains(text(), 'Next')]")
                ]
                
                for selector_type, selector in selectors_to_try:
                    try:
                        next_button = WebDriverWait(driver, 5).until(
                            EC.presence_of_element_located((selector_type, selector))
                        )
                        if next_button and next_button.is_displayed():
                            break
                    except:
                        continue
                
                if not next_button:
                    raise Exception("No se pudo encontrar el botón 'Next'")
                
                # Verificar si el botón está deshabilitado
                if "disabled" in next_button.get_attribute("class"):
                    print("[IEEE-LOG] No hay más páginas disponibles. Finalizando proceso.")
                    break
                
                # Asegurarse de que la página esté scrolleada hasta el botón
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                # Hacer scroll específicamente al botón
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", next_button)
                time.sleep(2)
                
                # Intentar diferentes métodos para hacer clic
                click_success = False
                try:
                    next_button.click()
                    click_success = True
                except:
                    try:
                        driver.execute_script("arguments[0].click();", next_button)
                        click_success = True
                    except:
                        actions = ActionChains(driver)
                        actions.move_to_element(next_button).click().perform()
                        click_success = True
                
                if not click_success:
                    raise Exception("No se pudo hacer clic en el botón 'Next'")
                
                print(f"[IEEE-LOG] Navegando a la página {page_num + 1}...")
                
                # Esperar a que la página actual desaparezca y la nueva página cargue
                time.sleep(5)
                
                # Esperar a que los nuevos resultados aparezcan
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "xpl-results-item"))
                )
                
                # Verificar que estamos en la página correcta
                try:
                    current_page_element = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".pagination-current-page"))
                    )
                    if str(page_num + 1) not in current_page_element.text:
                        raise Exception(f"La navegación no llevó a la página {page_num + 1}")
                except:
                    print("[IEEE-LOG] No se pudo verificar el número de página actual.")
                    
            except Exception as e:
                print(f"[IEEE-LOG] Error durante la paginación: {e}")
                print("[IEEE-LOG] Intentando recargar la página y continuar...")
                try:
                    driver.refresh()
                    time.sleep(5)
                    continue
                except:
                    print("[IEEE-LOG] No se pudo navegar a la siguiente página. Finalizando proceso.")
                    save_progress(page_num, download_path, status="pagination_error")
                    break
        else:
            print(f"\n[IEEE-LOG] Se alcanzó el límite de {max_pages} páginas.")
    
    # Resumen final
    print("\n" + "="*60)
    print("[IEEE-LOG] RESUMEN DEL PROCESO DE DESCARGA")
    print("="*60)
    
    # Cargar el progreso final para mostrar estadísticas
    _, completed_pages_final = load_progress(download_path)
    print(f"[IEEE-LOG] Páginas completadas exitosamente: {len(completed_pages_final)}")
    print(f"[IEEE-LOG] Páginas procesadas: {completed_pages_final}")
    print(f"[IEEE-LOG] Archivos descargados en: {download_path}")
    
    # Contar archivos .bib en el directorio
    try:
        bib_files = [f for f in os.listdir(download_path) if f.endswith('.bib')]
        print(f"[IEEE-LOG] Total de archivos .bib encontrados: {len(bib_files)}")
    except:
        pass
    
    print("="*60)
    print("[IEEE-LOG] Proceso de descarga finalizado.")