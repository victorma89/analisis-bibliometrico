import os
import time
import getpass
import argparse
import sys
from pathlib import Path
import functools

# Forzar que la función print haga flush automáticamente
print = functools.partial(print, flush=True)

# Agregar el directorio actual al PYTHONPATH
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Importar los scrapers específicos de cada base de datos
from scrapers import sage_scraper
from scrapers import ieee_scraper
from scrapers import sciencedirect_scraper

# --- Configuración Global ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def setup_driver(download_path: str) -> webdriver.Chrome:
    """
    Configura y devuelve una instancia del WebDriver de Chrome con opciones personalizadas.
    """
    print(f"Asegurando que el directorio de descarga exista: {download_path}")
    os.makedirs(download_path, exist_ok=True)

    print("Configurando opciones de Chrome para descarga automática y anti-detección...")
    chrome_options = Options()

    # Opciones anti-detección y de sesión
    chrome_options.add_argument("--guest")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    # Opciones de descarga
    prefs = {
        "download.default_directory": download_path,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": False,
        "plugins.always_open_pdf_externally": True,
        "profile.default_content_setting_values.automatic_downloads": 1
    }
    chrome_options.add_experimental_option("prefs", prefs)

    print("Instalando y configurando el servicio de ChromeDriver...")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # Forzar comportamiento de descarga vía Chrome DevTools Protocol
    print("[CDP] Forzando la descarga automática de archivos...")
    driver.execute_cdp_cmd("Page.setDownloadBehavior", {
        "behavior": "allow",
        "downloadPath": download_path
    })

    print("WebDriver iniciado correctamente.")
    return driver

def get_password():
    """
    Obtiene la contraseña del usuario de manera segura y sin '\n' sobrantes.
    """
    try:
        if not sys.stdin.isatty():
            # Si viene por pipeline o archivo
            return sys.stdin.readline().strip()

        # Si el usuario la ingresa interactivo
        return getpass.getpass("Ingrese su contraseña: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nOperación cancelada por el usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError al obtener la contraseña: {e}")
        sys.exit(1)

def main():
    """
    Función principal para seleccionar y ejecutar el scraper apropiado.
    """
    parser = argparse.ArgumentParser(description="Automatiza la descarga de citas bibliográficas.")
    parser.add_argument("--database", required=True, choices=['sage', 'ieee', 'sciencedirect'], 
                       help="La base de datos de la cual descargar.")
    parser.add_argument("--email", required=True, 
                       help="Correo electrónico institucional para el login.")
    parser.add_argument("--pages", type=int, default=20, 
                       help="Número de páginas de resultados a procesar.")
    parser.add_argument("--continue", dest='continue_last', action='store_true',
                       help="Continuar desde la última página procesada")
    parser.add_argument("--restart", dest='continue_last', action='store_false',
                       help="Comenzar desde la primera página")
    parser.set_defaults(continue_last=True)
    
    args = parser.parse_args()
    
    # Limitar las páginas a 20 como máximo para seguridad
    max_pages = min(args.pages, 20)
    print(f"Se procesarán un máximo de {max_pages} páginas.")

    # Crear ruta de descarga específica para la base de datos
    base_download_path = os.path.join(ROOT_DIR, 'datos', 'descargas')
    specific_download_path = os.path.join(base_download_path, args.database)
    print(f"La ruta de descarga para esta sesión es: {specific_download_path}")

    try:
        password = get_password()

        # Debug opcional para validar que quedó limpia (puedes comentarlo luego)
        print(f"[DEBUG] Longitud de contraseña: {len(password)}, valor: {repr(password)}")

        driver = None
        try:
            driver = setup_driver(specific_download_path)

            if args.database == 'sage':
                print("\n--- Iniciando Scraper para SAGE ---")
                sage_scraper.perform_login(driver, args.email, password)
                sage_scraper.search_and_download(driver, max_pages, specific_download_path, args.continue_last)
            
            elif args.database == 'ieee':
                print("\n--- Iniciando Scraper para IEEE Xplore ---")
                ieee_scraper.perform_login(driver, args.email, password)
                ieee_scraper.search_and_download(driver, max_pages, specific_download_path, args.continue_last)
            
            elif args.database == 'sciencedirect':
                print("\n--- Iniciando Scraper para ScienceDirect ---")
                sciencedirect_scraper.perform_login(driver, args.email, password)
                sciencedirect_scraper.search_and_download(driver, max_pages, specific_download_path, args.continue_last)

            print("\n--- Proceso completado ---")
            print(f"Los archivos han sido descargados en: {specific_download_path}")

        except Exception as e:
            print(f"\nOcurrió un error inesperado en el proceso: {e}")
            import traceback
            print(traceback.format_exc())
        finally:
            if driver:
                print("Cerrando el navegador en 15 segundos...")
                time.sleep(15)
                driver.quit()
                
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
        sys.exit(1)

if __name__ == "__main__":
    main()