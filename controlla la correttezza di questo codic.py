import os
from datetime import datetime, timezone
from PIL import Image
from PIL.ExifTags import TAGS

def get_exif_date(file_path):
    """
    Estrae la data dalla foto dai metadati EXIF.
    
    Args:
        file_path (str): Percorso del file immagine
    
    Returns:
        datetime.datetime or None: Data estratta dal metadato
    """
    try:
        img = Image.open(file_path)
        exif_data = img._getexif()
        
        # Cerca la data in vari campi EXIF
        date_fields = [
            TAGS.GPS_DATE_TIME,  # GPS timestamp
            TAGS.DATE_TIME_ORIGINAL,  # Data originale della foto
            TAGS.DATE_TIME_DIGITIZED,  # Data di digitalizzazione
            TAGS.DATE_TIME  # Ultima modifica del metadato
        ]
        
        for field in date_fields:
            if field in exif_data:
                date_str = str(exif_data[field])
                
                # Prova diversi formati data comuni
                formats = [
                    "%Y:%m:%d %H:%M:%S",  # Formato EXIF standard (2023:01:15 14:30:00)
                    "%Y-%m-%d %H:%M:%S",  # ISO format
                    "%Y/%m/%d %H:%M:%S",  # Slash format
                    "%Y%m%d_%H%M%S"      # Formato compatto (20230115_143000)
                ]
                
                for fmt in formats:
                    try:
                        parsed_date = datetime.strptime(date_str, fmt)
                        
                        # Se la data è nel passato e ragionevole
                        if 1970 <= parsed_date.year <= 2030:
                            return parsed_date.replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
        
        print(f"⚠️ Nessuna data valida trovata in {file_path}")
        return None
    
    except Exception as e:
        print(f"❌ Errore nel leggere i metadati di {file_path}: {e}")
        return None

def update_file_date(file_path, new_datetime):
    """
    Aggiorna la data di modifica del file.
    
    Args:
        file_path (str): Percorso del file
        new_datetime (datetime.datetime): Nuova data da impostare
    
    Returns:
        bool: True se l'aggiornamento è riuscito, False altrimenti
    """
    try:
        # Converti la data in timestamp Unix
        timestamp = new_datetime.timestamp()
        
        # Aggiorna sia la data di modifica che quella di creazione (su sistemi Linux/Mac)
        os.utime(file_path, (timestamp, timestamp))
        
        print(f"✅ Data aggiornata per {file_path}: {new_datetime}")
        return True
    
    except Exception as e:
        print(f"❌ Errore nell'aggiornare la data di {file_path}: {e}")
        return False

def process_directory(directory_path, recursive=True):
    """
    Processa tutti i file immagine in una cartella.
    
    Args:
        directory_path (str): Percorso della cartella
        recursive (bool): Se includere sottocartelle
    
    Returns:
        tuple: (files_processati, files_aggiornati)
    """
    # Estensioni di file immagine supportate
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp'}
    
    files_processati = 0
    files_aggiornati = 0
    
    if recursive:
        for root, dirs, files in os.walk(directory_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                
                # Controlla l'estensione del file
                ext = os.path.splitext(file_name)[1].lower()
                if ext not in image_extensions:
                    continue
                
                files_processati += 1
                
                # Estrae la data dai metadati
                photo_date = get_exif_date(file_path)
                
                if photo_date is None:
                    print(f"⚠️ Saltato {file_path}: nessuna data valida")
                    continue # Aggiorna la data del file
                if update_file_date(file_path, photo_date):
                    files_aggiornati += 1
    
    return files_processati, files_aggiornati

def main():
    """
    Funzione principale dello script.
    
    Chiede all'utente il percorso della cartella e processa i file.
    """
    print("📸 Script per aggiornare date foto dai metadati EXIF")
    print("-" * 50)
    
    # Chiedi la cartella da processare
    directory = input("Inserisci il percorso della cartella con le foto: ").strip()
    
    if not os.path.isdir(directory):
        print(f"❌ Errore: '{directory}' non è una cartella valida")
        return
    
    # Chiedi se processare sottocartelle
    recursive_input = input("Processare anche sottocartelle? (s/N): ").strip().lower()
    recursive = recursive_input in ('s', 'si', 'yes')
    
    print(f"\n🔍 Processando la cartella: {directory}")
    if recursive:
        print("📁 Modalità ricorsiva attivata")
    
    # Esegui il processamento
    files_processati, files_aggiornati = process_directory(directory, recursive)
    
    print("\n" + "=" * 50)
    print(f"✅ Processo completato!")
    print(f"📊 File trovati: {files_processati}")
    print(f"🔄 Date aggiornate: {files_aggiornati}")
    print("=" * 50)

if __name__ == "__main__":
    main()