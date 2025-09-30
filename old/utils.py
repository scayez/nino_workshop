import requests
from pathlib import Path


from pathlib import Path
import requests

def download_file(url: str, save_name: str, data_dir: str = "data") -> Path:
    """
    Download a file from a given URL and save it in a specified folder.

    Parameters
    ----------
    url : str
        Direct URL of the file to download
    save_name : str
        Name to save the downloaded file as
    data_dir : str, default "data"
        Directory in which to save the file

    Returns
    -------
    Path
        Path to the downloaded file
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    file_path = data_path / save_name

    print(f"Downloading dataset from {url} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    
    # Ecriture en chunks pour les gros fichiers
    with open(file_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Downloaded to {file_path}")
    return file_path


# def download_from_drive(file_id: str, save_name: str, data_dir: str = "data") -> Path:
#     """
#     Download a zip file from a Google Drive file ID, save it

#     Parameters
#     ----------
#     file_id : str
#         Google Drive file ID
#     save_name : str
#         Name to save the downloaded file as (including .zip)
#     data_dir : str, default "data"
#         Directory in which to save the file

#     Returns
#     -------
#     Path
#         Path to the downloaded  file/directory
#     """
#     import requests
#     from pathlib import Path
#     import zipfile

#     # Créer le dossier data si nécessaire
#     data_path = Path(data_dir)
#     data_path.mkdir(exist_ok=True)

#     zip_path = data_path / save_name

#     # Construire automatiquement le lien direct Google Drive
#     download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

#     print(f"Downloading dataset from {download_url} ...")
#     r = requests.get(download_url)
#     r.raise_for_status()
#     zip_path.write_bytes(r.content)
#     print(f"Downloaded to {zip_path}")

