def download_url(url:str, file_path:str, desc:str = "Downloading"):
    """
    Utility function for downloading a file from a given URL; the file is saved at the provided path

    Parameters:
        url: str
            The URL to download the file from
        file_path : str
            The path to save the downloaded file to
        desc : str
            Description string to print for the tqdm progress bar
    """
    import functools
    import pathlib
    import shutil
    import requests
    from tqdm.auto import tqdm
    
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))

    file_path = pathlib.Path(file_path).expanduser().resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    desc += " (Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with file_path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)

    return file_path

def zip_extractall(zipfile_path:str, destination_folder:str, desc="Extracting", delete_zipfile:bool=False):
    """
    Utility function for extracting a ZIP archive, provided by its path into a folder

    Parameters:
        zipfile_path: str
            The path to the ZIP file
        destination_folder : str
            The path to the destination folder where the archive will be extracted
        desc : str
            Description string to print for the tqdm progress bar
        delete_zipfile: bool
            When True, deletes the zip file after extracting it
    """
    import pathlib
    import shutil
    import zipfile
    from tqdm.auto import tqdm  # could use from tqdm.gui import tqdm
    from tqdm.utils import CallbackIOWrapper

    zipfile_path = pathlib.Path(zipfile_path).expanduser().resolve()
    destination_folder = destination_folder.expanduser().resolve()
    with zipfile.ZipFile(zipfile_path) as zipf, tqdm(
        desc=desc, unit="B", unit_scale=True, unit_divisor=1024,
        total=sum(getattr(i, "file_size", 0) for i in zipf.infolist()),
    ) as pbar:
        for i in zipf.infolist():
            if not getattr(i, "file_size", 0):  # directory
                zipf.extract(i, destination_folder)
            else:
                extracted_path = destination_folder / i.filename
                if not extracted_path.parent.exists():
                    extracted_path.parent.mkdir(parents=True, exist_ok=True)
                with zipf.open(i) as fi, open(destination_folder / i.filename, "wb") as fo:
                    shutil.copyfileobj(CallbackIOWrapper(pbar.update, fi), fo)
    if delete_zipfile:
        zipfile_path.unlink()