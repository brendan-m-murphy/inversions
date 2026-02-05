"""Helper functions for downloading and managing geographic data."""

import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


def download_natural_earth_data(data_dir: Path | None = None) -> Path:
    """Download Natural Earth 1:50m cultural data if not already present.
    
    Downloads the Natural Earth 1:50m cultural vectors dataset and saves it
    to the package data directory.
    
    Parameters
    ----------
    data_dir : Path, optional
        Directory to save the data file. If None, uses the package data directory.
    
    Returns
    -------
    Path
        Path to the downloaded natural_earth_50.zip file.
    
    Raises
    ------
    urllib.error.URLError
        If the download fails.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent
    
    data_dir.mkdir(parents=True, exist_ok=True)
    data_file = data_dir / "natural_earth_50.zip"
    
    if data_file.exists():
        return data_file
    
    # Natural Earth 1:50m Cultural Vectors download URL
    url = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
    
    logger.info(f"Downloading Natural Earth data to {data_file}...")
    urllib.request.urlretrieve(url, data_file)
    logger.info("Download complete.")
    
    return data_file


def get_world_map_path() -> Path:
    """Get the path to the world map data file, downloading if necessary.
    
    Returns
    -------
    Path
        Path to the natural_earth_50.zip file.
    """
    data_dir = Path(__file__).parent
    return download_natural_earth_data(data_dir)
