from pathlib import Path
import yaml,h5py
from numpy.typing import NDArray


def load_yaml(path : Path, yaml_fname : str) -> dict :
    """load yaml file (from Louis Soulard-Fischer's work)

    Args:
        path (Path): yaml file path
        yaml_fname (str): yaml file name

    Returns:
        dict: dict of yaml attributes
    """
    with open(path/yaml_fname, 'r') as file :
        dict_yaml = yaml.safe_load(file)
    return dict_yaml

def load_data_hdf5(infile: str) -> NDArray:
    """load hdf5 files

    Args:
        infile (str): hdf5 file name

    Returns:
        NDArray: data from hdf5 file
    """
    with h5py.File(infile,"r") as f:
        return f[()]


def grid_to_lat_lon_zone(coord: list[float],zone: str) -> tuple:
    """transform coordinates in the model grid into coordinates longitude/latitude

    Args:
        coord (list[float]): list of coordinates
        zone (str): subdomains (NW,NE,SW,SE or C)

    Returns:
        tuple: longitude, latitude
    """
    run_config=load_yaml(path=Path(''), yaml_fname='config.yml')
    [lat_min,lat_max,lon_min,lon_max]=run_config['AROME']['Limit_EURW1S40']
    [X_min,Y_min]=run_config['Zoom'][zone]
    maille=run_config['AROME']['Resolution_EURW1S40']
    lat=lat_min+(coord[1]+Y_min)*maille
    lon=lon_min+(coord[0]+X_min)*maille
    return lon,lat
