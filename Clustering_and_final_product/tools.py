from pathlib import Path
import yaml,h5py,pickle,datetime
import numpy as np
from numpy.typing import NDArray
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt



def extract_lonlat():
     """longitude and latitude of AROME-EPS grid (extraction)"""
     path_config=load_yaml(path=Path(''), yaml_fname='path.yml')
     with open(path_config['Path_lonlat'],'rb') as f:
                lonlat=pickle.load(f,encoding="latin1")
     return lonlat



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


def find_leadt(now: datetime.datetime, ref: datetime.datetime) -> int:
    """ compute difference (in hours) between two datetime objects."""
    leadt_seconds=(now-ref).total_seconds()
    leadt_hours=leadt_seconds/3600
    return int(leadt_hours)

def background_map_scenarios(zone: str, lonlat: np.ndarray, ax_predefined: plt.axes =None) -> plt.axes:
    """create a background map in cartopy with french departments for future plots

    Args:
        zone (str): subdomain (NW,NE,SW,SE or C)
        lonlat (np.ndarray): longitude and latitude of AROME grid (cf extract_lonlat function)
        ax_predefined (plt.axes, optional): predefined plt.axes. Defaults to None.

    Returns:
        plt.axes: new or current plt.axes with stereographic projection with zoom over the subdomain and french department.
    """
    run_config=load_yaml(path=Path(''), yaml_fname='config.yml')
    path_config=load_yaml(path=Path(''), yaml_fname='path.yml')
    proj=ccrs.Stereographic(central_latitude=46.7,central_longitude=2)
    proj._threshold /= 1000.
    [X_min,Y_min]=run_config['Zoom'][zone]
    nb_lat,nb_lon=run_config['Zoom']['Size_Y'],run_config['Zoom']['Size_X']
    if ax_predefined is None:
        ax = plt.axes(projection=proj)
    else:
        ax=ax_predefined

    Lat_min,Lon_min= grid_to_lat_lon_zone(-10,-10,zone=zone)
    Lat_max,Lon_max= grid_to_lat_lon_zone(nb_lon+50,nb_lat+10,zone=zone)
    lon_bor=[Lon_min,Lon_max]
    lat_bor=[Lat_min,Lat_max]

    #border projection in stereographic grid
    lon_lat_1=proj.transform_point(lon_bor[0],lat_bor[0],ccrs.PlateCarree())
    lon_lat_2=proj.transform_point(lon_bor[1],lat_bor[1],ccrs.PlateCarree())
    lon_bor=[lon_lat_1[0],lon_lat_2[0]]
    lat_bor=[lon_lat_1[1],lon_lat_2[1]]
    borders = lon_bor + lat_bor

    name_region=path_config['Plot']['department']
    region_shapes= list(shpreader.Reader(name_region).geometries())
    ax.set_extent(borders,proj) #map borders
    ax.add_geometries(region_shapes, ccrs.PlateCarree(),facecolor='none',edgecolor='lightgray',linewidth=0.5) #Add department 
    ax.add_feature(cfeature.COASTLINE.with_scale('10m')) #Add coast
    ax.add_feature(cfeature.BORDERS.with_scale('10m')) #Add country

    #domain contour in gray
    Lat_domaine_inf, Lon_domaine_inf= grid_to_lat_lon_zone(0,0,zone=zone)
    Lat_domaine_sup, Lon_domaine_sup= grid_to_lat_lon_zone(nb_lon,nb_lat,zone=zone)
    proj_plot=ccrs.PlateCarree()
    ax.add_patch(mpatches.Rectangle(xy=[lonlat[0][Y_min,X_min],lonlat[1][Y_min,X_min]], width=Lon_domaine_sup-Lon_domaine_inf, height=Lat_domaine_sup-Lat_domaine_inf,
                                    edgecolor='gray', fill=None, ls='--',
                                    alpha=0.5,
                                    transform=proj_plot, zorder=3))
    return ax
