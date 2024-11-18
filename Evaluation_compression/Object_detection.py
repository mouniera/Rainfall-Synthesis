from astropy.convolution import convolve_fft,Gaussian2DKernel
import numpy as np
from pathlib import Path
import pickle
from tools import *
import skimage.measure as skimage
from scipy import ndimage
from numpy.typing import NDArray
    

def attributs (list_attributs: dict,field: NDArray,MB: str,zone: str,objet: str,type_AE_PE: str,lead_t: str) :
    """Update list_attibuts from field

    Args:
        list_attributs (dict): attribut list to update
        field (NDArray): binary field (0 outiside object, 1 inside)
        MB (str): "AROME-EPS member
        zone (str): subdomain (NW,NE,SW,SE or C)
        objet (str): object type (moderate, heavy,...)
        type_AE_PE (str): object detection in original members (type 'EPS') or in reconstructed fields (type 'AE')
        lead_t (str): AROME-EPS lead time
    """
    list_attributs[MB][zone][lead_t][type_AE_PE][objet]["num_objet"]=[]
    list_attributs[MB][zone][lead_t][type_AE_PE][objet]["area"]=[]
    list_attributs[MB][zone][lead_t][type_AE_PE][objet]["Y_Lat"]=[]
    list_attributs[MB][zone][lead_t][type_AE_PE][objet]["X_Lon"]=[]
    if np.max(field) < 0.99 :
        return 'done'
    else :
        field_m,num=skimage.label(field, return_num=True)
        unique, counts = np.unique(field_m, return_counts=True)
        list_objet=dict(zip(unique, counts))
        c_mass=ndimage.measurements.center_of_mass(field,field_m,index=list(range(1,num+1)))
        for obj in list(list_objet.keys()) :
            if obj !=0:
                list_attributs[MB][zone][lead_t][type_AE_PE][objet]["num_objet"].append(int(obj))
                list_attributs[MB][zone][lead_t][type_AE_PE][objet]["area"].append(int(list_objet[obj]))
                lat,lon=grid_to_lat_lon_zone(c_mass[obj-1],zone)
                list_attributs[MB][zone][lead_t][type_AE_PE][objet]["X_Lon"].append(round(lon,2))
                list_attributs[MB][zone][lead_t][type_AE_PE][objet]["Y_Lat"].append(round(lat,2))
        return 'done'

def contours(field: NDArray,kernels) -> list[NDArray]:
    """object detection similar to Davies et al (2006) (smooth and threshold)

    Args:
        field (NDArray): rainfall field
        kernels (_type_): Gaussian kernels for smoothing step

    Returns:
        list[NDArray]: list of binary fields (one per object type) with 0 if grid point outside of object type, 1 otherwise
    """
    run_config=load_yaml(path=Path(''), yaml_fname='config.yml')
    nb_lon=field.shape[-1]
    nb_lat=field.shape[-2]
    objects_RR=[np.zeros((nb_lat,nb_lon)) for i in range(len(run_config['Object_detection']['name']))]
    for i,obj_threshold in enumerate(run_config['Object_detection']['threshold']):
        if np.max(field)>=obj_threshold :
            conv_RR=convolve_fft(field,kernels[i])
            objects_RR[conv_RR>obj_threshold]=1.
    return objects_RR

def compute_attributs(date:str,run:str,list_MB:list[str],zone_l:list[str],lead_times:list[int],cumul_RR: int,type_AE_EPS: str,name_config=None):
    """Compute object detection and save object attributes in a dedicated pickle file

    Args:
        date (str): AROME-EPS date
        run (str): AROME-EPS runtime
        list_MB (list[str]): list of AROME-EPS members
        zone_l (list[str]): list of subdomains (among NW,NE,SW,SE and C)
        lead_times (list[int]): list of lead times
        cumul_RR (int): accumulated rainfall period (1h in the paper)
        type_AE_EPS (str): object detection in original members (type 'EPS') or in reconstructed fields (type 'AE')
        name_config (str, optional): if type 'AE', autoencoder name. Defaults to None.

    Returns:
        pickle file : dictionary with object attributes
    """
    run_config=load_yaml(path=Path(''), yaml_fname='config.yml')
    kernels_radius=run_config['Object_detection']['radius']
    kernels=[Gaussian2DKernel(i) for i in kernels_radius]
    path_data_AE,path_data_EPS=run_config['path_data_IA'],run_config['path_data_EPS']
    path_attributs_AE,path_attributs_EPS=run_config['path_attributs_IA'],run_config['path_attributs_EPS']
    liste_Attributs=AutoVivification()
    if type_AE_EPS=='AE':
        path_data_date=path_data_AE+'/'+name_config+'/'+date+'/R'+run+'/'
        path_out=path_attributs_AE+'/'+name_config+'/'
    else:
        path_data_date=path_data_EPS+'/'+date+'/R'+run+'/'
        path_out=path_attributs_EPS+'/'
    for mb in list_MB :
        for zone in zone_l :
            path_AE=f"{path_data_date}ConvAE_RR{str(cumul_RR)}h_MB{mb}_begin_{str(lead_times[0])}_end_{str(lead_times[-1])}_{zone}.hdf5"
            ae_zone=load_data_hdf5(path_AE)
            for i,lead_t in enumerate(lead_times) :
                liste_RR=contours(ae_zone[i],kernels)
                for o,objet in enumerate(run_config['Object_detection']['name']) :
                    attributs(liste_Attributs,liste_RR[o],mb,zone,objet,type_AE_EPS,str(lead_t).zfill(2))
    path_attributs=f"{path_out}Attributs_RR{str(cumul_RR)}h_{date}_R{run}_begin_{str(lead_times[0])}_end_{str(lead_times[-1])}.pkl"
    with open(path_attributs,'wb') as f:
        pickle.dump(liste_Attributs,f,pickle.HIGHEST_PROTOCOL)
    return 'done'