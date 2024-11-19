from tools import *
from pathlib import Path
import matplotlib.pyplot as plt
import susi
import numpy as np
import pickle
from numpy import linalg as LA
from numpy.typing import NDArray





def load_data_latent(zone:str,latent: str,cumul: str) -> NDArray:
    """ Load latent space data"""
    run_config=load_yaml(path=Path(''), yaml_fname='config.yml')
    path_save_latent=f"{run_config['path_data_latent']}/Data_latent_space_{latent}/Zone_{zone}_RR{cumul}_only_max_N2_PHY.hdf5"  
    return load_data_hdf5(path_save_latent)



def num_class(x: int,y: int) -> int:
    """ change numbering from 2D to 1D"""
    run_config=load_yaml(path=Path(''), yaml_fname='config.yml')
    return run_config['n_col_SOM']*x+y

num_class_vect=np.vectorize(num_class)




def som(data_latent: NDArray,n_iter: int) -> 'susi SOM object, NDArray':
    """Self-Organizing Maps from susi package 

    Args:
        Data_latent (NDArray): Data, format : [sample, latent_space]
        n_iter (int): number of iterations

    Returns:
        tuple(susi SOM object, NDArray): susi SOM object and list of classes from Data_latent samples
    """
    run_config=load_yaml(path=Path(''), yaml_fname='config.yml')
    test_classif=susi.SOMClustering(n_rows=run_config['n_row_SOM'],n_columns=run_config['n_col_SOM'], n_iter_unsupervised=n_iter, train_mode_unsupervised='batch', random_state=1)
    test_classif.fit(data_latent)
    clusters=np.array(test_classif.get_clusters(data_latent))
    classes=num_class_vect(clusters[:,0],clusters[:,1])
    return test_classif,classes

def main(zone:str,latent: str,cumul: str):
    """SOM clustering from latent space data 

    Args:
        zone (str): subdomain (NW,NE,SW,SE or C)
        latent (str): latent space size (10,20 or 40)
        cumul (str): rainfall accumulation

    Returns:
        SOM clastering and associated classes from latent space data are saved in pickle file
    """
    run_config=load_yaml(path=Path(''), yaml_fname='config.yml')
    data_latent=load_data_latent(zone,latent,cumul)
    classif_list=[]
    classes_list=[]
    for n_iter in [1000,2000,5000,10000]:
        test_classif,classes=som(data_latent,n_iter)
        classif_list.append(test_classif)
        classes_list.append(classes)
        u_matrix_1=test_classif.get_u_matrix()
        plt.imshow(np.squeeze (u_matrix_1), cmap="Greys")
        plt.colorbar()
        plt.title('SOM online, U matrix, n_iter= '+str(n_iter))
        plt.savefig(f"Test_SOM_batch_subgroups_{run_config['n_row_SOM']}by{run_config['n_col_SOM']}_dim{latent}_RR{cumul}_{str(n_iter)}_{zone}.png")
        plt.close()
        with open(f"Classif_SOM_subgroups_{run_config['n_row_SOM']}by{run_config['n_col_SOM']}_dim{latent}_RR{cumul}_{zone}.pkl",'wb') as f:
            pickle.dump(classif_list,f)
        with open(f"Classes_SOM_subgroups_{run_config['n_row_SOM']}by{run_config['n_col_SOM']}_dim{latent}_RR{cumul}_{zone}.pkl",'wb') as f:
            pickle.dump(classes_list,f)  
    return 'OK'


