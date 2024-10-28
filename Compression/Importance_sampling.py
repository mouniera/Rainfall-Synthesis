import numpy as np
from numpy.typing import NDArray
import random

def importance(fields: NDArray[np.float32],pmin: float,m: float,s: int) ->  NDArray[np.float32]:
    """ Calculates the probability of saving each sample in a dataset using
    the importance sampling method (see section 2b)

    Args:
        fields (NDArray[np.float32]): rainfall dataset with shape (samples,x,y) 
        pmin (float): minimum probability of saving a sample
        m (float): multiplying factor
        s (int): rainfall interest threshold

    Returns:
        NDArray[np.float32]: probability of saving each sample 
    """

    importance_list=np.empty([fields.shape[0]])
    for index,k in  enumerate(fields) :
        mean_exp=np.mean(1-np.exp(-k/s)) 
        importance=pmin+m*mean_exp
        importance=np.min((1,importance))
        importance_list[index]=importance
    return importance_list  

def random_select(proba: float) -> bool :
    """ Choice to save sample or not

    Args:
        proba (float): probability of saving the sample

    Returns:
        bool: saved or not
    """

    if random.random()<proba :
        return True
    else :
        return False

random_select_vect=np.vectorize(random_select)

def extract_data(data_RR: NDArray[np.float32],pmin: float,m: float,s: int) -> NDArray[np.float32]:
    """Extract data with importance sampling method

    Args:
        data_RR (NDArray[np.float32]): rainfall dataset
        pmin (float): minimum probability of saving a sample_
        m (float): multiplying factor
        s (int): rainfall interest threshold

    Returns:
        NDArray[np.float32]: dataset with selected samples after importance sampling
    """
    imp_vect=importance(data_RR,pmin,m,s)
    bool=random_select_vect(imp_vect)
    data_RR_select=data_RR[bool,:,:]
    return data_RR_select
