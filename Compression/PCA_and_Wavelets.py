from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pywt
import pickle
from numpy.typing import NDArray

#############################################################

#PCA and Wavelet code for dimension reduction 

#############################################################

def PCA_fit(data_RR:NDArray[np.float32],n_compo: int) :
    """fit PCA in order to compare with autoencoder

    Args:
        data_RR (NDArray[np.float32]): training database (same than autoencoder)
        n_compo (int): number of PC for compression
    """
    data_RR=data_RR.reshape((data_RR.shape[0],data_RR.shape[1]*data_RR.shape[2]))
    sc=StandardScaler()
    data_RR=sc.fit_transform(data_RR)
    with open("StandardScaler.file","wb") as f :
        pickle.dump(sc,f,pickle.HIGHEST_PROTOCOL) #should be saved for prediction step
    pca = PCA(n_components=n_compo)
    pca.fit(data_RR)
    with open("PCA_model_"+str(n_compo)+".file","wb") as f :
        pickle.dump(pca,f,pickle.HIGHEST_PROTOCOL)
    return "Done"

def PCA_prediction(RR_field:NDArray[np.float32], n_compo: int)-> NDArray[np.float32]:
    """Compute rainfall fields after PCA compression

    Args:
        RR_field (NDArray[np.float32]): rainfall fields
        n_compo (int): number of PC for compression

    Returns:
        NDArray[np.float32]: reconstructed rainfall fields
    """
    with open("PCA_model_"+str(n_compo)+".file","rb") as f :
        pca=pickle.load(f)
    with open("StandardScaler.file","rb") as f :
        sc=pickle.load(f)
    X,Y=RR_field.shape[1],RR_field.shape[2]
    RR_field=RR_field.reshape((RR_field.shape[0],RR_field.shape[1]*RR_field.shape[2]))
    RR_field=sc.transform(RR_field)
    components = pca.transform(RR_field)
    projected = pca.inverse_transform(components)
    projected_no_norm=sc.inverse_transform(projected)
    projected_no_norm=projected_no_norm.reshape((RR_field.shape[0],X,Y))
    return projected_no_norm

def Wavelet_prediction(RR_field: NDArray[np.float32],dim: int,wavelet_mother='coif2',mode='symmetric')-> NDArray[np.float32] :
    """Compute rainfall fields after Wavelet compression

    Args:
        RR_field (NDArray[np.float32]): rainfall fields
        dim (int): number of non-zero coefficient 
        wavelet_mother (str, optional):  mother wavelet. Defaults to 'coif2'.
        mode (str, optional): signal extension mode. Defaults to 'symmetric'.

    Returns:
        NDArray[np.float32]: reconstructed rainfall fields
    """
    wavelets_output=np.empty((RR_field.shape[0],RR_field.shape[1],RR_field.shape[2]))
    for index,RR in enumerate(RR_field):
        coeffs=pywt.wavedec2(RR, wavelet_mother, mode=mode)
        l_coeff=np.array(coeffs[0].flatten())
        for i in range(1,len(coeffs)):
            lh=np.array(coeffs[i][0]).flatten()
            lv=np.array(coeffs[i][1]).flatten()
            ld=np.array(coeffs[i][2]).flatten()
            l_coeff=np.append(l_coeff,lh)
            l_coeff=np.append(l_coeff,lv)
            l_coeff=np.append(l_coeff,ld)
        threshold=np.sort(l_coeff)[-dim]
        coeffs[0][coeffs[0]<threshold]=0.
        for i in range(1,len(coeffs)):
            coeffs[i][0][coeffs[i][0]<threshold]=0.
            coeffs[i][1][coeffs[i][1]<threshold]=0.
            coeffs[i][2][coeffs[i][2]<threshold]=0.
        recons=pywt.waverec2(coeffs, wavelet_mother, mode=mode)
        recons=np.float32(recons)
        wavelets_output[index]=recons
    return wavelets_output



