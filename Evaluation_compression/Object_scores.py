import numpy as np
from pathlib import Path
import os,pickle
from tools import *
import datetime
import geopy.distance


def dist_lat_lon(x_lat:float,x_lon:float,y_lat:float,y_lon:float) -> float:
    """calculate distance between two points in latitude/longitude."""
    x=(x_lat, x_lon)
    y=(y_lat, y_lon)
    distance=geopy.distance.geodesic(x,y).km #in kilometer
    return distance

def scores(contingency: NDArray) -> tuple :
    """compute HR,FAR and Accuracy from contingency table

    Args:
        Contingency (NDArray): contingency table

    Returns:
        tuple: HR,FAR, Accuracy
    """
    hr=round(contingency[1,1]/(contingency[1,1]+contingency[0,1]),2)
    far=round(contingency[1,0]/(contingency[1,1]+contingency[1,0]),2)
    accuracy=round((contingency[0,0]+contingency[1,1])/(np.sum(contingency)),2)
    return hr,far,accuracy

def update_contingency(l_EPS: list[int],l_AE: list[int]) -> tuple:
    """determine according to EPS and AE objects, if it's a correct detection, miss, false alarm or correct no detection

    Args:
        l_EPS (list[int]): list of EPS objects
        l_AE (list[int]): list of AE objects

    Returns:
        tuple: index where add +1 in the contingency table 
    """
    size_EPS=np.array(l_EPS).size
    size_AE=np.array(l_AE).size
    bool_EPS= (size_EPS !=0)
    bool_AE= (size_AE !=0)
    if bool_EPS and bool_AE: #corect detection
        return (1,1)
    elif not bool_EPS and not bool_AE : #correct no detection
        return (0,0)
    elif bool_EPS and not bool_AE : #miss
        return (0,1)
    else : #false alarm
        return (1,0) 

def contingency(table: NDArray,l_EPS:list[int],l_AE:list[int]) -> NDArray:
    """update contingency table (+1 for one of the four values)

    Args:
        table (NDArray): previous contingency table
        l_EPS (list[int]): list of EPS objects
        l_AE (list[int]): list of AE objects

    Returns:
        NDArray: updated contingency table
    """
    index=update_contingency(l_EPS,l_AE)
    table[index]+=1
    return table


def main(date: str,runtime: str,cumul_RR: int,name_config: str,dict_all: dict,dict_contingency: dict) :
    """Compare EPS and AE attributes to compute contingency table and mean distance for a specific date

    Args:
        date (str): AROME-EPS date
        runtime (str): AROME-EPS runtime
        cumul_RR (int): accumulated rainfall period (1h in the paper)
        name_config (str): autoencoder configuration
        dict_all (dict): dictionnary with all features 
        dict_contingency (dict): specific dictionnary for contingency tables

    Returns:
        update dict_all and dict_contingency
    """
    run_config=load_yaml(path=Path(''), yaml_fname='config.yml')

    list_lead_t=range(cumul_RR+1,run_config['AROME']['limit_sup']+1)
    list_MB=[str(k).zfill(2) for k in range(run_config['val_database']['list_MB'][0],run_config['val_database']['list_MB'][1]+1)]
    zone_l=run_config['val_database']['zone_l']

    #path_data
    path_attributs_AE,path_attributs_EPS=run_config['path_attributs_IA'],run_config['path_attributs_EPS']
    path_attributs_AE=path_attributs_AE+'/'+name_config+'/'
    path_attributs_AE_file=f"{path_attributs_AE}Attributs_RR{str(cumul_RR)}h_{date}_R{runtime}_ech_deb_{str(list_lead_t[0])}_ech_fin_{str(list_lead_t[-1])}.pkl"
    path_attributs_EPS_file=f"{path_attributs_EPS}Attributs_RR{str(cumul_RR)}h_{date}_R{runtime}_ech_deb_{str(list_lead_t[0])}_ech_fin_{str(list_lead_t[-1])}.pkl"

    #open pickle file
    with open(path_attributs_AE_file,'rb') as f:
        attributs_dict_AE=pickle.load(f)
    with open(path_attributs_EPS_file,'rb') as f:
        attributs_dict_PE=pickle.load(f)
    
    #loop over MB, subdomains, lead times and object types
    for mb in list_MB :
        for zone in zone_l :
            for lead_t in list_lead_t:
                ae=attributs_dict_AE[mb][zone][str(lead_t).zfill(2)]["AE"]
                eps=attributs_dict_PE[mb][zone][str(lead_t).zfill(2)]["EPS"]
                for objet in ["tot","mod","heavy"]:
                    ae_obj=ae[objet]
                    eps_obj=eps[objet]
                    contingency(dict_contingency[objet],eps_obj["num_objet"],ae_obj["num_objet"])
                    nb_objet_AE=len(ae_obj["num_objet"])
                    nb_objet_EPS=len(eps_obj["num_objet"])
                    area_AE=ae_obj["area"]
                    area_EPS=eps_obj["area"]
                    x_AE=ae_obj["X_Lon"]
                    x_EPS=eps_obj["X_Lon"]
                    y_AE=ae_obj["Y_Lat"]
                    y_EPS=eps_obj["Y_Lat"]
                      
                    if nb_objet_EPS !=0 and nb_objet_AE !=0 :
                        couple_best=[[]]*nb_objet_AE #find best match
                        dist_best=np.empty((nb_objet_AE))
                        for i in range(nb_objet_AE):
                            x_AE_obj_by_obj=x_AE[i]
                            y_AE_obj_by_obj=y_AE[i]
                            dist_min=5000 #5000 km 
                            for j in range(nb_objet_EPS):
                                x_EPS_obj_by_obj=x_EPS[j]
                                y_EPS_obj_by_obj=y_EPS[j]
                                dist=dist_lat_lon(y_AE_obj_by_obj,x_AE_obj_by_obj,y_EPS_obj_by_obj,x_EPS_obj_by_obj)
                                if dist<dist_min :
                                    dist_min=dist
                                    couple_best[i]=(i,j)
                                    dist_best[i]=dist_min
                        for (i,j) in couple_best:
                            dict_all[objet]["Dist"].append(round(dist_best[i],2))
                            dict_all[objet]["Area_EPS"].append(area_EPS[j])
                            dict_all[objet]["Area_AE"].append(area_AE[i])
                                

                        
                    if nb_objet_AE+nb_objet_EPS != 0 :
                        dict_all[objet]["number_objet_EPS"].append(nb_objet_EPS)
                        dict_all[objet]["number_objet_AE"].append(nb_objet_AE)
                        dict_all[objet]["Area_global_EPS"].append(np.sum(area_EPS))
                        dict_all[objet]["Area_global_AE"].append(np.sum(area_AE))
                        
                    if nb_objet_EPS !=0 :
                        dict_all[objet]["Area_allEPS"].extend(area_EPS)
                        if nb_objet_AE ==0 :
                            dict_all[objet]["Area_ND"].extend(area_EPS)
                    if nb_objet_AE !=0:
                        dict_all[objet]["Area_allAE"].extend(area_AE)
                        if nb_objet_EPS ==0 :
                            dict_all[objet]["Area_FA"].extend(area_AE)
                             
                  
    return 'done'




    

def Scores_all(name_config:str,cumul_RR:int) :
    """Compute object-oriented scores over the validation database

    Args:
        name_config (str): autoencoder configuration
        cumul_RR (int): accumulated rainfall period (1h in the paper)

    Returns:
        pickle file: dictionary is saved with different scores (HR,FAR,...)
    """
    run_config=load_yaml(path=Path(''), yaml_fname='config.yml')
    dict_tot={"HR" : 0, "FAR" : 0, "Accuracy" : 0, "Dist_mean" : 0, "number_objet_EPS" : [], "number_objet_AE" : [],
                "Area_global_EPS":[],"Area_global_AE":[], "Area_EPS" : [], "Area_AE" : [], "Dist" : [], 
                "Area_ND" : [], "Area_FA" : [], "Area_allEPS" : [], "Area_allAE" : []}
    
    dict_mod={"HR" : 0, "FAR" : 0, "Accuracy" : 0, "Dist_mean" : 0, "number_objet_EPS" : [], "number_objet_AE" : [],
                "Area_global_EPS":[],"Area_global_AE":[], "Area_EPS" : [], "Area_AE" : [], "Dist" : [], 
                "Area_ND" : [], "Area_FA" : [], "Area_allEPS" : [], "Area_allAE" : []}
    
    dict_heavy={"HR" : 0, "FAR" : 0, "Accuracy" : 0, "Dist_mean" : 0, "number_objet_EPS" : [], "number_objet_AE" : [],
                "Area_global_EPS":[],"Area_global_AE":[], "Area_EPS" : [], "Area_AE" : [], "Dist" : [], 
                "Area_ND" : [], "Area_FA" : [], "Area_allEPS" : [], "Area_allAE" : []}
    dict_all={"tot": dict_tot, "mod" : dict_mod, "heavy" : dict_heavy}
    dict_contingency={"tot": np.zeros((2,2)), "mod" : np.zeros((2,2)), "heavy" : np.zeros((2,2))}

    #validation database
    date_begin,date_end,delta_run=run_config['val_database']['begin'],run_config['val_database']['end'],run_config['val_database']['delta_run']
    start=datetime.datetime.strptime(date_begin,"%Y%m%d%H")
    end=datetime.datetime.strptime(date_end,"%Y%m%d%H")
    seconds_in_day=3600*24
    diff_seconds=(end-start).days*seconds_in_day+(end-start).seconds
    yymmddhh= [start + datetime.timedelta(seconds=x) for x in range(0,diff_seconds+1,int(delta_run)*3600)]

    #loop over the list of dates
    for k in range(len(yymmddhh)):
        yymmddhh[k]=yymmddhh[k].strftime("%Y%m%d%H")
        date_now=yymmddhh[k][:8]
        run=yymmddhh[k][-2:]
        main(date_now,run,cumul_RR,name_config,dict_all,dict_contingency) #dict_all and dict_contingency are updated

    #HR,FAR,Acc and Dist_mean for each object type
    for objet in ["tot","mod","heavy"] :
        hr,far,acc=scores(dict_contingency[objet])
        dict_all[objet]["HR"]=hr
        dict_all[objet]["FAR"]=far
        dict_all[objet]["Accuracy"]=acc
        dist_mean=round(np.mean(dict_all[objet]["Dist"]),2)
        dict_all[objet]["Dist_mean"]=dist_mean

    path_out_scores=f"{run_config['path_scores']}/{name_config}/RR{str(cumul_RR)}/"
    if not os.path.exists(path_out_scores):
        os.system('mkdir -p '+ path_out_scores)
    path_scores=f"{path_out_scores}dict_scores_{name_config}_RR{str(cumul_RR)}.pkl"
    with open(path_scores,'wb') as f:
        pickle.dump(dict_all,f)
    return "Ok"

