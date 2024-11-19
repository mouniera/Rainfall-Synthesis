import numpy as np
import cartopy.crs as ccrs
import datetime,argparse
import matplotlib.pyplot as plt
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy.ndimage import center_of_mass
from PIL import Image,ImageDraw,ImageFont,ImageOps
from tools import *

def extract_image_members(date:str ,run: str ,zone:str ,mb: str ,leadt: int ,cumul_RR: str) -> Image:
    """Extract image from AROME-EPS members (already computed and saved by a different way)

    Args:
        date (str): AROME-EPS date
        run (str): AROME-EPS runtime
        zone (str): subdomain (NW,NE,SW,SE or C)
        mb (str): AROME-EPS member
        leadt (int): member lead time
        cumul_RR (str): accumulated rainfall period

    Returns:
        Image: image of AROME-EPS member for a specific lead time
    """
    path_config=load_yaml(path=Path(''), yaml_fname='path.yml')
    path_out_plots=path_config['data']['path_save_plot_RR_Scenarios']+zone+'/'+date+'/'+run+'/'+mb+'/'
    name_file='RR'+cumul_RR+'_'+date+'_'+run+'_'+mb+'_'+str(leadt)+'.png'
    return Image.open(path_out_plots+name_file)

def coeff_lim_plot_members(nb_image=12) -> tuple[float]:
    """Some hyerparameters for the plot of individual members"""
    if nb_image>=12:
        return 3,2.5
    elif nb_image>=2:
        return 2.8,2.1
    else:
        return 1.5,1.3
    

def init_image_members(nb_plot: int,size_w: int,size_h: int) -> tuple:
    """Init multiple values for the plot related to the number of image to concatenate

    Args:
        nb_plot (int): number of image to concatenate
        size_w (int): width of one individual image
        size_h (int): height of one individual image

    Returns:
        tuple: (plot: Init Image,nb_c: nb of column,nb_r: nb of row,shift_title: shift of plot title,text_size: title sizefont,
        l_w: constant for title position in order to take into account the colorbar object)
    """
    run_config=load_yaml(path=Path(''), yaml_fname='path.yml')
    #multiple arbitrary choices for plot
    if nb_plot>=10:
        nb_c,nb_r=4,3
        text_size=28
        shift_title=75
        colorbar=Image.open(run_config['Plot']['colorbar']+'Image_colorbar_n12.png')  
    elif nb_plot>=7:
        nb_c,nb_r=3,3
        text_size=28
        shift_title=72
        colorbar=Image.open(run_config['Plot']['colorbar']+'Image_colorbar_n12.png')  
    elif nb_plot>=5:
        nb_c,nb_r=3,2
        text_size=25
        shift_title=53
        colorbar=Image.open(run_config['Plot']['colorbar']+'Image_colorbar_n4.png')  
    elif nb_plot>=3:
        nb_c,nb_r=2,2
        text_size=20
        shift_title=42
        colorbar=Image.open(run_config['Plot']['colorbar']+'Image_colorbar_n4.png')  
    elif nb_plot==2:
        nb_c,nb_r=2,1
        text_size=20
        shift_title=42
        colorbar=Image.open(run_config['Plot']['colorbar']+'Image_colorbar.png')  
    else:
        nb_c,nb_r=1,1  
        text_size=18
        shift_title=75
        colorbar=Image.open(run_config['Plot']['colorbar']+'Image_colorbar.png')  
    ratio_h= (nb_r*size_h)/float(colorbar.height)
    l_w=int(float(colorbar.width)*ratio_h)
    plot=Image.new('RGB',(nb_c*size_w+l_w,nb_r*size_h+shift_title),color=(255,255,255))
    colorbar= colorbar.resize((l_w,nb_r*size_h), Image.LANCZOS)
    plot.paste(colorbar, (nb_c*size_w,shift_title))
    return plot,nb_c,nb_r,shift_title,text_size,l_w
    
def title_plot_members(class_n: str,date_format: str ,size_scenario_tot: int ,nb_diff_image:int ,count_image_diff:int ,
                         image_plot: Image,w_image: int,text_size: int, cumul_RR: str ='1', nb_paste_image: int =12) -> "title text":
    """create the title for the member plot in the same class

    Args:
        class_n (str): member class
        date_format (str): date and hours concerned
        size_scenario_tot (int): total size of the class
        nb_diff_image (int): number of different images for plotting all members (one plot if less than 12 members)
        count_image_diff (int): count of current image 
        image_plot (Image): Image (where adding title)
        w_image (int): width of image
        text_size (int): fontsize of title
        cumul_RR (str, optional): accumulated rainfall period. Defaults to '1'.
        nb_paste_image (int, optional): nb of members represented in this image. Defaults to 12.

    Returns:
        title text: title is added on the image
    """
    if nb_paste_image==1:
        title1=f"RR{cumul_RR}h members on {date_format}"
        if nb_diff_image==1:
            title2=f"Classe {class_n} ({str(size_scenario_tot)} members)"
        else:
            title2=f"Classe {class_n} ({str(size_scenario_tot)} members, part {str(count_image_diff)}/{str(nb_diff_image)})"
        Text_add=ImageDraw.Draw(image_plot)
        font = ImageFont.truetype("gautamib.ttf", text_size) 
        _, _,w_text,h_text=Text_add.textbbox((0,0),title1,font=font)
        Text_add.text(((w_image-w_text)/2,0),title1,fill=(0, 0, 0),font=font) 
        _, _,w_text,h_text=Text_add.textbbox((0,0),title2,font=font)
        Text_add.text(((w_image-w_text)/2,(text_size+3)),title2,fill=(0, 0, 0),font=font)  
        return 'OK'   
    elif nb_diff_image==1:
        if size_scenario_tot==1:
            title=f"RR{cumul_RR}h members on {date_format}, Classe {class_n} ({str(size_scenario_tot)}) member)"
        else :
            title=f"RR{cumul_RR}h members on {date_format}, Classe {class_n} ({str(size_scenario_tot)}) members)"
    else:
        title=f"RR{cumul_RR}h members on {date_format}, Classe {class_n} ({str(size_scenario_tot)}, part {str(count_image_diff)}/{str(nb_diff_image)})"
    Text_add=ImageDraw.Draw(image_plot)
    font = ImageFont.truetype("gautamib.ttf", text_size)
    _, _,w_text,h_text=Text_add.textbbox((0,0),title,font=font)
    Text_add.text(((w_image-w_text)/2,0),title,fill=(0, 0, 0),font=font)
    return 'OK'

def genere_plot_members(class_n: int,class_size: int, date: str,run_eps: str, leadt_eps: int, mb_eps: list[int],
leadt_plot: str,zone: str,path_save: Path,cumul_RR: str) -> 'png files':
    """Create plot with individual rainfall of members in the same class.

    Args:
        class_n (int): member class
        class_size (int): total size of the class
        date (str): AROME-EPS date
        run_eps (str): AROME-EPS runtime
        leadt_eps (int): AROME-EPS lead time
        mb_eps (list[int]): list of AROME-EPS members in the class
        leadt_plot (str): leadtime with format %d/%m/%Y - %HZ
        zone (str): subdomain (NW,NE,SW,SE or C)
        path_save (Path): path where plot will be saved
        cumul_RR (str): accumulated rainfall period

    Returns:
        png files: png files with the AROME-EPS members in the same class
    """
    run_config=load_yaml(path=Path(''), yaml_fname='config.yml')
    nb_diff_image=(class_size-1)//12 + 1 #no more 12 members per image
    count_image_diff=0
    [size_w,size_h]=run_config['Plot']['size_final']
    image_plot,nb_c,nb_l,decal_titre,text_size,l_w=init_image_members(class_size,size_w,size_h)
    nb_paste_image=0

    
    for i in mb_eps:
        im_PEARO=extract_image_members(date,run_eps,zone,str(i).zfill(2),leadt_eps,cumul_RR)
        num_ligne=nb_paste_image//nb_c
        num_col=nb_paste_image-num_ligne*nb_c
        image_plot.paste(im_PEARO, (size_w*num_col,size_h*num_ligne+decal_titre))
        nb_paste_image+=1
        if nb_paste_image==12:
            count_image_diff+=1
            title_plot_members(str(class_n),leadt_plot,class_size,nb_diff_image,count_image_diff,image_plot,nb_c*size_w+l_w,text_size,cumul_RR)
            image_plot=ImageOps.contain(image_plot,(int(coeff_lim_plot_members()[0]*size_w),int(coeff_lim_plot_members()[1]*size_h)))
            image_plot.save(str(path_save)+'/classe'+str(class_n)+'-'+str(int(run_eps))+'-'+str(leadt_eps)+'-'+str(count_image_diff)+'.png')
            image_plot,nb_c,nb_l,decal_titre,text_size,l_w=init_image_members(class_size-12*count_image_diff,size_w,size_h)
            if count_image_diff*nb_paste_image==class_size:
                return 'done'
            else:
                nb_paste_image=0
            
            
    count_image_diff+=1
    title_plot_members(str(class_n),leadt_plot,class_size,nb_diff_image,count_image_diff,image_plot,nb_c*size_w+l_w,text_size,cumul_RR,nb_paste_image)
    image_plot=ImageOps.contain(image_plot,(int(coeff_lim_plot_members(nb_paste_image)[0]*size_w),int(coeff_lim_plot_members(nb_paste_image)[1]*size_h)))
    image_plot.save(str(path_save)+'/classe'+str(class_n)+'-'+str(int(run_eps))+'-'+str(leadt_eps)+'-'+str(count_image_diff)+'.png')
    return 'OK'


def main(date: str, run: str, cumul_RR: str, zone: str) -> 'multiple plots':
    """Create final product for the rainfall classification (cf section 4)

    Args:
        date (str): AROME-EPS date
        run (str): AROME-EPS runtime
        cumul_RR (str): accumulated rainfall period
        zone (str): subdomain (NW,NE,SW,SE or C)

    Returns:
        multiple plots: rainfall synthesis plot + plots for individual rainfall forecasts. 
        AROME-EPS member plot are grouped by rainfall class.
    """
    run_config=load_yaml(path=Path(''), yaml_fname='config.yml')
     

    path_out_plots=Path(run_config['data']['path_save_classif']+'/final_plot/'+date+'/R'+run+'/'+zone+'/')
    path_out_plots.mkdir(parents=True,exist_ok=True)

    leadt_min_glob,leadt_max_glob=run_config['AROME']['limit_inf']+int(cumul_RR),run_config['AROME']['limit_sup'] #RR1 not available at first lead time
    eps_datetime=datetime.datetime.strptime(date+run,"%Y%m%d%H")
    date_min_plot=eps_datetime+datetime.timedelta(seconds=3600*int(leadt_min_glob))
    eps_str=eps_datetime.strftime("%d/%m/%Y - %HZ")

    #Prepare_data for plots
    smooth_Std=load_data_hdf5(run_config['Plot']['patterns_plot'])
    lonlat=extract_lonlat()
    [x_min,y_min]=run_config['Zoom'][zone]
    nb_lat,nb_lon=run_config['Plot']['Size_Y'],run_config['Plot']['Size_X']
    x_coord,y_coord=lonlat[0][y_min:(y_min+nb_lat),x_min:(x_min+nb_lon)],lonlat[1][y_min:(y_min+nb_lat),x_min:(x_min+nb_lon)]   
    [lim_weak_classe,lim_moderate_classe]=run_config['Plot']['severity_class']

    #list_member
    if run_config['AROME']['membre0']:
        list_MB=range(run_config['AROME']['n_member'])
    else:
        list_MB=range(1,run_config['AROME']['n_member']+1)


    nb_leadt=leadt_max_glob-leadt_min_glob+1
    for time_plot in [h for h in (date_min_plot+datetime.timedelta(seconds=3600*int(hour)) for hour in range(nb_leadt))]:
        leadt_plot=time_plot.strftime("%d/%m/%Y - %HZ")

        #Plot_init
        fig=plt.figure(figsize=(25,9))
        proj=ccrs.Stereographic(central_latitude=46.7,central_longitude=2)
        proj_plot=ccrs.PlateCarree()
        axes_class= (GeoAxes,dict(projection=proj))
        grid = AxesGrid(fig, 111, axes_class=axes_class,nrows_ncols=(1,3),axes_pad=1,label_mode='')
        ax_weak=background_map_scenarios(zone,lonlat,ax_predefined=grid[0])
        ax_weak.set_title('light rainfall (1-10mm)',fontsize=15)
        ax_moderate=background_map_scenarios(zone,lonlat,ax_predefined=grid[1])
        ax_moderate.set_title('moderate rainfall (5-15mm)',fontsize=15)
        ax_heavy=background_map_scenarios(zone,lonlat,ax_predefined=grid[2])
        ax_heavy.set_title('heavy rainfall (10-30mm)',fontsize=15)

        classes_eps=np.full([len(list_MB)],fill_value=-1) #classes and severity

        #New_EPS
        leadt_eps=find_leadt(now=time_plot, ref=datetime.datetime.strptime(date+run,"%Y%m%d%H"))
        for m,mb in enumerate(list_MB):
            path_file_newPE=Path(run_config['path_save_classif']+'/'+zone+'/'+date+'/Scenarios_'+date+'_'+str(mb).zfill(2)+'_'+zone+'_RR'+cumul_RR+'.hdf5')
            if path_file_newPE.is_file():
                with h5py.File(path_file_newPE,"r") as f:
                    data_file='/'+str(run.zfill(2))+'/'+str(leadt_eps)
                    if data_file+'/Classe' in f:
                        classes_eps[m]=f[data_file+'/Classe'][()]




        classes_available=classes_eps[classes_eps >= 0]
        n_available=len(classes_available)
        color_pattern=plt.get_cmap('cool',n_available+1)
        classes, counts = np.unique(classes_available, return_counts=True)
        for ind_c, classe in enumerate(classes):
            size_classe=counts[ind_c]
            index_eps_classe=np.where(classes_eps==classe)[0]
            mb_eps=np.array(list_MB)[index_eps_classe]

            color_classe=color_pattern(size_classe/n_available)
            map_for_center_mass=np.zeros([nb_lat,nb_lon])
            alpha_opacity=0.25*(2*size_classe+n_available-3)/(n_available-1) #empirical formula
            size_font=run_config['Plot']['size_font_num_classe']

            #classes no/few precip
            if classe ==0: 
                    ax_no_precip=fig.add_axes(run_config['Plot']['classe0']['Rect'])
                    ax_no_precip.get_xaxis().set_visible(False)
                    ax_no_precip.get_yaxis().set_visible(False)
                    [text_x,text_y]=run_config['Plot']['classe0']['Posi_text']
                    ax_no_precip.text(text_x,text_y,'No or very light rainfall (0-5 mm) \n ('+str(size_classe)+' members)',fontfamily='serif',
                                       fontsize=run_config['Plot']['classe0']['Size_font'], fontweight='bold')
                    rect = ax_no_precip.patch
                    rect.set_facecolor(color_classe)

                    #individual members
                    path_save_vignettes=Path(str(path_out_plots)+'/no_few_precip/')
                    path_save_vignettes.mkdir(parents=True,exist_ok=True)
                    genere_plot_members(classe,size_classe,date,run, leadt_eps, mb_eps,
                        leadt_plot,zone,path_save_vignettes,cumul_RR)
            
            #classes weak precip
            elif classe <=lim_weak_classe:
                    threshold_std_weak=0.3
                    ax_weak.contourf(x_coord,y_coord,smooth_Std[classe],levels=np.array([threshold_std_weak,100]),colors=[tuple(color_classe)],zorder=4,transform=proj_plot,alpha=alpha_opacity)
                    map_for_center_mass[smooth_Std[classe]>threshold_std_weak]=1.
                    center_mass_classe=center_of_mass(map_for_center_mass)
                    x_text,y_text=grid_to_lat_lon_zone([center_mass_classe[1],center_mass_classe[0]],x_min,y_min)
                    ax_weak.text(x_text,y_text,str(size_classe),transform=proj_plot,fontfamily='serif', fontsize=size_font, 
                                           fontweight='bold',zorder=4,ha='center',bbox=dict(facecolor='white', edgecolor='white',pad=0.1))

                    #individual members
                    path_save_vignettes=Path(str(path_out_plots)+'/weak_precip/Classe'+str(classe))
                    path_save_vignettes.mkdir(parents=True,exist_ok=True)
                    genere_plot_members(classe,size_classe,date,run, leadt_eps, mb_eps,
                        leadt_plot,zone,path_save_vignettes,cumul_RR)
                           
            #classes moderate precip
            elif classe <=lim_moderate_classe:
                    threshold_std=0.75*np.max(smooth_Std[classe])
                    ax_moderate.contourf(x_coord,y_coord,smooth_Std[classe],levels=np.array([threshold_std,100]),colors=[tuple(color_classe)],alpha=alpha_opacity,zorder=4,transform=proj_plot)
                    map_for_center_mass[smooth_Std[classe]>threshold_std]=1.
                    center_mass_classe=center_of_mass(map_for_center_mass)
                    x_text,y_text=grid_to_lat_lon_zone([center_mass_classe[1],center_mass_classe[0]],x_min,y_min)
                    ax_moderate.text(x_text,y_text,str(size_classe),transform=proj_plot,fontfamily='serif', fontsize=size_font,
                                      fontweight='bold',zorder=4,ha='center',bbox=dict(facecolor='white', edgecolor='white',pad=0.1))
                    #individual members
                    path_save_vignettes=Path(str(path_out_plots)+'/moderate_precip/Classe'+str(classe))
                    path_save_vignettes.mkdir(parents=True,exist_ok=True)
                    genere_plot_members(classe,size_classe,date,run, leadt_eps, mb_eps,
                        leadt_plot,zone,path_save_vignettes,cumul_RR)

            #classes heavy precip        
            else: 
                    threshold_std=0.75*np.max(smooth_Std[classe])
                    ax_heavy.contourf(x_coord,y_coord,smooth_Std[classe],levels=np.array([threshold_std,100]),colors=[tuple(color_classe)],alpha=alpha_opacity,zorder=4,transform=proj_plot)
                    map_for_center_mass[smooth_Std[classe]>threshold_std]=1.
                    center_mass_classe=center_of_mass(map_for_center_mass)
                    x_text,y_text=grid_to_lat_lon_zone([center_mass_classe[1],center_mass_classe[0]],x_min,y_min)
                    ax_heavy.text(x_text,y_text,str(size_classe),transform=proj_plot,fontfamily='serif', fontsize=size_font, 
                                  fontweight='bold',zorder=4,ha='center',bbox=dict(facecolor='white', edgecolor='white',pad=0.1))
                    
                    #individual members
                    path_save_vignettes=Path(str(path_out_plots)+'/precip_forte/Classe'+str(classe))
                    path_save_vignettes.mkdir(parents=True,exist_ok=True)
                    genere_plot_members(classe,size_classe,date,run, leadt_eps, mb_eps,
                        leadt_plot,zone,path_save_vignettes,cumul_RR)

        if 0 not in classes:
                ax_no_precip=fig.add_axes(run_config['Plot']['classe0']['Rect'])
                ax_no_precip.get_xaxis().set_visible(False)
                ax_no_precip.get_yaxis().set_visible(False)
                [text_x,text_y]=run_config['Plot']['classe0']['Posi_text']
                ax_no_precip.text(text_x,text_y*1.5,'No or very light rainfall (0-5mm)',fontfamily='serif', 
                                  fontsize=run_config['Plot']['classe0']['Size_font'], fontweight='bold',zorder=4)
                rect = ax_no_precip.patch
                rect.set_facecolor('white')
        

        #Position_title
        fig.text(0.3,0.89,"Rainfall synthesis:", ha="left", va="bottom",color="red", fontsize=16, fontweight='bold')
        fig.text(0.65,0.89,"Valid:", ha="left", va="bottom",color="red", fontsize=16, fontweight='bold')
        fig.text(0.43,0.89,"AROME-EPS", ha="left", va="bottom",color="black", fontsize=16, fontweight='bold')
        fig.text(0.5,0.89,eps_str, ha="left", va="bottom",color="black", fontsize=16) 
        fig.text(0.7,0.89,leadt_plot, ha="left", va="bottom",color="black", fontsize=16)    
        plt.savefig(str(path_out_plots)+'/Cartes_SOM_ech'+str(leadt_eps)+'.png',bbox_inches="tight")  
    return 'Done'
        
        
        
parser = argparse.ArgumentParser(description = 'Final_product_classification')
parser.add_argument('Date', help='AROME-EPS date (format YYYYMMDD)')
parser.add_argument('Run', help='Runtime (03, 09, 15 or 21)')
parser.add_argument("Zone", help="Subdomain (NW, NE, SW, SE or C)")
parser.add_argument('--Cumul_RR', help='Rainfall accumulation (default: 1h)',default='1')
args = parser.parse_args()
main(args.Date, args.Run.zfill(2), Cumul_RR=args.Cumul_RR, Zone=args.Zone)




        

