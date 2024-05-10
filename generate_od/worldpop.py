import requests

from tqdm import tqdm
from io import BytesIO
from multiprocessing import Pool

import numpy as np
import geopandas as gpd
from rasterio.mask import mask
from rasterio.io import MemoryFile

from .utils import calculate_utm_epsg


def population_one_region(args):
    '''
    get the population of given region, 
    by aggregating the all pixels located 
    in that region from worldpop tiff.
    '''
    # arguments
    region = args

    # get the worldpop tiff for the region
    base_url = 'https://worldpop.arcgis.com/arcgis/rest/services/WorldPop_Total_Population_100m/ImageServer/exportImage?f=image&format=tiff&noData=0&'
    left, bottom, right, top = region.geometry.bounds
    url = base_url + f"bbox={left},{bottom},{right},{top}"

    max_times = 10
    flag = False
    for i in range(max_times):
        try:
            response = requests.get(url)
            response.raise_for_status()
            flag = True
            break
        except requests.exceptions.HTTPError as http_err:
            continue
        except requests.exceptions.ConnectionError as conn_err:
            continue
        except requests.exceptions.Timeout as timeout_err:
            continue
        except requests.exceptions.RequestException as req_err:
            continue

    if flag == False:
        raise("Network error for accessing https://worldpop.arcgis.com/.")

    img_bytes = BytesIO(response.content)
    with MemoryFile(img_bytes) as memfile:
        with memfile.open() as raster:
            # get the population of the region with the boundary of the region
            coords = region.geometry
            try:
                out_img, _ = mask(dataset=raster, shapes=[coords], crop=True)
                population = float(out_img[out_img > 0].sum())
            except:
                population = 0
            
            if not isinstance(population, (int, float)):
                raise ValueError(f"Population of {region} is invalid: {population}")

            # EPSG code calculation
            epsg = calculate_utm_epsg(left, bottom)
            # get the area of the region
            geo_series = gpd.GeoSeries([coords], crs="EPSG:4326")
            area = geo_series.to_crs(f"EPSG:{epsg}").area.item() / 1e6

    return population, area



def worldpop(area_shp, token=None, num_proc=10):
    '''
    get regional population via worldpop posited on 
    Esri ArcGIS Living Atlas of the World "WorldPop_Total_Population_100m"
    '''
    # args
    args = [region for _, region in area_shp.iterrows()]

    # get the population of regions in the target area
    # # parallel version
    # populations = []
    # areasizes = []
    # with Pool(processes=num_proc) as pool:
    #     with tqdm(total=len(area_shp), desc=f" -- Population of regions") as pbar:
    #         for result in pool.imap(population_one_region, args):
    #             populations.append(result[0])
    #             areasizes.append(result[1])
    #             pbar.update(1)

    # sequential version
    populations = []
    areasizes = []
    for _, region in tqdm(area_shp.iterrows(), total=area_shp.shape[0], desc=f" -- Population of regions"):
        population, areasize = population_one_region(region)
        populations.append(population)
        areasizes.append(areasize)
    
    # save the population and areasize of the area
    feat = np.array([populations, areasizes]).T

    return feat