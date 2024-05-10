import os
import time
import shutil
import requests

from tqdm import tqdm
from io import BytesIO
from urllib import request
from multiprocessing import Pool

import numpy as np

import torch, open_clip

from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from scipy.ndimage import binary_fill_holes

from .utils import *






def download_one_tile(args):
    '''
    Download the tile for the given x and y coordinates.
    From Esri living atlas, the token is required to download the tiles.

    Depending on the API of the tile server.
    '''
    # args
    X, Y, token = args

    # the url of the tile
    base_url = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/41468/15/'
    url = base_url + str(Y) + '/' + str(X) + '?token=' + token

    # the filename of the tile
    filename = str(Y) + "_" + str(X)

    try:
        try:
            response = requests.get(url)
            response.raise_for_status()
            img_bytes = BytesIO(response.content)
            return filename, img_bytes
        except:
            time.sleep(1)
            response = requests.get(url)
            response.raise_for_status()
            img_bytes = BytesIO(response.content)
            return filename, img_bytes
    except:
        return filename
    

def download_all_tiles(Y_X, token, num_proc=50):
    '''
    Download all the tiles for the given x and y coordinates.
    From Esri living atlas, the token is required to download the tiles.

    The tiles are downloaded into the given directory.
    
    return: failed tiles.
    '''

    # the arguments for the download_one_tile function
    tile_args = [(int(X), int(Y), token) for Y, X in [y_x.split("_") for y_x in Y_X]]

    # download the tiles
    cached_tiles = {}
    fail_tile_list = []

    # # parallel version
    # pool = Pool(processes=num_proc)
    # with tqdm(total=len(Y_X), desc=" -- Downloading remaining tiles...") as t:
    #     for result in pool.imap(download_one_tile, tile_args):
    #         if isinstance(result, tuple):
    #             tile_no, img = result
    #             cached_tiles[tile_no] = img
    #         elif isinstance(result, str):
    #             fail_tile_list.append(result)
    #         t.update(1)

    # sequential version
    for arg in tqdm(tile_args, desc=" -- Downloading remaining tiles..."):
        result = download_one_tile(arg)
        if isinstance(result, tuple):
            tile_no, img = result
            cached_tiles[tile_no] = img
        elif isinstance(result, str):
            fail_tile_list.append(result)
    
    return cached_tiles, fail_tile_list


def concat_img_one_region(args):
    '''
    Concatenate the tiles to obtain the regional satellite images.
    '''
    # args
    region, tiles, files = args
    
    # the region should not be too large
    if len(tiles) < 20000:
        # get the x and y coordinates of the tiles
        ys, xs = zip(*[(int(i.split('_')[0]), int(i.split('_')[1])) for i in tiles])

        # get the min and max x and y coordinates
        row_min, col_min, row_max, col_max = min(ys), min(xs), max(ys), max(xs)

        # generate the mask
        mask_width = col_max - col_min + 1
        mask_height = row_max - row_min + 1

        mask_whole = np.zeros([mask_height * 256, mask_width * 256, 3], dtype=np.int32)
        
        # the pixel coordinates
        pixel_coords = []
        for tile in tiles:
            try:
                # read the tile
                img_temp = Image.open(files[tile])
                
                # get the tile's x and y coordinates
                row, col = [int(x) for x in tile.split("_")]

                # convert the tile's x and y coordinates to the longitude and latitude
                minlat, minlon = XY2deg(col, row, 15)
                maxlat, maxlon = XY2deg(col+1, row+1, 15)
                
                if region.iloc[0].geometry.geom_type != 'MultiPolygon':
                # if the region is not a multi-polygon
                    for coord in region.iloc[0].geometry.exterior.coords:
                        # convert the longitude and latitude to the pixel coordinates, offset
                        x = (coord[0] - minlon) * (256 / (maxlon - minlon))
                        y = (coord[1] - minlat) * (256 / (maxlat - minlat))

                        # convert the pixel coordinates to the regional pixel coordinates, overall
                        temp_x, temp_y = x + (col - col_min) * 256, y + (row - row_min) * 256
                        if temp_x > 256 * mask_width:
                            temp_x = 256 * mask_width
                        if temp_y > 256 * mask_height:
                            temp_y = 256 * mask_height
                        if temp_x < 0:
                            temp_x = 0
                        if temp_y < 0:
                            temp_y = 0

                        # store the pixel coordinates
                        pixel_coords.append((temp_x, temp_y))
                else:
                # if the region is a multi-polygon
                    for polygon in region.iloc[0].geometry.geoms:
                        for coord in polygon.exterior.coords:
                            # convert the longitude and latitude to the pixel coordinates, offset
                            x = (coord[0] - minlon) * (256 / (maxlon - minlon))
                            y = (coord[1] - minlat) * (256 / (maxlat - minlat))

                            # convert the pixel coordinates to the regional pixel coordinates, overall
                            temp_x, temp_y = x + (col - col_min) * 256, y + (row - row_min) * 256
                            if temp_x > 256 * mask_width:
                                temp_x = 256 * mask_width
                            if temp_y > 256 * mask_height:
                                temp_y = 256 * mask_height

                            if temp_x < 0:
                                temp_x = 0
                            if temp_y < 0:
                                temp_y = 0

                            # store the pixel coordinates
                            pixel_coords.append((temp_x, temp_y))
            
                mask_whole[(row - row_min) * 256: (row - row_min + 1) * 256, (col - col_min) * 256: (col - col_min + 1) * 256, :] = img_temp
            except:
                continue
            
        # fill the holes in the mask
        if mask_whole.sum() != 0:
            # convert the mask to the image
            im = np.uint8(mask_whole)
            im_zero = Image.new('L', (mask_width*256, mask_height*256), 0)
            draw = ImageDraw.Draw(im_zero)
            for i in range(len(pixel_coords)):
                draw.line([pixel_coords[i], pixel_coords[(i + 1) % len(pixel_coords)]], fill="red", width=5)
            filled_mask = binary_fill_holes(im_zero)

            # save the image
            im = im * filled_mask.reshape(filled_mask.shape[0], filled_mask.shape[1], 1)
            im = Image.fromarray(np.uint8(im))
            img_memory = BytesIO()
            im.save(img_memory, format='PNG')
            img_memory.seek(0)

            regional_img = img_memory
            
            return int(region.index[0]), regional_img
        else:
            return int(region.index[0])
        
        
    else:
        return int(region.index[0])


def concatentate_tiles(region_list, area_shp, cached_tiles, num_proc=10):
    '''
    Concatenate the tiles to obtain all the regional satellite images given the area.
    '''

    # list of region idx
    region_list = sorted(region_list)

    # list of the arguments
    args = [(area_shp[area_shp.index == i], area_shp.loc[area_shp.index == i, "Y_X"].item(), {key: cached_tiles[key] for key in area_shp.loc[area_shp.index == i, "Y_X"].item() if key in cached_tiles}) for i in region_list]

    # concatenate the tiles
    regional_imgs = {}
    fail_region_list = []
        
    # sequential version
    for arg in tqdm(args, desc=" **Concating regional imgs..."):
        result = concat_img_one_region(arg)
        if isinstance(result, tuple):
            idx, img = result
            regional_imgs[idx] = img
        else:
            fail_region_list.append(result)

    # # parallel version
    # pool = Pool(processes=num_proc)
    # with tqdm(total=len(region_list), desc=f" -- Concating regional imgs...") as t:
    #     for result in pool.imap(concat_img_one_region, args):
    #         if isinstance(result, tuple):
    #             idx, img = result
    #             regional_imgs[idx] = img
    #         else:
    #             fail_region_list.append(result)
    #         t.update(1)

    return regional_imgs, fail_region_list


def area_SateImgs(area_shp, token='', num_proc=50):
    '''
    Download the satellite images for the given area.
    From Esri living atlas, the token is required to download the tiles.

    The tiles are downloaded into the given directory.
    And the regional satellite images are obtained by combining the tiles.
    '''
    # check the token
    if token == '':
        raise Exception("Please provide the token via \{set_satetoken\} for downloading the satellite images.")
    
    # coordinate system
    area_shp = area_shp.to_crs(epsg=4326)

    # get the x and y coordinates of the tiles
    area_shp, Y_X = get_YX_area(area_shp)
    
    # download the tiles
    cached_tiles = {}
    remaining = Y_X
    while remaining != []:
        cached_tiles_tmp, fail_tile_list = download_all_tiles(remaining, token, num_proc)
        cached_tiles.update(cached_tiles_tmp)
        print(f" -- {len(fail_tile_list)} tiles failed to download.")
        remaining = fail_tile_list

    # concatenate the tiles to obtain the regional satellite images
    regional_imgs = {}
    remaining = area_shp.index.to_list()
    while remaining != []:
        regional_imgs_tmp, fail_region_list = concatentate_tiles(remaining, area_shp, cached_tiles, num_proc)
        regional_imgs.update(regional_imgs_tmp)
        remaining = fail_region_list

    return regional_imgs


def check_download_RemoteCLIP(model_name):
    '''
    check whether the model exists locally and download it if not.
    '''
    # some path
    path_to_your_checkpoints = 'checkpoints/models--chendelong--RemoteCLIP'
    os.makedirs(path_to_your_checkpoints, exist_ok=True)
    model_path = f"{path_to_your_checkpoints}/RemoteCLIP-{model_name}.pt"

    # download it if not exists
    if os.path.exists(model_path):
        print(f" **The vision model RemoteCLIP-{model_name} is available.")
        return "success"
    else:
        print(f" **Downloading the vision model RemoteCLIP-{model_name}...")
        try:
            # through huggingface
            checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{model_name}.pt", cache_dir='checkpoints')
            shutil.copy(checkpoint_path, model_path)
            print(f' **{model_name} is downloaded to {checkpoint_path} via hf_hub.')
        except:
            # through urllib.request
            url = f"https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-{model_name}.pt"
            request.urlretrieve(url, model_path, schedule)
            print(f' **{model_name} is downloaded to {path_to_your_checkpoints} via https.')
        
        if os.path.exists(model_path):
            return "success"
        else:
            return "failed"
        

def extract_imgfeats_RemoteCLIP(vision_model, model_name, regional_imgs, device): # @param ['RN50', 'ViT-B-32', 'ViT-L-14']
    '''
    model: 'RN50', 'ViT-B-32', 'ViT-L-14' from RemoteCLIP
    '''
    # load the model
    _, _, preprocess = open_clip.create_model_and_transforms(model_name)

    # tensor accept by RemoteCLIP
    def load_and_preprocess_image(image):
        image = Image.open(image)
        image_tensor = preprocess(image).unsqueeze(0) # for batch input
        return image_tensor

    # extract features
    img_feats = []
    for region_idx, img in tqdm(regional_imgs.items(), desc=f" -- Extracting img features..."):
        # extract img features
        with torch.no_grad(), torch.cuda.amp.autocast():
            img_tensor = load_and_preprocess_image(img).to(device)
            img_features = vision_model.encode_image(img_tensor).squeeze().cpu().numpy()
            img_feats.append(img_features)
    img_feats = np.stack(img_feats)

    return img_feats