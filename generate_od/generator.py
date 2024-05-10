import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = "1"
import shutil

from tqdm import tqdm
from urllib import request

import pickle as pkl
import geopandas as gpd

import open_clip
from huggingface_hub import hf_hub_download

from .worldpop import worldpop
from .sateimgs import area_SateImgs, check_download_RemoteCLIP, extract_imgfeats_RemoteCLIP
from .model import *
from .utils import *



class Generator:
    def __init__(self):
        # load models needed for generating
        self.fetch_numproc = None

        self._generate_config()
        self._load_vision_model()
        self._load_od_generation_model()
        self._load_od_scalers()


    def _generate_config(self):
        self.config = {
            "n_indim": 97,
            "e_indim": 2,
            "e_outdim": 1,
            "img_dim": 1026,
            "LaPE_dim": 0,

            "hiddim" : 32,
            "num_head" : 4,
            "num_head_cross" : 1,
            "num_layer" : 4,
            "dropout" : 0,

            "if_imgAugDenosing": 1,
            "if-logscaleEval": 0,
            "if-ImgAttrAug": 0,

            "T": 250,
            "DDIM_T_sample" : 25,
            "sample_times" : 50,
            "DDIM_eta" : 0, 
            "beta_scheduler" : "cosine",

            "norm_type": "layer"
        }
        self.config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def set_device(self, device):
        self.config["device"] = device
        self.od_model.to(device)
        self.vision_model.to(device)


    def set_satetoken(self, token):
        self.sate_token = token


    def set_fetch_numproc(self, numproc):
        self.fetch_numproc = numproc


    def _load_vision_model(self, name="RN50"):
        if check_download_RemoteCLIP(name) == "success":
            self.model_name = name
            self.vision_model, _, _ = open_clip.create_model_and_transforms(name)
            path_to_your_checkpoints = 'checkpoints/models--chendelong--RemoteCLIP'
            ckpt = torch.load(f"{path_to_your_checkpoints}/RemoteCLIP-{name}.pt", map_location=self.config["device"])
            message = self.vision_model.load_state_dict(ckpt)
            self.vision_model.to(self.config["device"])
        else:
            raise Exception("Failed to download vision model.")

    
    def _load_od_generation_model(self):
        self.od_model = Diffusion(self.config).to(self.config["device"])
        path_to_your_checkpoints = 'checkpoints/models--loooffeeeey--DiffODSate'
        os.makedirs(path_to_your_checkpoints, exist_ok=True)
        model_path = f"{path_to_your_checkpoints}/SateOD-diff.pkl"

        # download model if not exists
        if not os.path.exists(model_path):
            print(f" **Downloading the OD generation model SateOD-diff...")
            try:
                # through huggingface
                checkpoint_path = hf_hub_download("loooffeeeey/DiffODSate", "SateOD-diff.pkl", cache_dir='checkpoints')
                shutil.copy(checkpoint_path, model_path)
                print(f' **SateOD-diff is downloaded to {checkpoint_path} via hf_hub.')
            except:
                # through urllib.request
                url = f"https://huggingface.co/loooffeeeey/DiffODSate/resolve/main/SateOD-diff.pkl"
                request.urlretrieve(url, model_path, schedule)
                print(f' **SateOD-diff.pkl is downloaded to {path_to_your_checkpoints} via https.')
        else:
            print(f" **The od generation model SateOD-diff.pkl is available.")
            
        ckpt = torch.load(model_path, map_location=self.config["device"])
        self.od_model.load_state_dict(ckpt)
        self.od_model.to(self.config["device"])


    def _load_od_scalers(self):
        path_to_your_checkpoints = 'checkpoints/models--loooffeeeey--DiffODSate'
        os.makedirs(path_to_your_checkpoints, exist_ok=True)
        model_path = f"{path_to_your_checkpoints}/SateOD-datascalers.pkl"

        # download data scalers if not exists
        if not os.path.exists(model_path):
            print(f" **Downloading data scalers...")
            try:
                # through huggingface
                checkpoint_path = hf_hub_download("loooffeeeey/DiffODSate", "SateOD-datascalers.pkl", cache_dir='checkpoints')
                shutil.copy(checkpoint_path, model_path)
                print(f' **SateOD-datascalers is downloaded to {checkpoint_path} via hf_hub.')
            except:
                # through urllib.request
                url = f"https://huggingface.co/loooffeeeey/DiffODSate/resolve/main/SateOD-datascalers.pkl"
                request.urlretrieve(url, model_path, schedule)
                print(f' **SateOD-datascalers.pkl is downloaded to {path_to_your_checkpoints} via https.')
        else:
            print(f" **The data scaler SateOD-datascalers.pkl are available.")

        with open('checkpoints/models--loooffeeeey--DiffODSate/SateOD-datascalers.pkl', 'rb') as file:
            self.data_scalers = pkl.load(file)


    def load_area(self, area_shp: gpd.GeoDataFrame):
        '''
        area_shp: geopandas.GeoDataFrame
        '''
        self.area = area_shp
        if self.area.crs is None:
            raise Exception("CRS not defined for area.")
        
        # convert to WGS84
        self.area = self.area.to_crs(epsg=4326)


    def _fetch_worldpop(self, area_shp: gpd.GeoDataFrame):
        if self.fetch_numproc:
            worldpop_feats = worldpop(area_shp, num_proc=self.fetch_numproc)
        else:
            worldpop_feats = worldpop(area_shp)
        return worldpop_feats


    def _fetch_sateimgs(self, area_shp: gpd.GeoDataFrame):
        if self.fetch_numproc:
            imgs = area_SateImgs(area_shp, self.sate_token, num_proc=self.fetch_numproc)
        else:
            imgs = area_SateImgs(area_shp, self.sate_token)
        return imgs


    def _compute_distance(self):
        if self.area is None:
            raise Exception("Area not loaded. Load area first.")
        
        # compute distance matrix
        distance = extract_dis_adj_matrix(self.area)
        return distance


    def _construct_inputs(self):
        
        if self.area is None:
            raise Exception("Area not loaded. Load area first.")
        
        # fetch worldpop
        print(" **Fetching pop features from WorldPop...")
        worldpop = self._fetch_worldpop(self.area)
        
        # fetch sate imgs
        print(" **Fetching img features based on Satellite Images from Esri...")
        imgs = self._fetch_sateimgs(self.area)
        img_feats = extract_imgfeats_RemoteCLIP(self.vision_model, self.model_name, imgs, self.config["device"])
        
        nfeat = np.concatenate([img_feats, np.log1p(worldpop)], axis=1)

        # fetch edge features
        distance = self._compute_distance()
        distance = self.data_scalers["dis"].transform(distance.reshape([-1, 1])).reshape([distance.shape[0], distance.shape[1]])

        # construct input to od generation model
        nfeat = torch.FloatTensor(nfeat).to(self.config["device"])
        distance = torch.FloatTensor(distance).to(self.config["device"])
        
        dim = nfeat.shape[0]
        od_placeholder = torch.zeros((dim, dim), dtype=torch.float32).to(self.config["device"])
        batchlization = torch.ones((dim, dim), dtype=torch.float32).to(self.config["device"])
        noiser_factor = 0

        n = nfeat
        n_attr = n[:, 0: self.config["n_indim"]]
        noisy_attr = n_attr * noiser_factor + torch.randn_like(n_attr) * (1 - noiser_factor)
        n_img = n
        pred_attr = noisy_attr * 0
        n = torch.cat([n_img, noisy_attr, pred_attr], dim=1)

        e = od_placeholder

        net = (n, e)

        mask_n = np.ones(list(n.shape))
        mask_e = np.zeros(list(e.shape))

        mask_n = torch.FloatTensor(mask_n).to(self.config["device"])
        mask_e = torch.FloatTensor(mask_e).to(self.config["device"])

        masks = (mask_n, mask_e)

        c = (net, masks, distance, batchlization)

        return c


    def generate(self, sample_times=50):
        if self.area is None:
            raise Exception("Area not loaded. Load area first.")

        with torch.no_grad():
            # construct inputs to od model
            c = self._construct_inputs()
            net, masks, distance, batchlization = c
            n, e = net

            # generate OD matrix
            e_hats = []
            for _ in tqdm(range(sample_times), desc=" **Generating OD matrix"):
                net_hat = self.od_model.DDIM_sample_loop(n.shape, e.shape, c)
                _, e_hat = net_hat
                e_hats.append(e_hat.detach().cpu().numpy())
            e_hat = np.mean(np.stack(e_hats), axis=0)
            od_hat = e_hat
            od_hat = self.data_scalers["od"].inverse_transform(od_hat.reshape([-1, 1])).reshape([od_hat.shape[0], od_hat.shape[1]])
            od_hat = self.data_scalers["od_normer"].inverse_transform(od_hat)
            for i in range(od_hat.shape[0]): # set the diagonal to 0
                od_hat[i,i] = 0
            od_hat[od_hat < 0] = 0
            od_hat = np.floor(od_hat)

            self.od_hat = od_hat

            return od_hat
        
    
    def plot_arc_chart(self):
        """
        Plot the arc chart for the generated OD matrix on map.

        For the specific OD matrix generated on the given area.

        return: f: figure from matplotlib.pyplot.subplot
        """
        f = plot_od_arc_chart(self.od_hat, self.area)
        return f