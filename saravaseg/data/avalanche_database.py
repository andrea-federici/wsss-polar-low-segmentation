import os
import numpy as np
import skimage
import cv2
import numbers
from tqdm import tqdm
import logging

from torch.utils.data import Dataset, DataLoader
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import lightning

try: 
    from skreddata import database
except ModuleNotFoundError as e: 
    print('Failed to import skreddata!')
    

# TODO: CONFIGURABLE: 
_ROOT_DIR = os.getenv('DATA_DIR')
if _ROOT_DIR is None: 
    DATA_DIR = None
    MASK_DIR = None
    CACHE_DIR = None
else:  
    DATA_DIR = os.path.join(_ROOT_DIR, 'avalanche_input')
    MASK_DIR = os.path.join(_ROOT_DIR, 'avalanche_masks')
    CACHE_DIR = os.path.join(_ROOT_DIR, 'avalanche_cache')


# -------------------------------- #
#   UTILS
# -------------------------------- #


def _rescale(arr, vmin, vmax, fill_value=None): 
    
    if fill_value is None: 
        fill_value = 0

    # Invalid inputs: 
    invalid = ~np.isfinite(arr)

    # Rescale: 
    if vmin is None: 
        vmin = np.nanmin(arr)
    if vmax is None: 
        vmax = np.nanmax(arr)
        
    if vmin == vmax: 
        arr = np.zeros_like(arr)
    else: 
        arr = (arr - vmin)/(vmax - vmin)
        
    # Fill invalids: 
    if isinstance(fill_value, numbers.Number): 
        arr[invalid] = fill_value
    elif fill_value == 'random': 
        arr[invalid] = np.random.rand(np.sum(invalid))

    # Clip: 
    arr[arr < 0] = 0
    arr[arr > 1] = 1

    return arr


def _get_dem_slope_in_degrees(dem, spacing):
    gx, gy = np.gradient(dem, *spacing)
    return np.rad2deg(np.arctan(np.sqrt(gx**2 + gy**2)))


def _get_dem_aspect_in_degrees(dem, spacing):
    gx, gy = np.gradient(dem, *spacing)
    return np.rad2deg(np.arctan2(-gy, -gx))


def _get_hillshade(dem, spacing, azimuth_angle=None, incidence_angle=None):

    if azimuth_angle is None:
        azimuth_angle = 60
    if incidence_angle is None:
        incidence_angle = 45

    az = np.deg2rad(azimuth_angle)
    inc = np.deg2rad(incidence_angle)
    slp = np.deg2rad(_get_dem_slope_in_degrees(dem, spacing))
    asp = np.deg2rad(_get_dem_aspect_in_degrees(dem, spacing))

    shaded = np.sin(inc) * np.sin(np.pi*.5 - slp) + \
                np.cos(inc) * np.cos(np.pi*.5 - slp) * \
                np.cos(az - np.pi*.5 - asp)
    shaded = (shaded + 1) / 2
    
    return shaded


def _apply_transform(x, y, transform): 
    aug = transform(image=x, mask=y)
    return aug["image"], aug["mask"]


def _crop_pad(arr, shp, fill_value=np.nan): 
    d = arr.shape - np.array(shp)
    a = d//2
    b = d - a
    slc = tuple(slice(i if i>0 else None, -j if j>0 else None) for i,j in  zip(a,b))
    pad = tuple((-i if i<0 else 0, -j if j<0 else 0) for i,j in zip(a,b))
    return np.pad(arr[slc], pad, constant_values=fill_value)
    

def _read_x(folder, uuid):
    
    # Find files: 
    fn_rcs = os.path.join(folder, uuid, f'{uuid}_rcs.tif')
    if not os.path.isfile(fn_rcs): 
        fn_rcs = os.path.join(folder, uuid, f'{uuid}_epsg32633_rcs.tif')
    fn_dem = os.path.join(folder, uuid, f'{uuid}_dem.tif')
    if not os.path.isfile(fn_dem): 
        fn_dem = os.path.join(folder, uuid, f'{uuid}_epsg32633_dem.tif')

    # Read files: 
    arr_rcs = skimage.io.imread(fn_rcs).astype('float32')
    arr_dem = skimage.io.imread(fn_dem).astype('float32')

    # Feature selection: 
    #   rcs time diff: 
    arr_dif = np.stack((
            arr_rcs[:,:,1] - arr_rcs[:,:,0], 
            arr_rcs[:,:,3] - arr_rcs[:,:,2]
        ), axis=2)

    #   combined VV HV rcs time diff: 
    arr_cmb = np.sqrt( arr_dif[:,:,0].clip(0) * arr_dif[:,:,1].clip(0) )

    #   dem slope: 
    arr_slp = _get_dem_slope_in_degrees(arr_dem, (10, 10))
    
    #   dem hillshade: 
    arr_hill = _get_hillshade(arr_dem, (10, 10))

    #   collect: 
    selection = [
        arr_rcs, # [0, 1, 2, 3] -> [hv0, hv1, vv0, vv1]
        arr_dif, # [4, 5]
        arr_cmb, # [6]
        arr_dem, # [7]
        arr_slp, # [8]
        arr_hill, # [9]
    ]
    return np.concatenate([np.atleast_3d(_x) for _x in selection], axis=2)


def _read_y(folder, uuid): 
    fn_msk = os.path.join(folder, f'{uuid}_mask.png')
    arr_msk = skimage.io.imread(fn_msk).astype('?').astype('float32')
    return arr_msk


# -------------------------------- #
#   DATASET
# -------------------------------- #


class AvalancheDataset(Dataset):
    def __init__(self, transform=None, partition=None, mmap_path=None, channels=None, recache=False, limits=None):
        """
        """
            
        self.transform = transform
        self.channels = tuple(channels) if channels is not None else None
        
        if limits is None: 
            limits = ((-25,5),)*4 + ((-10, 10),)*2 + ((0, 10),) + ((0,3000),) + ((0,90),) + ((0,1),)
            if self.channels is not None: 
                limits = tuple([limits[i] for i in self.channels])
        else: 
            limits = tuple([(None, None) if lim is None else lim for lim in limits])
        self.limits = limits
        
        self.data_dir = DATA_DIR
        self.mask_dir = MASK_DIR
        
        # UNGLY TEMP HACK: 
        if 'FILIPPO_HEART_DOCKER' in os.environ and os.environ['FILIPPO_HEART_DOCKER']: 
            import pickle 
            with open('/data/tmp_items.pkl', 'rb') as fp: 
                items = pickle.load(fp)
                items = [database.Item(**itm) for itm in items]
                items0 = [itm for itm in items if itm.label == 0]
                items1 = [itm for itm in items if itm.label == 1]
            # logging.warning('FILIPPO LOVES DOCKER')
        else: 
            db = database.Database('skreddata-mongo')
            items0 = db.get_by_label(0)
            items1 = db.get_by_label(1)
            items = items1 + items0
        
        if partition is None: 
            self.items = items
        elif partition.lower() in ('train', 'training'): 
            self.items = [itm for itm in items if itm.t_1.weekday() in (0,2,3,4,5,6)]
        elif partition.lower() in ('val', 'valid', 'validation'): 
            self.items = [itm for itm in items if itm.t_1.weekday() in (1,)]  # TODO: change for hyperparam search
        elif partition.lower() in ('test'): 
            self.items = [itm for itm in items if itm.t_1.weekday() in (1,)]
        elif partition.lower() in ('display'): 
            z = zip(
                [itm for itm in items0 if itm.t_1.weekday() in (1,)][:16], 
                [itm for itm in items1 if itm.t_1.weekday() in (1,)][:16]
            )
            self.items = [item for sublist in z for item in sublist]
        else: 
            raise Exception(f'argument "partition"={partition} not understood')

        self.length = len(self.items)

        self.mmap_path = mmap_path
        if self.mmap_path is None: 
            self.mmap_x = None
            self.mmap_y = None
        else: 

            height, width = 512, 512
            
            #_transform = alb.Compose([
            #    alb.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0), 
            #    alb.CenterCrop(height, width, p=1.0), 
            #])
            
            os.makedirs(self.mmap_path, exist_ok=True)
            fn_x, fn_y = os.path.join(self.mmap_path, 'x.data'), os.path.join(self.mmap_path, 'y.data')
            x, y = self._read_files(0)
            #x, y = _apply_transform(x, y, _transform)
            x = _crop_pad(x, (height, width, x.shape[2])) 
            y = _crop_pad(y, (height, width)) 
            if os.path.isfile(fn_x) and os.path.isfile(fn_y) and not recache: 
                mode = 'r+'
                self.mmap_x = np.memmap(fn_x, dtype=x.dtype, shape=(self.length, *x.shape), mode=mode)
                self.mmap_y = np.memmap(fn_y, dtype=y.dtype, shape=(self.length, *y.shape), mode=mode)
            else: 
                mode = 'w+'
                self.mmap_x = np.memmap(fn_x, dtype=x.dtype, shape=(self.length, *x.shape), mode=mode)
                self.mmap_y = np.memmap(fn_y, dtype=y.dtype, shape=(self.length, *y.shape), mode=mode)
                for idx in tqdm(range(self.length)): 
                    x, y = self._read_files(idx)
                    #x, y = _apply_transform(x, y, _transform)
                    x = _crop_pad(x, (height, width, x.shape[2])) 
                    y = _crop_pad(y, (height, width)) 
                    self.mmap_x[idx][:] = x[:]
                    self.mmap_y[idx][:] = y[:]

    def _read_files(self, index): 
        itm = self.items[index]
        x = _read_x(self.data_dir, itm.uuid)
        try: 
            y = _read_y(self.mask_dir, itm.uuid)
        except FileNotFoundError as e: 
            if itm.label == 0: 
                logging.debug(f'missing file, but label is zero - assuming zero mask. UUID: {itm.uuid}')
                y = np.zeros(x.shape[0:2])
            else: 
                raise e
        return x, y

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        if self.mmap_path is None: 
            x, y = self._read_files(index)
        else: 
            x = self.mmap_x[index].copy()
            y = self.mmap_y[index].copy()
        
        # Select channels: 
        if self.channels is not None: 
            x = x[..., self.channels].copy()
        
        # Scale and clip x values: 
        assert len(self.limits) == x.shape[-1], \
            f'number of limits ({len(self.limits)}) should be the same as the number of input features ({x.shape[-1]})'
        for i, (_min, _max) in enumerate(self.limits):
            x[...,i] = _rescale(x[...,i], _min, _max)
        
        x[np.isnan(x)] = 0
        y[np.isnan(y)] = 0
        
        # Apply transformation: 
        if self.transform is not None:
            x, y = _apply_transform(x, y, self.transform)
        else: 
            x = np.moveaxis(x, -1, 0)
        
        return x, y


# -------------------------------- #
#   LOADERS
# -------------------------------- #


def get_train_loader(height=512, width=512, augment=True, **kwargs): 
    """
    All possible augmentations at https://github.com/albumentations-team/albumentations#list-of-augmentations
    """
    if augment:
        transform = alb.Compose([
            alb.ShiftScaleRotate(shift_limit=.2, scale_limit=.1, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0), 
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.RandomRotate90(p=0.5),
            alb.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0), 
            alb.RandomCrop(height, width, p=1.0),  
            ToTensorV2(),
        ])
    else:
        transform = alb.Compose([
            alb.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0), 
            alb.CenterCrop(height, width, p=1.0), 
            ToTensorV2(),
        ])
    ds_kws = dict(
        mmap_path=kwargs.pop('mmap_path', None), 
        channels=kwargs.pop('channels', None), 
        recache=kwargs.pop('recache', False)
        )
    ds = AvalancheDataset(partition='train', transform=transform, **ds_kws)
    return DataLoader(ds, **kwargs)


def get_valid_loader(height=512, width=512, **kwargs): 
    transform = alb.Compose([
        alb.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0), 
        alb.CenterCrop(height, width, p=1.0), 
        ToTensorV2(),
    ])
    ds_kws = dict(
        mmap_path=kwargs.pop('mmap_path', None), 
        channels=kwargs.pop('channels', None), 
        recache=kwargs.pop('recache', False)
        )
    ds = AvalancheDataset(partition='valid', transform=transform, **ds_kws)
    return DataLoader(ds, **kwargs)


def get_test_loader(height=512, width=512, **kwargs): 
    transform = alb.Compose([
        alb.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0), 
        alb.CenterCrop(height, width, p=1.0), 
        ToTensorV2(),
    ])
    ds_kws = dict(
        mmap_path=kwargs.pop('mmap_path', None), 
        channels=kwargs.pop('channels', None), 
        recache=kwargs.pop('recache', False)
        )
    ds = AvalancheDataset(partition='test', transform=transform, **ds_kws)
    return DataLoader(ds, **kwargs)


def get_disp_loader(height=512, width=512, **kwargs): 
    transform = alb.Compose([
        alb.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0), 
        alb.CenterCrop(height, width, p=1.0), 
        ToTensorV2(),
    ])
    ds_kws = dict(
        mmap_path=kwargs.pop('mmap_path', None), 
        channels=kwargs.pop('channels', None), 
        recache=kwargs.pop('recache', False)
        )
    ds = AvalancheDataset(partition='disp', transform=transform, **ds_kws)
    return DataLoader(ds, **kwargs)


class AvalancheDataModule(lightning.LightningDataModule):
    def __init__(self, cache_dir, 
                 height: int = 400, 
                 width: int = 400, 
                 batch_size: int = 16, 
                 num_workers: int = 0, 
                 channels=None, 
                 recache=False, 
                 limits=None,
                 augment: bool = True,
                 ):
        super().__init__()
        self.cache_dir = cache_dir
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.channels = channels
        self.num_workers = num_workers
        self.recache = recache
        self.limits = limits
        self.augment = augment

    def train_dataloader(self):
        kws = dict(
            mmap_path=os.path.join(self.cache_dir, 'train'), 
            channels=self.channels, recache=self.recache, batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True, 
            pin_memory=True
        )
        return get_train_loader(self.height, self.width, self.augment, **kws)
    
    def val_dataloader(self):
        kws = dict(
            mmap_path=os.path.join(self.cache_dir, 'valid'), 
            channels=self.channels, 
            recache=self.recache, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False, 
            pin_memory=True
        )
        return get_valid_loader(self.height, self.width, **kws)

    def test_dataloader(self):
        kws = dict(
            mmap_path=os.path.join(self.cache_dir, 'test'), 
            channels=self.channels, 
            recache=self.recache, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False, 
            pin_memory=True
        )
        return get_test_loader(self.height, self.width, **kws)

    def predict_dataloader(self):
        kws = dict(
            mmap_path=os.path.join(self.cache_dir, 'pred'), 
            channels=self.channels, 
            recache=self.recache, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False, 
            pin_memory=True
        )
        return get_disp_loader(self.height, self.width, **kws)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...

