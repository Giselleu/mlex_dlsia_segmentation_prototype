import torch
from tiled.client import from_uri

class TiledDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_tiled_uri,
            mask_uri,
            seg_uri,
            mask_idx,
            data_tiled_api_key=None,
            mask_api_key=None,
            seg_api_key=None,
            shift=0,
            transform=None):
        '''
        Args:
            data_tiled_uri:      str,    Tiled URI of the input data
            mask_uri:            str,    Tiled URI of mask
            seg_uri:             str,    Tiled URI of segmentation results
            mask_idx:            list,   List for slice indexs that maps the mask to corresponding input images  
            data_tiled_api_key:  str,    Tiled API key for input data access
            mask_api_key:        str,    Tiled API key for mask access
            seg_api_key:         str,    Tiled API key for segmentation access
            shift:               int,    Optional shift for pixel values in masks to bring down the class id to start from 0, (and -1 for unlabeled pixles)
            transform:           callable, if not given return PIL image

        Return:
            ml_data:        tuple, (data_tensor, mask_tensor)
        '''
        self.data_client = from_uri(data_tiled_uri, api_key=data_tiled_api_key)
        self.mask_client = from_uri(mask_uri, api_key=mask_api_key)
        self.seg_client = from_uri(seg_uri, api_key=seg_api_key)
        self.mask_idx = list(mask_idx)
        self.shift = int(shift)
        self.transform = transform

    def __len__(self):
        return len(self.mask_client)

    def __getitem__(self, idx):
        data = self.data_client[self.mask_idx[idx],]
        mask = self.mask_client[idx,].astype('int') - self.shift # Conversion to int is needed as switching unlabeled pixels to -1 would cause trouble in uint8 format
        if self.transform:
            return self.transform(data), mask
        else:
            return data, mask