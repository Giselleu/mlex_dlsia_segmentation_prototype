import argparse
import glob
import os
import numpy as np

import torch
import yaml
from qlty.qlty2D import NCYXQuilt
# from tiled.client import from_uri
from torchvision import transforms
import tifffile
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from network import baggin_smsnet_ensemble, load_network
from parameters import (
    # IOParameters,
    MSDNetParameters,
    SMSNetEnsembleParameters,
    TUNet3PlusParameters,
    TUNetParameters,
)
from seg_utils import crop_seg_save
# from tiled_dataset import TiledDataset
# from utils import allocate_array_space

def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed%(2**32 - 1))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="path of yaml file for parameters")
    args = parser.parse_args()

    # Open the YAML file for all parameters
    with open(args.yaml_path, "r") as file:
        # Load parameters
        parameters = yaml.safe_load(file)

    # Validate and load I/O related parameters
    # io_parameters = parameters["io_parameters"]
    # io_parameters = IOParameters(**io_parameters)

    # Detect which model we have, then load corresponding parameters
    raw_parameters = parameters["model_parameters"]
    network = raw_parameters["network"]

    model_parameters = None
    if network == "MSDNet":
        model_parameters = MSDNetParameters(**raw_parameters)
    elif network == "TUNet":
        model_parameters = TUNetParameters(**raw_parameters)
    elif network == "TUNet3+":
        model_parameters = TUNet3PlusParameters(**raw_parameters)
    elif network == "SMSNetEnsemble":
        model_parameters = SMSNetEnsembleParameters(**raw_parameters)

    assert model_parameters, f"Received Unsupported Network: {network}"

    print("Parameters loaded successfully.")

    # data_tiled_client = from_uri(
    #     io_parameters.data_tiled_uri, api_key=io_parameters.data_tiled_api_key
    # )
    # mask_tiled_client = None
    # if io_parameters.mask_tiled_uri:
    #     mask_tiled_client = from_uri(
    #         io_parameters.mask_tiled_uri, api_key=io_parameters.mask_tiled_api_key
    #     )
    # dataset = TiledDataset(
    #     data_tiled_client,
    #     mask_tiled_client=mask_tiled_client,
    #     is_training=False,
    #     using_qlty=False,
    #     qlty_window=model_parameters.qlty_window,
    #     qlty_step=model_parameters.qlty_step,
    #     qlty_border=model_parameters.qlty_border,
    #     transform=transforms.ToTensor(),
    # )
    
    is_distributed = False
    if 'WORLD_SIZE' in os.environ:
        is_distributed = int(os.environ['WORLD_SIZE']) > 1
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        world_size = 1
    
    world_rank = 0
    local_rank = 0
    if is_distributed:
        torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
        world_rank = torch.distributed.get_rank()
        local_rank = int(os.environ['LOCAL_RANK'])
    
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:%d'%local_rank)
    print(f"Inference will be processed on: {device}")
    
    image_directory = "/pscratch/sd/s/shizhaou/projects/2024-Tanny-sand-images/Sand data_Tanny_07_03_2024/"
    image_list = os.listdir(image_directory)
    dataset = np.zeros((125, 2560, 2560))
    for i in range(125):
        dataset[i] = tifffile.imread(image_directory + image_list[i])

    qlty_inference = NCYXQuilt(
        X=dataset.shape[-1],
        Y=dataset.shape[-2],
        window=(model_parameters.qlty_window, model_parameters.qlty_window),
        step=(model_parameters.qlty_step, model_parameters.qlty_step),
        border=(model_parameters.qlty_border, model_parameters.qlty_border),
        border_weight=0.2,
    )
    
    local_data_size = dataset.shape[0] / world_size
    
    
    
    if is_distributed:
        data_sampler = DistributedSampler(dataset, shuffle=False)
        dataloader = DataLoader(dataset,
                                 shuffle=False,
                                 batch_size=1,
                                 sampler=data_sampler,
                                 num_workers=1,
                                 worker_init_fn=worker_init,
                                 persistent_workers=True,
                                 pin_memory=torch.cuda.is_available())
    else:
        dataloader = DataLoader(dataset,
                                 shuffle=False,
                                 batch_size=1,
                                 num_workers=1,
                                 worker_init_fn=worker_init,
                                 persistent_workers=True,
                                 pin_memory=torch.cuda.is_available())

    # model_dir = os.path.join(io_parameters.models_dir, io_parameters.uid_retrieve)
    model_dir = '/pscratch/sd/s/shizhaou/projects/2024-Tanny-sand-images/trained_TUNet.pt'

    # Load Network
    if network == "SMSNetEnsemble":
        net = baggin_smsnet_ensemble(model_dir)
    else:
        # net_files = glob.glob(os.path.join(model_dir, "*.pt"))
        # net = load_network(network, net_files[0])
        net = load_network(network, model_dir)

    # Allocate Result space in Tiled
    # seg_client = allocate_array_space(
    #     tiled_dataset=dataset,
    #     seg_tiled_uri=io_parameters.seg_tiled_uri,
    #     seg_tiled_api_key=io_parameters.seg_tiled_api_key,
    #     uid=io_parameters.uid_save,
    #     model=network,
    #     array_name="seg_result",
    # )
    # print(
    #     f"Result space allocated in Tiled and segmentation will be saved in {seg_client.uri}."
    # )
    
    local_frame_count = 0

    for batch in dataloader:
        seg_result = crop_seg_save(
            net=net,
            device=device,
            image=batch[0],
            qlty_object=qlty_inference,
            parameters=model_parameters,
            # tiled_client=seg_client,
            frame_idx=world_rank * local_data_size + local_frame_count,
        )
        local_frame_count += 1
    print("Segmentation completed.")

