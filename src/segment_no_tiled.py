import time
start = time.time()
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
from torch.distributed import ReduceOp
from torchvision.utils import save_image as tensor_save_image

from network import baggin_smsnet_ensemble, load_network
from parameters import (
    # IOParameters,
    MSDNetParameters,
    SMSNetEnsembleParameters,
    TUNet3PlusParameters,
    TUNetParameters,
)
from seg_utils_mp import crop_seg_save
# from tiled_dataset import TiledDataset
# from utils import allocate_array_space

end = time.time()
load_module_time = end - start

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def listdir_nohidden_no_dashone(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            if '-' not in f.split('.')[0].split('_')[-1]:
                yield f
            
def sort_support_fun(n):
    return int(n.split('.')[0].split('_')[-1])

if __name__ == "__main__":
    
    start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
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
    
    if device != "cpu":
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda:%d'%local_rank)
    # print(f"Inference will be processed on: {device}")
    
    # wait for all gpus to be initialized
    if is_distributed:
        torch.distributed.barrier()
    
    end = time.time()
    torch_dist_init_time = end - start
    
    if is_distributed:
        torch_dist_init_time = torch.tensor(torch_dist_init_time).to(device)
        load_module_time = torch.tensor(load_module_time).to(device)
    
    if world_rank == 0:
        print(f"number of gpus is %i" %world_size)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, help="path of yaml file for parameters")
    parser.add_argument("--slurm_run_name", type=str, help="slurm_run_identifier")
    parser.add_argument("--model_config", type=str, help="model configuration")
    parser.add_argument("--inference_batch_size", type=int, help="inference batch size")
    parser.add_argument("--qlty_window_size", type=int, help="qlty window size")
    parser.add_argument("--qlty_step_size", type=int, help="qlty step size")
    parser.add_argument("-save_results", action="store_true", help="whether or not save inference tiffs")
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
    raw_parameters["batch_size_inference"] = args.inference_batch_size
    raw_parameters["qlty_window"] = args.qlty_window_size
    raw_parameters["qlty_step"] = args.qlty_step_size

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

    if world_rank == 0:
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
    
    start = time.time()
    # image_directory = "/pscratch/sd/s/shizhaou/projects/2024-Tanny-sand-images/Sand data_Tanny_07_03_2024/"
    # image_directory = "/global/cfs/cdirs/als/nist-sand/microct-als/rec20240425_161650_nist-sand-30-200-mix_27keV_z8mm_n2625/"
    
    # image_directory = "/pscratch/sd/s/shizhaou/projects/2024-Tanny-sand-images/Sand_data_000%i/" %world_rank
    
    # image_list = list(listdir_nohidden_no_dashone(image_directory))
    # image_list = sorted(image_list, key=sort_support_fun)
    
    # dataset = np.zeros((len(image_list), 2560, 2560))
    # dataset = np.zeros((125, 2560, 2560))
    
    # for i in range(len(image_list)):
    #     dataset[i] = tifffile.imread(image_directory + image_list[i])

    # for i in range(125):
    #     dataset[i] = tifffile.imread(image_directory + image_list[i])

#     for i in range(1,8):
#         dataset[125 * i: 125 * (i + 1)] = np.copy(dataset[:125])

    # if world_rank == 0:
    #     np.save("/pscratch/sd/s/shizhaou/projects/2024-Tanny-sand-images/2160_frame_sand_data.npy", dataset)
    
    dataset = np.load("/pscratch/sd/s/shizhaou/projects/2024-Tanny-sand-images/2160_frame_sand_data.npy", mmap_mode='c')
    
    qlty_inference = NCYXQuilt(
        X=dataset.shape[-1],
        Y=dataset.shape[-2],
        window=(model_parameters.qlty_window, model_parameters.qlty_window),
        step=(model_parameters.qlty_step, model_parameters.qlty_step),
        border=(model_parameters.qlty_border, model_parameters.qlty_border),
        border_weight=0.2,
    )
    
    local_data_size = dataset.shape[0] / world_size
    # local_data_size = dataset.shape[0]
    
    if is_distributed:
        data_sampler = DistributedSampler(dataset, shuffle=False) if is_distributed else None
        dataloader = DataLoader(dataset,
                                 shuffle=False,
                                 batch_size=1,
                                 sampler=data_sampler,
                                 num_workers=1,
                                 pin_memory=True,
                                 persistent_workers=True,
                               )
    else:
        dataloader = DataLoader(dataset,
                                 shuffle=False,
                                 batch_size=1,
                                 num_workers=1,
                                 pin_memory=True,
                                 persistent_workers=True,
                               )

    # model_dir = os.path.join(io_parameters.models_dir, io_parameters.uid_retrieve)
    model_dir = '/pscratch/sd/s/shizhaou/projects/mlex_dlsia_segmentation_prototype/models/%s.pt' %args.model_config
    
    results_dir = "/pscratch/sd/s/shizhaou/projects/mlex_dlsia_segmentation_prototype/inference_results/" + args.slurm_run_name
    if args.save_results == True and world_rank == 0:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    # Load Network
    if network == "SMSNetEnsemble":
        net = baggin_smsnet_ensemble(model_dir)
    else:
        # net_files = glob.glob(os.path.join(model_dir, "*.pt"))
        # net = load_network(network, net_files[0])
        net = load_network(network, model_dir)
    
    net.eval().to(device)
    
    end = time.time()
    
    load_data_model_time = end - start
    if is_distributed:
        load_data_model_time = torch.tensor(load_data_model_time).to(device)

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
    
    start = time.time()
    # for batch in range(dataset.shape[0]):
    for batch in dataloader:
        if world_rank == 0:
            if local_frame_count == 5:
                torch.cuda.profiler.start()
            if local_frame_count == 16:
                torch.cuda.profiler.stop()
        
        torch.cuda.nvtx.range_push(f"step {local_frame_count}")
        seg_result = crop_seg_save(
            net=net,
            device=device,
            image=batch[0],
            # image=dataset[batch],
            qlty_object=qlty_inference,
            parameters=model_parameters,
            # tiled_client=seg_client,
            frame_idx=world_rank * local_data_size + local_frame_count,
        )
        torch.cuda.nvtx.range_pop()
        
        # save segmentation result per frame
        if args.save_results == True and world_rank == 0:
            torch.cuda.nvtx.range_push(f"save_seg_result")
            tifffile.imwrite(results_dir + "/seg_result_%d.tiff" % local_frame_count, seg_result)
            # tensor_save_image(seg_result, results_dir + "/seg_result_%d.png" % local_frame_count, format='png')
            torch.cuda.nvtx.range_pop()
        local_frame_count += 1
        
    end = time.time()
    
    inference_time = end - start
    if is_distributed:
        inference_time = torch.tensor(inference_time).to(device)
    
    if is_distributed:
        torch.distributed.barrier()
        torch.distributed.all_reduce(load_module_time, op=ReduceOp.AVG)
        torch.distributed.all_reduce(torch_dist_init_time, op=ReduceOp.AVG)
        torch.distributed.all_reduce(load_data_model_time, op=ReduceOp.AVG)
        torch.distributed.all_reduce(inference_time, op=ReduceOp.AVG)
    
    if world_rank == 0 and is_distributed:
        print("Module loading completed in %1.2fs." %load_module_time.cpu())
        print("Torch distributed initilization completed in %1.2fs." %torch_dist_init_time.cpu())
        print("Loading data and trained model completed in %1.2fs." %load_data_model_time.cpu())
        print("Segmentation completed in %1.2fs." %inference_time.cpu())

    if world_rank == 0 and is_distributed == False:
        print("Module loading completed in %1.2fs." %load_module_time)
        print("Torch distributed initilization completed in %1.2fs." %torch_dist_init_time)
        print("Loading data and trained model completed in %1.2fs." %load_data_model_time)
        print("Segmentation completed in %1.2fs." %inference_time)

