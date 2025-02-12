import json
import warnings
import os
import sys
import torchvision

from metrics.uciqe import batch_uciqe
from metrics.uiqm import batch_uiqm

from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from config import Config
from data import get_data
from models import create_model
from utils import seed_everything, load_checkpoint
import torch

warnings.filterwarnings('ignore')


def test():
    # Collect command-line override arguments after "--config_yaml config.yml"
    override_list = []
    if len(sys.argv) > 3:
        override_list = sys.argv[3:]
    opt = Config('config.yml', override_list)
    opt.finalize_config()

    accelerator = Accelerator()
    device = accelerator.device

    seed_everything(opt.OPTIM.SEED)
    criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)

    # Data Loader
    val_dir = opt.TESTING.VAL_DIR

    if not val_dir:
        print("No validation directory specified. Skipping dataset loading.")
        val_dataset = []
    else:
        val_dataset = get_data(val_dir, opt.TESTING.INPUT, opt.TESTING.TARGET, 'test', opt.TRAINING.ORI,
                               {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8,
                            drop_last=False, pin_memory=True) if val_dataset else []

    # Model & Metrics
    model = create_model(opt.MODEL.NAME)

    weight_path = opt.TESTING.WEIGHT
    if not weight_path or not os.path.exists(weight_path):
        raise FileNotFoundError(f"Checkpoint file '{weight_path}' not found or invalid path.")

    load_checkpoint(model, None, None, weight_path, device)

    model, testloader = accelerator.prepare(model, testloader)

    model.eval()

    size = len(testloader)

    stat_psnr = 0
    stat_ssim = 0
    stat_lpips = 0
    stat_uciqe = 0
    stat_uiqm = 0

    for _, test_data in enumerate(tqdm(testloader)):
        # get the inputs; data is a list of [targets, inputs, filename]
        inp = test_data[0].contiguous()
        tar = test_data[1]

        with torch.no_grad():
            res = model(inp)

        if not os.path.isdir(opt.TESTING.RESULT_DIR):
            os.makedirs(opt.TESTING.RESULT_DIR)
        torchvision.utils.save_image(res, os.path.join(opt.TESTING.RESULT_DIR, test_data[2][0]))

        stat_psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
        stat_ssim += structural_similarity_index_measure(res, tar, data_range=1).item()
        stat_lpips += criterion_lpips(res, tar).item()
        stat_uciqe += batch_uciqe(res)
        stat_uiqm += batch_uiqm(res)

    stat_psnr /= size
    stat_ssim /= size
    stat_lpips /= size
    stat_uciqe /= size
    stat_uiqm /= size

    test_info = ("Test Result on {}, check point {}, testing data {}".
                 format(opt.MODEL.SESSION, opt.TESTING.WEIGHT, opt.TESTING.VAL_DIR))
    log_stats = ("PSNR: {}, SSIM: {}, LPIPS: {}, UCIQUE: {}, UIQM: {}".
                 format(stat_psnr, stat_ssim, stat_lpips, stat_uciqe, stat_uiqm))
    print(test_info)
    print(log_stats)
    os.makedirs(opt.LOG.LOG_DIR, exist_ok=True)
    with open(os.path.join(opt.TESTING.LOG_DIR, opt.TESTING.LOG_FILE), mode='a', encoding='utf-8') as f:
        f.write(json.dumps(test_info) + '\n')
        f.write(json.dumps(log_stats) + '\n')


if __name__ == '__main__':
    os.makedirs('result', exist_ok=True)
    test()
