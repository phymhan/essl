import torch
import torch.distributed as dist
from classy_vision.generic.distributed_util import (
    convert_to_distributed_tensor,
    convert_to_normal_tensor,
    is_distributed_training_run,
)
import random
from PIL import Image, ImageOps, ImageFilter

import os
import sys
import shutil
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from glob import glob
import torch.nn.functional as F
try:
    import wandb
except ImportError:
    wandb = None


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    """
    Similar to classy_vision.generic.distributed_util.gather_from_all
    except that it does not cut the gradients
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        gathered_tensors = GatherLayer.apply(tensor)
        gathered_tensors = [
            convert_to_normal_tensor(_tensor, orig_device)
            for _tensor in gathered_tensors
        ]
    else:
        gathered_tensors = [tensor]
    gathered_tensor = torch.cat(gathered_tensors, 0)
    return gathered_tensor


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img
    
    def __repr__(self):
        fmt_str = self.__class__.__name__ + '('
        fmt_str += 'p={}'.format(self.p) + ')'
        return fmt_str


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
    
    def __repr__(self):
        fmt_str = self.__class__.__name__ + '('
        fmt_str += 'p={}'.format(self.p) + ')'
        return fmt_str


""" additional utils """
class Config(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class logging_file(object):
    def __init__(self, path, mode='a+', time_stamp=True, **kwargs):
        self.path = path
        self.mode = mode
        if time_stamp:
            # self.path = self.path + '_' + time.strftime('%Y%m%d_%H%M%S')
            # self.write(f'{time.strftime("%Y%m%d_%H%M%S")}\n')
            self.write(f'{datetime.now()}\n')
    
    def write(self, line_to_print):
        with open(self.path, self.mode) as f:
            f.write(line_to_print)

    def flush(self):
        pass

    def close(self):
        pass

    def __del__(self):
        pass


import logging
def log(output, flush=True):
    logging.info(output)
    if flush:
        print(output)


def setup_wandb_run_id(log_dir, resume=False):
    # NOTE: if resume, use the existing wandb run id, otherwise create a new one
    os.makedirs(log_dir, exist_ok=True)
    file_path = Path(log_dir) / 'wandb_run_id.txt'
    if resume:
        assert file_path.exists(), 'wandb_run_id.txt does not exist'
        with open(file_path, 'r') as f:
            run_id = f.readlines()[-1].strip()  # resume from the last run
    else:
        run_id = wandb.util.generate_id()
        with open(file_path, 'a+') as f:
            f.write(run_id + '\n')
    return run_id


def setup_wandb(args):
    # try:
    #     import wandb
    print('Setting up wandb...')
    if wandb is not None:
        name = Path(args.log_dir).name
        resume = getattr(args, 'resume', False)
        run_id = setup_wandb_run_id(args.log_dir, resume)
        args.wandb_run_id = run_id
        run = wandb.init(
            project=args.wandb_project,
            name=name,
            id=run_id,
            config=args,
            resume=True if resume else "allow",
            # settings=wandb.Settings(init_method='fork'),
        )
        return run
    else:
        log_str = "Failed to set up wandb - aborting"
        log(log_str, level="error")
        raise RuntimeError(log_str)


def get_hostname():
    try:
        import socket
        return socket.gethostname()
    except:
        return 'unknown'

def print_args(parser, args, is_dict=False):
    # args = deepcopy(args)
    if not is_dict and hasattr(args, 'parser'):
        delattr(args, 'parser')
    name = getattr(args, 'name', Path(args.log_dir).name if hasattr(args, 'log_dir') else 'unknown')
    datetime_now = datetime.now()
    message = f"Name: {name} Time: {datetime_now}\n"
    message += f"{os.getenv('USER')}@{get_hostname()}:\n"
    if os.getenv('CUDA_VISIBLE_DEVICES'):
        message += f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}\n"
    message += '--------------- Arguments ---------------\n'
    args_vars = args if is_dict else vars(args)
    for k, v in sorted(args_vars.items()):
        comment = ''
        default = None if parser is None else parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '------------------ End ------------------'
    # print(message)  # suppress messages to std out

    # save to the disk
    log_dir = Path(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir / 'src', exist_ok=True)
    file_name = log_dir / 'args.txt'
    with open(file_name, 'a+') as f:
        f.write(message)
        f.write('\n\n')

    # save command to disk
    file_name = log_dir / 'cmd.txt'
    with open(file_name, 'a+') as f:
        f.write(f'Time: {datetime_now}\n')
        if os.getenv('CUDA_VISIBLE_DEVICES'):
            f.write('CUDA_VISIBLE_DEVICES=%s ' % os.getenv('CUDA_VISIBLE_DEVICES'))
        f.write('deepspeed ' if getattr(args, 'deepspeed', False) else 'python3 ')
        f.write(' '.join(sys.argv))
        f.write('\n\n')

    # backup train code
    shutil.copyfile(sys.argv[0], log_dir / 'src' / f'{os.path.basename(sys.argv[0])}.txt')


def get_last_checkpoint(ckpt_dir, ckpt_ext='.pt', latest=None):
    assert ckpt_ext.startswith('.')
    if latest is None:
        ckpt_path = sorted(glob(os.path.join(ckpt_dir, '*'+ckpt_ext)), key=os.path.getmtime, reverse=True)[0]
    else:
        if not latest.endswith(ckpt_ext):
            latest += ckpt_ext
        ckpt_path = Path(ckpt_dir) / latest
    return ckpt_path


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
