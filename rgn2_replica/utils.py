# Author: Eric Alcaide ( @hypnopump ) 
import random
import math
import torch
import numpy as np
import urllib
from einops import repeat, rearrange
# random hacks - device utils for pyTorch - saves transfers
to_cpu = lambda x: x.cpu() if x.is_cuda else x
to_device = lambda x, device: x.to(device) if x.device != device else x

# system-wide utility functions
def get_prot(dataloader_=None, vocab_=None, min_len=80, max_len=150, 
             verbose=True, subset="train", xray_filter=False, full_mask=True):
    """ Gets a protein from sidechainnet and returns
        the right attrs for training. 
        Inputs: 
        * dataloader_: sidechainnet iterator over dataset
        * vocab_: sidechainnet VOCAB class
        * min_len: int. minimum sequence length
        * max_len: int. maximum sequence length
        * verbose: bool. verbosity level
        * subset: str. which subset to load proteins from. 
        * xray_filter: bool. whether to return only xray structures.
        * mask_tol: bool or int. bool: whether to return seqs with unknown coords.
                    int: number of minimum label positions
        Outputs: (cleaned, without padding)
        (seq_str, int_seq, coords, angles, padding_seq, mask, pid)
    """
    if xray_filter: 
        with urllib.request.urlopen("https://gist.githubusercontent.com/hypnopump/a05772052b823d7cb9fde0949d558589/raw/948c35c221fef096eb9760d532675888722478d6/xray_pdb_codes.txt") as f: 
            lines = f.read().decode().split("\n")
        xray_ids = set([line.replace("\n", "") for line in lines])

    while True:
        for b,batch in enumerate(dataloader_[subset]):
            for i in range(batch.int_seqs.shape[0]):
                # skip not xray
                if xray_filter: 
                    if batch.pids[i].split("#")[-1][:4].upper() not in xray_ids: 
                        continue

                # skip too short
                if batch.int_seqs[i].shape[0] < min_len:
                    continue

                # strip padding - matching angles to string means
                # only accepting prots with no missing residues (mask is 0)
                padding_seq = (batch.int_seqs[i] == 20).sum().item()
                padding_mask = -(batch.msks[i] - 1).sum().item() # find 0s

                if (full_mask and padding_seq == padding_mask) or \
                    (full_mask is not True and batch.int_seqs[i].shape[0] - full_mask > 0):
                    # check for appropiate length
                    real_len = batch.int_seqs[i].shape[0] - padding_seq
                    if max_len >= real_len >= min_len:
                        # strip padding tokens
                        seq = batch.str_seqs[i] # seq is already unpadded - see README at scn repo 
                        int_seq = batch.int_seqs[i][:-padding_seq or None]
                        angles  = batch.angs[i][:-padding_seq or None]
                        mask    = batch.msks[i][:-padding_seq or None]
                        coords  = batch.crds[i][:-padding_seq*14 or None]

                        if verbose:
                            print("stopping at sequence of length", real_len)
                        
                        yield seq, int_seq, coords, angles, padding_seq, mask, batch.pids[i]
                    else:
                        if verbose:
                            print("found a seq of length:", batch.int_seqs[i].shape,
                                  "but oustide the threshold:", min_len, max_len)
                else:
                    if verbose:
                        print("paddings not matching", padding_seq, padding_mask)
                    pass
    return None
    
def expand_dims_to(t, length):
    """ Expands up to N dimensions. Different from AF2 (inspo drawn):
    	* Only works for torch Tensors
    	* Expands to `t`, NOT `adds t dims`
        https://github.com/lucidrains/alphafold2/blob/main/alphafold2_pytorch/utils.py#L63
        Ex: 
        >>> expand_dims_to( torch.eye(8), length = 3) # (1, 8, 8)
        >>> expand_dims_to( torch.eye(8), length = 1) # (8, 8)
    """
    if not length - len(t.shape) > 0:
        return t
    return t.reshape(*((1,) * length - len(t.shape)), *t.shape) 


def set_seed(seed, verbose=False): 
    try: random.seed(seed)
    except: "Could not set `random` module seed"

    try: np.random.seed(seed)
    except: "Could not set `np.random` module seed"

    try: 
        torch.manual_seed(seed)
        if torch.cuda.is_available(): 
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except:"Could not set `torch.manual_seed` module seed"

    
    
    if verbose: 
        print("Seet seed to {0}".format(seed))






