import torch
import numpy as np
import os
import logging
log = logging.getLogger(__name__)


class NoImprovementException(Exception):
    pass

class NaNLossException(Exception):
    pass


def turn_off_randomness(seed):
    # Turn off randomness
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_state(model_path, model, optimizer = None, tracker = None):
    print(f'Loading model {model_path}')

    checkpoint = torch.load(model_path)
    if 'model_state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and 'optimizer_state' in checkpoint.keys():
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    if tracker is not None and 'tracker_iter' in checkpoint.keys():
        tracker.eval_iter = checkpoint['tracker_iter']


def save_state(model_path, model, optimizer=None, tracker=None):
    results = {'model_state_dict': model.state_dict()}
    if optimizer is not None:
        results['optimizer_state_dict'] = optimizer.state_dict()
    if tracker is not None:
        results['tracker_iter'] = tracker.eval_iter

    torch.save(results, model_path)

def configLogger(log, args, log_stdout = True):
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    fHandler = logging.FileHandler(os.path.join(args.path_output, args.run_name, 'loggings.log'))
    log.addHandler(fHandler)
    fHandler.setFormatter(logging.Formatter("%(asctime)s, %(message)s","%Y-%m-%d %H:%M:%S"))
    if log_stdout:
        sHandler = logging.StreamHandler()
        log.addHandler(sHandler)
        sHandler.setFormatter(logging.Formatter("%(asctime)s, %(message)s","%Y-%m-%d %H:%M:%S"))



def check_experiment_setup(base):
    from utils.Args import Args
    args = Args(base)

    # Check whether the output path is in the Experiments directory
    assert 'Experiments' in args.path_output.split('/'), f'The current output path is {args.path_output} which ' \
                                                        f'does not contain Experiments. Is that correct?'
