import logging
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import sys
from datetime import datetime
import shutil
import copy
from utils.common_utils import save_state
from myutils.DL.Metrics import Metrics


class FakeSummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scaler(self, *args, **kwargs):
        pass

    def add_figure(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


def init_writer(log_results, log_dir):
    if log_results:
        return SummaryWriter(log_dir)
    else:
        return FakeSummaryWriter


class Tracker:
    """
    Class to keep track of all relevant hyperparameters and metrics
    To read all descriptions specified in the args.yaml files, navigate to the output directory and type
    find . -name 'description.txt' | xargs grep '^desc.*'
    """

    def __init__(self, path_output, log_results,
                 base_config = None, extension_config = None, args=None):
        self.path_output = path_output
        self.mode = 'debug' if args.debug else 'train'

        # Record start time
        start_time = datetime.now().strftime('%m_%d_%H_%M')
        self.start_time = start_time
        args.start_time = start_time
        self.run_name = args.run_name
        self.set_up_dirs(log_results, path_output, base_config, extension_config, args)
        self.writer = init_writer(log_results, log_dir = os.path.join(path_output, args.run_name))

        # Model prediction
        self.plot_pred_iter = 0

        # Model evaluation
        self.eval_iter = 0

        # Highest values
        self.best_loss = 1e9
        self.max_iou = 0
        self.max_dice = 0
        self.max_acc = 0
        self.path_output = args.path_output
        self.run_name = args.run_name

        # Buffer for writing to summary writer
        self.buffer = {}

        # Early stopping
        self.eval_iterLastImprovement = 0

    def set_up_dirs(self, log_results, path_output, base_config, extension_config, args):
        # Set up directories
        if log_results:
            try:
                os.makedirs(os.path.join(path_output, args.run_name))
            except AttributeError as err:
                logging.info(f"No name for the run given in the args file! \n {err}")
                sys.exit()
            except FileExistsError:
                logging.info("There is already a directory with the same name!"
                             f'{os.path.join(path_output, args.run_name)}')
                if not args.run_name.lower().startswith('debug'):
                    self._move_results_dir(path_output, args.run_name)  # Move old directory and ..
                    os.makedirs(os.path.join(path_output, args.run_name))  # Create new directory
                    logging.info('Moved the old directory into "old_experiments"')
                    # sys.exit()

            # Save current configuration:
            with open(os.path.join(args.path_output, args.run_name, 'description.txt'), 'w') as f:
                description = ""
                description += args.to_string()
                f.write(description)

            # Copy args file if they are present
            if os.path.isfile(base_config):
                shutil.copyfile(base_config, os.path.join(args.path_output, args.run_name, 'args.yaml'))
            # else: #
            #     raise ValueError(f"Baseconfig not a file! {base_config}")
            if extension_config is not None:
                if os.path.isfile(extension_config):
                    shutil.copyfile(extension_config, os.path.join(args.path_output, args.run_name, 'args_extension.yaml'))
                # else:
                #     raise ValueError(f"Extension config not a file! {extension_config}")

        else:
            logging.info("Results are NOT logged!")

    def _move_results_dir(self, path_dir: str, run_name: str) -> None:
        """Appends the current date to the given directory and moves it into 'old_experiments'"""
        path_run = os.path.join(path_dir, run_name)

        # Create 'old_experiments' directory if it doesnt exist yet
        if not os.path.exists(os.path.join(path_dir, 'old_experiments')):
            os.makedirs(os.path.join(path_dir, 'old_experiments'))

        # Rename current run
        creation_date = self._find_creation_date(path_run)
        new_name = path_run.rstrip('/') + '_' + creation_date + '/'
        os.rename(path_run, new_name)

        # Move the directory into 'old_experiments'
        base, run_name = os.path.split(os.path.normpath(new_name))
        shutil.move(new_name, os.path.join(base, 'old_experiments', run_name))

    def _find_creation_date(self, path: str) -> str:
        """Finding the creation date apparently is tricky, so I just use the earlist modification date"""
        dates = []
        for file in os.listdir(path):
            c_path = os.path.join(path, file)
            date = datetime.fromtimestamp(os.path.getmtime(c_path)).strftime('%Y_%m_%d_%H_%M')
            dates.append(date)
        return min(dates)

    def _shorten_name(self, name: str) -> str:
        """Shortens a long run name by taking 5 character substrings between separting underscores"""
        parts = name.split('_')
        base = parts.pop(0)
        shortened_parts = []
        for part in parts:
            try:
                _ = float(part)
                shortened_parts.append(part)
            except ValueError:
                shortened_parts.append(part[-5:])
        return '_'.join([base] + shortened_parts)

    def attach_data_loader(self, data_loader):
        """Must be a supervised data_loader"""
        self.data = data_loader

        # Set up evaluation batch for feature embedding:
        it = iter(data_loader)
        data = next(it)
        self.batch_img, labels = data[0], data[1]
        self.batch_annotation = labels

    def eval_model(self, model, data_loader, save_eval_txt = False, iteration = None,
                   use_best_model: bool = False):
        # Parameter iteration is only used to print in which training iteration the best model was found
        loss = 0
        metrics = Metrics('IoU Dice Accuracy Recall Precision')

        with torch.no_grad():
            if use_best_model:
                model = self.loadBestModel(model)
            model.eval()
            for img, annotation in data_loader:
                img,  annotation = img.to(model.device), annotation.to(model.device)
                pred_mask = model.predict(img)

                # Compute loss which buffers the values to be written to the Summary Writer later
                loss += model.comp_loss(img, semi=None, annotations=annotation, iteration=self.eval_iter,
                                        tracker=self, mode='val')
                gt_target = model.comp_target(annotation)
                metrics.comp_metrics_from_bool((pred_mask > 0.5), gt_target)

        self.flush_buffer()  # Write the loss which was buffered during the compute
        loss = (loss / len(data_loader)).cpu().item()

        # Save metrics for this epoch to the summary writer
        metrics.summarise_metrics()
        cur_iou, cur_dice, cur_acc = metrics.get('iou dice accuracy')
        metrics.add_metric_tensorboard(self.writer, self.eval_iter)

        if not use_best_model:
            self.saveBestModel(model, loss, cur_iou, cur_dice, cur_acc, iteration)

        self.eval_iter += 1
        eval_results = f'Loss: {loss:.6f}, IoU: {cur_iou:.4f}, Dice: {cur_dice:.4f}, Accuracy: {cur_acc:.4f}'

        if save_eval_txt:
            if any('Experiments' in dirname for dirname in self.path_output.split('/')):
                # For Experiments I want to additionally (!) summarize all the test results in one file:
                with open(os.path.join(self.path_output, 'finalTestResult.txt'), 'a') as f:
                    short_name = self._shorten_name(self.run_name)
                    f.write(f'{short_name:45}: {eval_results}, Training for {self.eval_iter} evaluations/epochs '
                            f'Best model from iteration {self.eval_iterLastImprovement}  -  {self.start_time}\n')
            with open(os.path.join(self.path_output, self.run_name, 'finalResult.txt'), 'w') as f:
                f.write(eval_results)

        return eval_results

    def add_scalar(self, tag, value, niter, mode = 'train'):
        # If we are in train mode but the scalar is only supposed to be added for debugging, just skip
        if mode == 'debug' and self.mode == 'train':
            return
        self.writer.add_scalar(tag, value, niter)

    def buffer_scalar(self, tag, value, niter, mode = 'train'):
        """
        I often log the results in the training loop for every iteration but only want to report the values over
        the whole epoch.
        Therefore, I'm just gonna buffer them here in a list and the next method will form the mean value and
        just pick the minimal iteration and then write them out to the SummaryWriter.
        """
        if tag not in self.buffer:
            self.buffer[tag] = {'value': [value], 'niter' : [niter], 'mode' : mode}
        else:
            self.buffer[tag]['value'].append(value)
            self.buffer[tag]['niter'].append(niter)

    def flush_buffer(self):
        for key, value_dict in self.buffer.items():
            mean_value = np.mean(value_dict['value'])
            min_iter = np.min(value_dict['niter'])
            self.add_scalar(key, mean_value, min_iter, mode = value_dict['mode'])

        # Empty the buffer
        self.buffer = {}

    def saveBestModel(self, model, loss, cur_iou, cur_dice, cur_acc, iter):
        # # Save the model if two of these metrics are better then the old best ones
        # better = np.array([cur_iou, cur_dice, cur_acc]) > np.array([self.max_iou, self.max_dice, self.max_acc])
        # better = np.sum(better) >= 2
        if loss < self.best_loss:
            self.best_loss, self.max_iou, self.max_dice, self.max_acc = loss, cur_iou, cur_dice, cur_acc
            save_state(os.path.join(self.path_output, self.run_name, 'BestModel.pt'), model)

            with open(os.path.join(self.path_output, self.run_name, 'bestResult.txt'), 'w') as f:
                f.write(f'IoU: {self.max_iou:.4f}, Dice: {self.max_dice:.4f}, '
                        f'Accuracy: {self.max_acc:.4f} in iteration {iter}')

            self.eval_iterLastImprovement = self.eval_iter

    def loadBestModel(self, model):
        logging.info('Model with the best validation performance loaded')
        path = os.path.join(self.path_output, self.run_name, 'BestModel.pt')
        model = copy.deepcopy(model)
        model.load_state_dict(torch.load(path)['model_state_dict'])
        model.eval()
        return model

    def close(self):
        self.writer.close()

