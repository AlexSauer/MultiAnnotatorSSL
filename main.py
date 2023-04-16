"""
Project idea:
"""
import os
import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s, %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')

from torch import optim
from train import train
from utils import Args
from utils.Tracker import Tracker
from data.Transforms import Augmenter
from data.dataloader import build_dataloader
from models.build_model import build_model
from utils.common_utils import turn_off_randomness, load_state, save_state, configLogger


def main(base_config = 'args.yaml', extension_config = None):
    # Load arguments
    # print(f"Loading: {base_config}\n {extension_config}")
    args = Args.Args(base_config, extension_config)
    tracker = Tracker(args.path_output, args.log_results, base_config, extension_config, args)
    turn_off_randomness(seed = args.seed)

    log = logging.getLogger()
    configLogger(log, args, log_stdout=args.log_stdout)
    logging.info(f'Running:  {args.run_name}')

    # Load data
    transforms = Augmenter(args.transforms)
    log.info(f'Loading in dataset {args.dataset}...')
    train_loader, train_loader_semi = build_dataloader(args.dataset, args.path_super, args.batch_size,
                                                      transforms=transforms, percSuper=args.percSuper,
                                                      ignoreSemi=args.ignoreSemi, nSamples= args.debug_n_samples)
    logging.info('Finished reading in data reader for training!')

    val_loader = build_dataloader(args.dataset, args.path_val, args.batch_size,
                                  transforms=None, shuffle=True, nSamples= args.debug_n_samples)
    tracker.attach_data_loader(val_loader)
    logging.info('Finished reading in supervised data reader for validation!')

    test_loader = build_dataloader(args.dataset, args.path_test, args.batch_size,
                                   transforms=None, shuffle=False, nSamples= args.debug_n_samples)
    logging.info('Finished reading in test data!')

    # Load model
    model = build_model(args)
    model.to(args.device)
    logging.info(f'{args.model_name} loaded!')

    # Training
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.sched_step, gamma = args.sched_gamma)

    logging.info('Start training!')
    train(model,
          train_loader, train_loader_semi, val_loader,
          optimizer, scheduler,
          args.n_iterations, args.device,
          tracker, args.early_stopping)

    # Eval
    logging.info('Final evaluation: ')
    test_results = tracker.eval_model(model, test_loader, save_eval_txt=True,
                                      use_best_model=True)
    logging.info(f'Test results: {test_results}')
    # tracker.plot_pred_dataset(model, test_loader, args.path_output)

    # Save results
    if args.save_model:
        save_state(os.path.join(args.path_output, args.run_name,  'model.pt'),
                   model, optimizer, tracker)

    tracker.close()
    logging.info('Finished!')


if __name__ == '__main__':
    script_path = os.getcwd()
    main(base_config=os.path.join(script_path, 'args.yaml'))