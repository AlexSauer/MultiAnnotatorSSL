import time
from utils.common_utils import NoImprovementException, NaNLossException
import logging
log = logging.getLogger(__name__)


def train(model, train_loader, train_loader_semi, val_loader,optimizer, scheduler,
          n_iterations, device, tracker, earlyStopping = None):

    try:
        epochs = n_iterations // len(train_loader)
        log.info(f'Iterations per epoch: {len(train_loader)}')
        log.info(f'Epochs to run: {epochs}')
        log.info(f'Early Stopping: '
                 f'{"Disabled" if earlyStopping is None else str(earlyStopping)} epochs')
        iteration = 0
        for epoch in range(epochs):
            start_time = time.time()
            loss_sum = 0.

            semi_iter = iter(train_loader_semi)
            for step, (patch, masks) in enumerate(train_loader):
                # Sample from the unlabeled data infinitely
                try:
                    # patch_unsuper, _ = next(semi_iter)
                    patch_unsuper = next(semi_iter)  # For true aleatoric experiment
                except StopIteration:
                    # Restart the unlabelled sampling once we are through it
                    semi_iter._reset(train_loader_semi)
                    patch_unsuper, _ = next(semi_iter)

                # Move to GPU
                patch = patch.to(device)
                # patch_unsuper = patch_unsuper.to(device) if patch_unsuper is not None else None
                patch_unsuper = [x.to(device) for x in patch_unsuper]   if patch_unsuper != (None, None) \
                                else (None, None) # For true Aleatoric Experiment
                masks = masks.to(device)

                # Compute loss
                model.train()
                loss = model.comp_loss(patch, patch_unsuper, masks, iteration, tracker, mode = 'train')

                if not loss.isfinite():
                    raise NaNLossException

                # Update parameters
                model.update(loss, optimizer)
                scheduler.step()
                iteration += 1

                loss_sum += loss.detach().item()

            duration = time.time() - start_time
            tracker.flush_buffer()  # Write the mean of the accumulated scalars from model.comp_loss to the SummaryWriter

            model.eval()
            eval_results = tracker.eval_model(model, val_loader, iteration=iteration)

            log.info(f'epoch {epoch:3} took {duration:5.2f} sec, Loss: {loss_sum/len(train_loader):.6f}, '
                     f'Val: {eval_results}')

            if earlyStopping is not None:
                if (tracker.eval_iter - tracker.eval_iterLastImprovement) > earlyStopping:
                    raise NoImprovementException

    # Catch exceptions and log them
    except KeyboardInterrupt:
        logging.info("Training aborted, saving predictions and model...")
    except NoImprovementException:
        logging.info(f"Training aborted because there was no improvement for {earlyStopping} iterations, "
                     f"savings predictions and model ...")
    except NaNLossException:
        logging.info(f"Training aborted because the loss turned into NaN, "
                     f"savings predictions and model ...")