# Leveraging Inter-Annotator Disagreement For Semi-Supervised Segmentation

This is the code my ISBI submission 'Leveraging Inter-Annotator Disagreement for Semi-Supervised Segmentation'.

## Code structure:
The code is structured in the following way:
- The `main.py` file is the file that loads in all the parameters from the `args.yaml` file using the `/utils/Args` class.
- In that `yaml` file all important decisions are specified like which dataset to use (`LIDC` data vs `Prostate`) and which model to run.
- This flexiblity is achieved by having wrapper function that decide which model to load (`models/build_model.py`) and how to build the dataset (`/data/dataloader.py`)
- Morever, the `main` function calls a `train` function that has the logic for the standard deep learning training loop.
- However, the functionality to compute the loss is specific to each model. Therefore, each model has a `comp_loss` function that returns the loss specific to it.
- All the functionality with respect to logging (like a log file, setting up directories for the output and the Tensorboard writer) is handled by `utils/Tracker`. 
 
 
## Questions?
Please feel free to reach out if you have any questions!