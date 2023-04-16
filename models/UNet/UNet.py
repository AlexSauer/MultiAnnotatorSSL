import os

import torch
from torch import nn
import numpy as np
import segmentation_models_pytorch as smp
from myutils import DL as myutils
from typing import Optional, Union, List
from utils.common_utils import load_state


class BaseUNet(nn.Module):
    """
    Wrapper class for a standard UNet which implements
        - comp_loss(patch: torch.Tensor, semi: Optional[torch.Tensor] = None, annotations: torch.Tensor = None,
                  iteration: int = 0, tracker = None, mode = 'train'): Function to calculate the current loss
        - update(loss: torch.Tensor, optimizer: torch.optim.Optimizer): Update parameters
        - forward(x: torch.Tensor): Perform forward pass
        - predict(x: torch.Tensor): Perform prediction
    """
    def __init__(self, input_channels: int = 1, device: Union[torch.device, str] = 'cpu') -> None:
        super().__init__()
        self.input_channels = input_channels
        self.device = device

        self.pretrained = False

        if self.pretrained:
            self.UNet = smp.Unet(
                encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=input_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                # 'aux_params'= {'dropout': 0.5},  # This dropout is only applied to a classificatino head
                classes = 1,  # model output channels (number of classes in my dataset)
                activation=None,
            )
        else:
            self.UNet = myutils.UNet(encoder_channels=[input_channels, 64, 128, 256, 512],
                                     decoder_channels=[512, 256, 128, 64, 32, 1],
                                     type='2D',
                                     interpolate=True,
                                     dropout=False)

        self.loss = nn.BCEWithLogitsLoss(reduction='mean')

    def supervised_loss(self, patch: torch.Tensor, annotations: torch.Tensor) -> torch.Tensor:
        target = self.comp_target(annotations)
        pred = self.UNet(patch)
        return self.loss(pred, target)

    def comp_loss(self, patch: torch.Tensor, semi: Optional[torch.Tensor] = None, annotations: torch.Tensor = None,
                  iteration: int = 0, tracker = None, mode = 'train') -> torch.Tensor:
        """ Mode is used in the tracker as an additional label"""
        loss = self.supervised_loss(patch, annotations)
        tracker.buffer_scalar(f"Loss ({mode})", loss.detach().cpu().item(), iteration, mode)
        return loss

    def update(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Unet(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.UNet(x)
        return torch.sigmoid(pred)

    def get_model(self) -> nn.Module:
        if self.pretrained:
            return smp.Unet(
                encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=self.input_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                # 'aux_params'= {'dropout': 0.5},  # This dropout is only applied to a classificatino head
                classes = 1,  # model output channels (number of classes in my dataset)
                activation=None,
            )
        else:
            return myutils.UNet(encoder_channels=[self.input_channels, 64, 128, 256, 512],
                             decoder_channels=[512, 256, 128, 64, 32, 1],
                             type='2D',
                             interpolate=True,
                             dropout=False)

    @staticmethod
    def majority_vote(masks):
        # If we have multiple annotators return the majority vote
        # Otherwise just return the input
        if len(masks.shape) == 4 and masks.shape[1] > 1:
            total_votes = torch.sum(masks, dim = 1, keepdim=True)
            return (total_votes >= masks.shape[1]//2 + 1).float()
        else:
            return masks

    def comp_target(self, annotations):
        return BaseUNet.majority_vote(annotations)

    @staticmethod
    def sigmoid_rampup(current, length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if length == 0:
            return 0
        else:
            current = np.clip(current, 0, length)
            phase = 1.0 - current / length
            return float(np.exp(-5.0 * phase * phase))


class UncertaintyDetectionNetwork(BaseUNet):
    """
    Instead of predicting the majority vote of the annotators, we try to predict
    the areas where annotators disagree
    """
    def __init__(self, input_channels: int = 1, uncertainty_threshold: float = 0.45,
                 device: Union[torch.device, str] = 'cpu'):
        '''
        0.45 is the threshold if for uneven annotators, half +- 1 agree
        '''
        super().__init__(input_channels=input_channels, device=device)
        self.uncertainty_threshold = uncertainty_threshold

    def comp_target(self, annotations):
        if annotations.shape[1] > 2:
            std_error = torch.std(annotations, dim=1, keepdim=True) * 2  # to normalise to [0,1]
            uncertainty_mask = (std_error >= self.uncertainty_threshold).float()
        else:
            uncertainty_mask = ((annotations[:, 0] != annotations[:, 1])[:, None]).float()
        return uncertainty_mask


class NaivePseudoLabel(BaseUNet):
    """
    Naive Pseudo Labelling startegy in which we just make predictions on the unlabelled data
    and try to match them in training
    """
    def __init__(self, input_channels: int = 1, iterationSemiBuildup: float = 1e5, semiLambda: float = 0.01,
                 tau: List[float] = [0.1, 0.9], device: Union[torch.device, str] = 'cpu'):
        super().__init__(input_channels, device)
        # Number of iterations until semi-supervised is completely included
        self.iterationSemiBuildup = iterationSemiBuildup
        self.semiLambda = semiLambda
        # Confidence threshold
        self.tau_negative, self.tau_positive = tau
        assert self.tau_negative <= self.tau_positive, "Tau negative should be smaller than tau positive!"

        self.semi_loss = nn.BCEWithLogitsLoss(reduction='none')

    def comp_semi_mask(self, pseudo_pred_prob, semi):
        return torch.logical_or(pseudo_pred_prob > self.tau_positive,
                                pseudo_pred_prob < self.tau_negative)

    def comp_loss(self, patch: torch.Tensor, semi: Optional[torch.Tensor] = None, annotations: torch.Tensor = None,
                  iteration: int = 0, tracker = None, mode = 'train') -> torch.Tensor:

        # Generate and fit pseudo labels (not present in evaluation)
        if semi is not None:
            if len(semi) == 2:
                semi, semi_gt = semi
            else:
                semi_gt = None
            pseudo_pred = self.UNet(semi)
            pseudo_pred_prob = torch.sigmoid(pseudo_pred)
            pseudo_labels = (pseudo_pred_prob > 0.5).detach().clone().float()
            semi_mask = self.comp_semi_mask(pseudo_pred_prob, semi)

            # Calculate the semi supervised loss and adjust for imbalance
            loss_semi = self.semi_loss(pseudo_pred[semi_mask], pseudo_labels[semi_mask])
            assert len(loss_semi) != 0, 'Unlabelled data completley masked!'

            per_pos_labels_gt = annotations.mean()
            per_pos_labels_pred = pseudo_labels[semi_mask].mean()
            pos_predictions = pseudo_labels[semi_mask] == 1
            loss_semi[pos_predictions] = (per_pos_labels_gt / per_pos_labels_pred) * loss_semi[pos_predictions]
            loss_semi[torch.logical_not(pos_predictions)] = ((1-per_pos_labels_gt) / (1-per_pos_labels_pred)) * \
                                                            loss_semi[torch.logical_not(pos_predictions)]
            loss_semi = loss_semi.mean()

            smooth_in_parameter = self.sigmoid_rampup(iteration, self.iterationSemiBuildup)
            tracker.buffer_scalar(f'Uncertainty mask %',
                                  (semi_mask.sum() / semi_mask.numel()).detach().cpu().item(),
                                  iteration, mode)
            tracker.buffer_scalar('Semi Ratio Positive predictions',
                                  (per_pos_labels_gt / per_pos_labels_pred).detach().cpu().item(),
                                  iteration, mode)
            tracker.buffer_scalar('Semi Precision',
                                  myutils.Metrics.comp_precision_bool(pseudo_labels[semi_mask] > 0.5,
                                                self.comp_target(semi_gt)[semi_mask] > 0.5).detach().cpu().item(),
                                  iteration, mode)

        else:
            loss_semi = torch.Tensor([0]).to(patch.device)
            smooth_in_parameter = 0

        # Supervised part
        loss_supervised = self.supervised_loss(patch, annotations)

        # Loss
        loss = loss_supervised + loss_semi * smooth_in_parameter * self.semiLambda
        tracker.buffer_scalar(f"Loss ({mode})", loss.detach().cpu().item(), iteration, mode)
        if semi is not None:
            tracker.buffer_scalar(f"Loss scaled semi",
                                  loss_semi.detach().cpu().item() * smooth_in_parameter,
                                  iteration, mode)

        return loss


class MCPseudoLabel(NaivePseudoLabel):
    def __init__(self, input_channels: int = 1, iterationSemiBuildup: float = 1e5, semiLambda: int = 1,
                 MC_runs: int = 10, tau: List[float] = [0.1, 0.9], psi: List[float] = 0.1,
                 device: Union[torch.device, str] = 'cpu'):
        super(NaivePseudoLabel, self).__init__()
        self.input_channels = input_channels
        self.MC_runs = MC_runs
        self.semiLambda = semiLambda
        self.device = device
        self.pretrained = False

        self.UNet = myutils.UNet(encoder_channels=[input_channels, 64, 128, 256, 512],
                                 decoder_channels=[512, 256, 128, 64, 32, 1],
                                 type='2D',
                                 interpolate=True,
                                 dropout=True)

        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.semi_loss = nn.BCEWithLogitsLoss(reduction='none')

        # Number of iterations until semi-supervised is completely included
        self.iterationSemiBuildup = iterationSemiBuildup
        # Confidence threshold
        self.tau_negative, self.tau_positive = tau
        assert self.tau_negative <= self.tau_positive, "Tau negative should be smaller than tau positive!"

        # Uncertainty thresholds
        self.uncert_positive, self.uncert_negative = psi


    def comp_semi_mask(self, pseudo_pred_prob: torch.Tensor, semi: torch.Tensor) -> torch.Tensor:
        """
        Builds on the certainy mask based on several MC forward runs through the network
        """
        confidence_mask = torch.logical_or(pseudo_pred_prob > self.tau_positive,
                                           pseudo_pred_prob < self.tau_negative)

        # Build a mask based on the uncertainty defined by the standard error over several MC runs
        with torch.no_grad():
            first_moment = pseudo_pred_prob.detach().clone()
            second_moment = first_moment ** 2

            for n_pass in range(self.MC_runs - 1):
                cur_pred = torch.sigmoid(self.UNet(semi))
                first_moment += cur_pred
                second_moment += cur_pred ** 2

            first_moment = first_moment / self.MC_runs
            second_moment = second_moment / self.MC_runs

            std_error = torch.sqrt(second_moment - first_moment ** 2)
            certainty_mask = torch.logical_and(std_error < self.uncert_positive,
                                               pseudo_pred_prob > self.tau_positive)
            certainty_mask = torch.logical_or(certainty_mask,
                                              torch.logical_and(std_error < self.uncert_negative,
                                                                pseudo_pred_prob < self.tau_negative))

        semi_mask = torch.logical_and(confidence_mask, certainty_mask)
        return semi_mask


class AleatoricPseudoLabel(BaseUNet):
    def __init__(self, input_channels: int = 1, iterationSemiBuildup: float = 1e5, semiLambda: float = 0.1,
                 device: Union[torch.device, str] = 'cpu'):
        super().__init__(input_channels, device)


        # Number of iterations until semi-supervised is completely included
        self.iterationSemiBuildup = iterationSemiBuildup
        self.semiLambda = semiLambda
        # Confidence threshold
        self.tau_positive = 0.75
        self.tau_negative = 0.25
        # Uncertainty thresholds
        self.uncert_positive = 0.05
        self.uncert_negative = 0.005


    def comp_loss(self, patch: torch.Tensor, semi: Optional[torch.Tensor] = None, annotations: torch.Tensor = None,
                  iteration: int = 0, tracker=None, mode='train') -> torch.Tensor:
        # Generate and fit pseudo labels (not present in evaluation)
        if semi is not None:
            semi, annotations_semi = semi
            pseudo_pred = self.UNet(semi)
            pseudo_pred_prob = torch.sigmoid(pseudo_pred)
            confidence_mask = torch.logical_or(pseudo_pred_prob > self.tau_positive, pseudo_pred_prob < self.tau_negative)
            pseudo_labels = (pseudo_pred_prob > 0.5).detach().clone().float()

            # Build a mask based on the uncertainty defined by the standard error over several MC runs
            # Compute the standard error based on the uncertainty of the multiple annotators
            # Or just the pixels with disagreement if there are only two annotators
            if annotations_semi.shape[1] > 2:
                std_error = torch.std(annotations_semi, dim=1, keepdim=True) * 2  # to normalise to [0,1]
                certainty_mask = torch.logical_and(std_error < self.uncert_positive,
                                                     pseudo_pred > self.tau_positive)
                certainty_mask = torch.logical_or(certainty_mask,
                                                    torch.logical_and(std_error < self.uncert_negative,
                                                                      pseudo_pred < self.tau_negative))
            else:
                certainty_mask = (annotations_semi[:, 0] == annotations_semi[:, 1])[:, None]

            semi_mask = torch.logical_and(confidence_mask, certainty_mask)
            loss_semi = self.loss(pseudo_pred[semi_mask], pseudo_labels[semi_mask])
            smooth_in_parameter = self.sigmoid_rampup(iteration, self.iterationSemiBuildup)
            tracker.buffer_scalar(f'Uncertainty mask %', (semi_mask.sum() / semi_mask.numel()).detach().cpu().item(),
                                  iteration, mode)
        else:
            loss_semi = torch.Tensor([0]).to(patch.device)
            smooth_in_parameter = 0

        # Supervised part
        loss_supervised = self.supervised_loss(patch, annotations)

        # Loss
        loss = loss_supervised + loss_semi * smooth_in_parameter * self.semiLambda
        tracker.buffer_scalar(f"Loss ({mode})", loss.detach().cpu().item(), iteration, mode)

        if semi is not None:
            tracker.buffer_scalar(f"Loss scaled semi",
                                  loss_semi.detach().cpu().item() * smooth_in_parameter * self.semiLambda,
                                  iteration, mode)
        return loss


class PredAleatoricPL(NaivePseudoLabel):
    def __init__(self, input_channels: int = 1, iterationSemiBuildup: float = 1e5, semiLambda: float = 0.1,
                 detectionNetworkPath: Optional[str] = None, tau: List[float] = [0.1, 0.9],
                 pretrained_uncertainty_network: bool = False,
                 uncertaintyThreshold: float = 0.5, device: Union[torch.device, str] = 'cpu'):
        super().__init__(input_channels, iterationSemiBuildup, semiLambda, tau, device)

        if detectionNetworkPath is not None:
            pretrain = pretrained_uncertainty_network
            if pretrain:
                self.uncertainty_network = smp.Unet(
                    encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                    in_channels=input_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    # 'aux_params'= {'dropout': 0.5},  # This dropout is only applied to a classificatino head
                    classes = 1,  # model output channels (number of classes in my dataset)
                    activation=None,
                )
            else:
                self.uncertainty_network = myutils.UNet(encoder_channels=[input_channels, 64, 128, 256, 512],
                                         decoder_channels=[512, 256, 128, 64, 32, 1],
                                         type='2D',
                                         interpolate=True,
                                         dropout=False)

            saved_model = torch.load(detectionNetworkPath)['model_state_dict']
            old_keys = list(saved_model.keys())
            for key in old_keys:
                saved_model[key[key.find('.') + 1:]] = saved_model.pop(key)
            self.uncertainty_network.load_state_dict(saved_model)
        else:
            raise ValueError("No prediction networkpath provided")

        self.uncertainty_threshold = uncertaintyThreshold

    def comp_semi_mask(self, pseudo_pred_prob, semi):
        confidence_mask = torch.logical_or(pseudo_pred_prob > self.tau_positive,
                                           pseudo_pred_prob < self.tau_negative)
        # Predict aleatoric uncertainty
        with torch.no_grad():
            # Pick areas with LOW uncertainty
            certainty_mask = torch.sigmoid(self.uncertainty_network(semi)) < self.uncertainty_threshold

        semi_mask = torch.logical_and(confidence_mask, certainty_mask)
        return semi_mask


class EnsembleNetwork(NaivePseudoLabel):
    def __init__(self, input_channels: int = 1, iterationSemiBuildup: float = 1e5, semiLambda: float = 0.1,
                  tau: List[float] = [0.1, 0.9],
                 psi: List[float] = 0.1,
                 ensemble_path: str = '',
                 device: Union[torch.device, str] = 'cpu') -> None:
        super().__init__(input_channels=input_channels,
                         iterationSemiBuildup=iterationSemiBuildup,
                         semiLambda=semiLambda,
                         tau=tau,
                         device=device)

        self.ensemble = self.load_ensemble(ensemble_path)
        self.n_ensemble = len(self.ensemble)

        # Uncertainty thresholds
        self.uncert_positive, self.uncert_negative = psi

    def load_ensemble(self, path: str) -> List[nn.Module]:
        dirs = [d for d in os.listdir(path) if d.startswith('Ensemble_Networks')]
        assert len(dirs) > 0, 'No networks found!'
        model_paths = [os.path.join(path, d, 'BestModel.pt') for d in dirs]
        ensemble = []
        for p in model_paths:
            cur_model = self.get_model()
            saved_model = torch.load(p)['model_state_dict']
            old_keys = list(saved_model.keys())
            for key in old_keys:
                saved_model[key[key.find('.') + 1:]] = saved_model.pop(key)
            cur_model.load_state_dict(saved_model)
            ensemble.append(cur_model.to(self.device))
        return ensemble

    def comp_semi_mask(self, pseudo_pred_prob: torch.Tensor, semi: torch.Tensor) -> torch.Tensor:
        """
        Builds on the certainy mask based on several MC forward runs through the network
        """
        confidence_mask = torch.logical_or(pseudo_pred_prob > self.tau_positive,
                                           pseudo_pred_prob < self.tau_negative)

        # Build a mask based on the uncertainty defined by the standard error over several MC runs
        with torch.no_grad():
            first_moment = pseudo_pred_prob.detach().clone() * 0
            second_moment = first_moment ** 2

            for cur_model in self.ensemble:
                cur_model.eval()
                cur_pred = torch.sigmoid(cur_model(semi))
                first_moment += cur_pred
                second_moment += cur_pred ** 2

            first_moment = first_moment / self.n_ensemble
            second_moment = second_moment / self.n_ensemble

            std_error = torch.sqrt(second_moment - first_moment ** 2)
            certainty_mask = torch.logical_and(std_error < self.uncert_positive,
                                               pseudo_pred_prob > self.tau_positive)
            certainty_mask = torch.logical_or(certainty_mask,
                                              torch.logical_and(std_error < self.uncert_negative,
                                                                pseudo_pred_prob < self.tau_negative))

        semi_mask = torch.logical_and(confidence_mask, certainty_mask)
        return semi_mask


class UncertaintyUNet(BaseUNet):
    """Defines a UNet model that tries to predict the uncertainty based on multiple annotations"""

    def calc_aleatoric_uncertainty(self, masks: torch.Tensor) -> torch.Tensor:
        """Mask with shape [B, C, H, W] where C is the number annotations"""
        return torch.std(masks, dim=1, keepdim=True) * 2  # to normalise to [0,1]

    def comp_loss(self, patch: torch.Tensor, semi: Optional[torch.Tensor] = None, annotations: torch.Tensor = None,
                  iteration: int = 0, tracker=None, mode='train') -> torch.Tensor:
        """ Mode is used in the tracker as an additional label"""
        target = self.calc_aleatoric_uncertainty(annotations)
        pred = self.UNet(patch)
        loss = self.loss(pred, target)
        tracker.buffer_scalar(f"Loss ({mode})", loss.detach().cpu().item(), iteration, mode)
        return loss


class StudentTeacherUNet(BaseUNet):
    """This class basically implementes Yu et al (2019) Uncertainty aware self ensembling model"""
    def __init__(self, teacher = False, max_iter = None, consistency_lambda = 0.1, consistency_rampup = 40,
                 input_channels = 1, device = 'cpu'):
        super(BaseUNet, self).__init__()

        UNet_args = {
            "encoder_name": "resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            "encoder_weights": "imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            "in_channels": input_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            "activation": None,
            "classes": 1                      # model output channels (number of classes in your dataset)
        }
        self.UNet = smp.Unet(**UNet_args)
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.device = device
        self.max_iter = max_iter
        self.consistency_lambda = consistency_lambda
        self.consistency_rampup = consistency_rampup
        self.noise_max = 0.2

        # Copy model and detach the parameters
        self.teacher = smp.Unet(**UNet_args)
        # self.teacher.load_state_dict(self.UNet.state_dict())
        self.teacher_update_counter = 0
        for teacher_par in  self.teacher.parameters():
            teacher_par.detach_()

    def comp_loss(self, patch, semi = None, annotations = None, iteration = None, tracker = None, mode ='train'):
        """ Mode is used in the tracker as an additional label"""

        m_vote = self.majority_vote(annotations).float()
        pred = self.UNet(patch)
        loss = self.loss(pred, m_vote)
        tracker.buffer_scalar(f'Supervised Loss ({mode})', loss.detach().cpu().item(), iteration)

        # Student teacher interaction
        stud_pred = self.UNet(semi)

        noise = torch.clamp(torch.randn_like(semi) * 0.1, -self.noise_max, self.noise_max)
        with torch.no_grad():
            teacher_pred = self.teacher(semi + noise)

        # Compute uncertainty for thresholding
        T = 8
        batch_size = semi.shape[0]

        pred = torch.zeros([T, batch_size, 1, semi.shape[-2], semi.shape[-1]]).to(self.device)
        for i in range(T):
            noise = torch.clamp(torch.randn_like(semi) * 0.1, -self.noise_max, self.noise_max)
            with torch.no_grad():
                pred[i] = self.teacher(semi + noise)
        pred = torch.sigmoid(pred)
        pred = torch.mean(pred, dim = 0)  # (BS, 1, H, W)
        # To calculate the entropy we concat p with 1-p
        pred = torch.cat((pred, 1-pred), dim =1)  # (BS, 2, H, W)
        uncertainty = -1 * torch.sum(pred * torch.log(pred + 1e-8), dim = 1, keepdim = True)  # (batch, 1, H, W)
        uncertainty_threshold = (0.75+0.25*self.sigmoid_rampup(iteration, self.max_iter))*np.log(2)
        mask = (uncertainty < uncertainty_threshold).float()
        tracker.buffer_scalar(f'Uncertainty threshold ({mode})', uncertainty_threshold,
                              iteration, mode = 'debug')

        # Compute MSE consistency between student and teacher prediction
        trans = torch.sigmoid
        consistency_loss = (trans(stud_pred) - trans(teacher_pred))**2
        consistency_loss = torch.sum(mask * consistency_loss) / (2*torch.sum(mask) + 1e-8)

        # Weight by ramp up function
        consistency_weight = self.consistency_lambda * self.sigmoid_rampup(iteration, self.consistency_rampup)
        tracker.buffer_scalar(f'Concistency weight ({mode})', consistency_weight, iteration, mode='debug')
        semi_loss = consistency_loss * consistency_weight

        tracker.buffer_scalar(f"Concistency loss ({mode})", semi_loss.detach().cpu().item(), iteration, mode)
        loss += semi_loss

        tracker.buffer_scalar(f"Total loss ({mode})", loss.detach().cpu().item(), iteration, mode)
        return loss

    def update(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.update_teacher()

    def update_teacher(self, alpha = 0.999):
        alpha = min(1 - 1 / (self.teacher_update_counter + 1), alpha)  # Taken from Yu 2019, use the true average until EA is more correct
        with torch.no_grad():
            for ema_param, param in zip(self.teacher.parameters(), self.UNet.parameters()):
                ema_param.data.mul_(alpha).add_(param.data, alpha = (1-alpha))

        self.teacher_update_counter += 1

    def predict(self, x, use_teacher = True):
        pred = self.teacher(x)
        return torch.sigmoid(pred)

    def sample(self, x, n_samples):
        # Since the model is deterministic we just repeat the same prediction n_samples time
        pred = self.teacher(x)

        # Make sure pred is shape [B, 1, W, H]
        assert pred.shape[1] == 1
        return pred.expand(-1, n_samples, -1, -1)


