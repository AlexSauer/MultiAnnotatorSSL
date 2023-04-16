import torch
from torch import nn
import torch.nn.functional as F
from models.ProbUNet.unet_blocks import *
from models.ProbUNet.unet import Unet
from models.ProbUNet.utils import init_weights, init_weights_orthogonal_normal, l2_regularisation
from torch.distributions import Normal, Independent, kl as KL



class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """

    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers, padding=True, posterior=False):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            # To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += 1

        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]

            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block - 1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output


class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """

    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim,
                 initializers, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block,
                               initializers, posterior=self.posterior)
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1, 1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):

        # If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input)
        self.show_enc = encoding

        # We only want the mean of the resulting h x w image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        # Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)

        # We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]

        # This is a multivariate normal with diagonal covariance matrix sigma
        # https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        return dist


class Fcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """

    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, no_convs_fcomb, initializers,
                 use_tile=True, device = 'cpu'):
        super(Fcomb, self).__init__()
        self.num_channels = num_output_channels  # output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2, 3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb
        self.name = 'Fcomb'
        self.device = device

        if self.use_tile:
            layers = []

            # Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(self.num_filters[0] + self.latent_dim, self.num_filters[0], kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb - 2):
                layers.append(nn.Conv2d(self.num_filters[0], self.num_filters[0], kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)

            self.last_layer = nn.Conv2d(self.num_filters[0], self.num_classes, kernel_size=1)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
            self.device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is batch_size x latent_dim and feature_map is batch_size x no_channels x H x W.
        So broadcast Z to batch_size x latent_dim x H x W. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z, 2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z, 3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

            # Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            output = self.layers(feature_map)
            return self.last_layer(output)


class ProbabilisticUnet(nn.Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    def __init__(self, input_channels=1, num_classes=1, num_filters=[32, 64, 128, 192], latent_dim=6, no_convs_fcomb=4,
                 beta=1.0, beta_w=1.0, beta_l2reg = 1e-5, entropy_coeff = 1, device = 'cpu', teacher = False):
        super(ProbabilisticUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {'w': 'he_normal', 'b': 'normal'}
        self.beta = beta  # Balancing the reconstruction loss with the KL divergence for the ELBO
        self.beta_w = beta_w  # Balance the KL divergence between a normal prior on the model weights and the actual weights
        self.l2_reg = beta_l2reg
        self.entropy_coeff = entropy_coeff
        self.device = device
        self.sig = nn.Sigmoid()

        self.unet = Unet(self.input_channels, self.num_classes, self.num_filters, self.initializers,
                         apply_last_layer=False, padding=True).to(self.device)
        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block,
                                             self.latent_dim, self.initializers, ).to(self.device)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block,
                                                 self.latent_dim, self.initializers, posterior=True).to(self.device)
        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.input_channels, self.num_classes,
                           self.no_convs_fcomb, {'w': 'orthogonal', 'b': 'normal'},
                           use_tile=True, device=device).to(self.device)

        if teacher:
            # Copy model and detach the parameters
            self.teacher = Unet(self.input_channels, self.num_classes, self.num_filters, self.initializers,
                         apply_last_layer=False, padding=True).to(self.device)
            # self.teacher.load_state_dict(self.UNet.state_dict())
            for teacher_par in  self.teacher.parameters():
                teacher_par.detach_()
        else:
            self.teacher = None

    def forward(self, patch, segm=None, training=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        if training:
            self.posterior_latent_space = self.posterior.forward(patch, segm)
        self.prior_latent_space = self.prior.forward(patch)
        self.unet_features = self.unet.forward(patch, val=False)

    def sample(self, patch=None, unet_features=None, testing=False, n_samples = 1, z_prior_dist = None):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """

        # Compute the U-Net features if not given
        if unet_features is None:
            if testing != False:
                self.unet._sample_on_eval(True)
            unet_features = self.unet.forward(patch, False)

        # You can choose whether you mean a sample or the mean here. For the GED it is important to take a sample.
        if z_prior_dist is None:
            z_prior_dist = self.prior.forward(patch)

        if n_samples == 1:
            z_prior = z_prior_dist.rsample()
            reconstruction = self.fcomb.forward(unet_features, z_prior)
        else:
            rec = []

            for i in range(n_samples):
                z_prior = z_prior_dist.rsample()
                rec.append(self.fcomb.forward(unet_features, z_prior))

            reconstruction = torch.cat(rec, dim = 1)

        return reconstruction

    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None, unet_features = None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.base_dist.loc
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()

        if unet_features is None:
            return self.fcomb.forward(self.unet_features, z_posterior)
        else:
            return self.fcomb.forward(unet_features, z_posterior)

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            # Need to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = KL.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(self, segm, analytic_kl=True, reconstruct_posterior_mean=False, tracker = None, iteration = None, mode = 'train'):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        modified Eq. (4) of https://arxiv.org/abs/1806.05034
        sum log P(y|x) >= L, where
        L = sum E[log p(y|z,w,x)] - sum KL[q(z|x, y) || p(z|x)] - KL[q(w) || p(w)]
        """

        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        z_posterior = self.posterior_latent_space.rsample()

        # kl of z (second term) pulls the prior towards the posterior embedding
        # kl = torch.Tensor([0]).to(self.device)  # For debugging
        kl = torch.mean(self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior))
        # print(f'KL component: {kl}')

        # Here we use the posterior sample sampled above
        reconstruction = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, calculate_posterior=False,
                                               z_posterior=z_posterior)  #z_posterior=torch.zeros([8,6]).to(self.device))
        reconstruction_loss = criterion(input=reconstruction, target=segm)  # -log_prob

        if tracker is not None:
            tracker.buffer_scalar(f"Reconstruction loss ({mode})", reconstruction_loss.detach().cpu().item(), iteration, mode)
            tracker.buffer_scalar(f"KL latent space ({mode})", kl.detach().cpu().item(), iteration, mode)

        elbo = -(reconstruction_loss + self.beta * kl)
        # print('elbo ' + str(elbo.item()) + ', recon_loss ' + str(reconstruction_loss.item()) + ', kl_z ' + str(
        #     kl.item()))

        return elbo

    def comp_loss(self, patch, semi = None, masks = None, iteration = None, tracker = None, mode = 'train'):
        """
        Computes and returns the loss
        """

        elbo_sum = 0.
        prior_preds = []

        num_graders = masks.shape[1]
        batch_size = patch.shape[0]

        self.unet._set_deterministic(True)  # use mean of weights in unet
        self.unet_features = self.unet.forward(patch, False)
        self.unet._set_deterministic(False)  # use random weights in unet

        for g in range(num_graders):
            mask = masks[:, g, :, :]
            mask = torch.unsqueeze(mask, 1)

            # Set latent space
            self.posterior_latent_space = self.posterior.forward(patch, mask)
            self.prior_latent_space = self.prior.forward(patch)

            elbo = self.elbo(mask, tracker=tracker, iteration=iteration, mode=mode)  # FIRST AND SECOND TERM
            elbo_sum += elbo

            # for entropy loss, This is their innovation to include the inter-annotator variablility (LAST TERM)
            if self.entropy_coeff != 0:
                prior_pred = self.sample(patch, self.unet_features, testing=False, z_prior_dist=self.prior_latent_space)
                prior_preds.append(self.sig(prior_pred))

        elbo_mean = elbo_sum / (num_graders * batch_size)  # Different from original implementation which continues with the sum

        # This is their innovation to include the inter-annotator variablility (LAST TERM)
        if self.entropy_coeff != 0:
            prior_preds = torch.cat(prior_preds, 1)
            mean_prior_preds = torch.mean(prior_preds, 1)
            mean_masks = torch.mean(masks, 1)

            ce = F.binary_cross_entropy(mean_prior_preds, mean_masks.detach(), reduction='none')
            ce /= torch.log(torch.tensor(2.)).to(self.device)  # log in binary_cross_entropy has base e
            entropy_loss = torch.mean(ce)
        else:
            entropy_loss = torch.Tensor([0]).to(self.device)

        tracker.buffer_scalar(f"BCE inter-annotator variability ({mode})",
                              entropy_loss.detach().cpu().item(), iteration, mode)

        # print('entropy loss ' + str(entropy_loss.item()))

        # weight regularization loss
        if self.l2_reg != 0:
            reg_loss = l2_regularisation(self.posterior) + \
                       l2_regularisation(self.prior) + \
                       l2_regularisation(self.fcomb.layers)
        else:
            reg_loss = torch.Tensor([0]).to(self.device)

        # kl of w for variational dropout (THIRD TERM)
        if self.beta_w == 0:
            kl_w = torch.Tensor([0]).to(self.device)
        else:
            kl_w = self.unet.regularizer()
        # print('kl_w ' + str(kl_w.item()))
        tracker.buffer_scalar(f"KL model parameters w ({mode})", kl_w.detach().cpu().item(), iteration, mode)

        # total loss
        inv_datalen = 1. / batch_size
        # print('ELBO ' + str(round(-elbo_mean.item(),2)) +
        #       ', kl_w ' + str(round(inv_datalen * self.beta_w * kl_w.item(), 2)) +
        #       ', entropy loss ' + str(round(self.entropy_coeff * entropy_loss.item(), 2)))

        loss = -elbo_mean + inv_datalen * self.beta_w * kl_w + self.entropy_coeff * entropy_loss + self.l2_reg * reg_loss
        tracker.buffer_scalar(f"Total loss ({mode})", loss.detach().cpu().item(), iteration, mode)
        return loss

    def update(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def predict(self, img):
        self.eval()
        with torch.no_grad():
            # If we look at a single image we need to add a dimension to have [B, C, H, W]
            if len(img.shape) == 3:
                img = img[None]

            # Init self.unet_features
            self.forward(img, training=False)
            pred = self.reconstruct(use_posterior_mean=True)
            return torch.sigmoid(pred)
