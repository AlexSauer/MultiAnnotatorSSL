from models.ProbUNet.ProbUNet import ProbabilisticUnet
from models.UNet.UNet import StudentTeacherUNet, UncertaintyUNet, BaseUNet, NaivePseudoLabel, \
                             MCPseudoLabel, AleatoricPseudoLabel, UncertaintyDetectionNetwork, \
                             PredAleatoricPL, EnsembleNetwork
def build_model(args, **kwargs):
    if args.model_name == "ProbUNet" or args.model_name == 'PUNet':
        return ProbabilisticUnet(input_channels=args.input_channels,
                                 num_classes=1,
                                 num_filters=[32, 64, 128, 192],
                                 latent_dim=6,
                                 no_convs_fcomb=4,
                                 beta=args.beta_kl,
                                 beta_w=args.beta_w,
                                 entropy_coeff= args.entropy_coeff,
                                 beta_l2reg = args.l2_reg,
                                 device = args.device)

    elif args.model_name == 'StudentTeacher':
        return StudentTeacherUNet(teacher = args.teacher, max_iter=args.n_iterations,
                                  consistency_lambda = args.consistency_lambda,
                                  consistency_rampup = args.consistency_rampup,
                                  input_channels=args.input_channels,
                                  device = args.device)

    elif args.model_name == 'UNet':
        return BaseUNet(input_channels=args.input_channels, device=args.device)

    elif args.model_name == 'UncertaintyUNet':
        return UncertaintyUNet(input_channels=args.input_channels, device=args.device)

    elif args.model_name == 'NaivePseudoLabel':
        return NaivePseudoLabel(input_channels=args.input_channels,
                                iterationSemiBuildup=args.iterationSemiBuildup,
                                semiLambda=args.semiLambda,
                                tau=args.tau,
                                device=args.device)

    elif args.model_name == 'MCPseudoLabel':
        return MCPseudoLabel(input_channels=args.input_channels,
                             iterationSemiBuildup=args.iterationSemiBuildup,
                             semiLambda=args.semiLambda,
                             MC_runs=args.MC_runs,
                             tau=args.tau,
                             psi=args.psi,
                             device=args.device)

    elif args.model_name == 'AleatoricPseudoLabel':
        return AleatoricPseudoLabel(input_channels=args.input_channels,
                                    iterationSemiBuildup=args.iterationSemiBuildup,
                                    semiLambda=args.semiLambda,
                                    device=args.device)

    elif args.model_name == 'UncertaintyDetectionNetwork':
        return UncertaintyDetectionNetwork(input_channels=args.input_channels,
                                           device=args.device)

    elif args.model_name == 'PredAleatoricPL':
        return PredAleatoricPL(input_channels=args.input_channels,
                               iterationSemiBuildup=args.iterationSemiBuildup,
                               semiLambda=args.semiLambda,
                               detectionNetworkPath=args.detectionNetworkPath,
                               tau=args.tau,
                               uncertaintyThreshold=args.uncertaintyThreshold,
                               device=args.device,
                               **kwargs)
    elif args.model_name == 'EnsembleNetwork':
        return EnsembleNetwork(input_channels=args.input_channels,
                               iterationSemiBuildup=args.iterationSemiBuildup,
                               semiLambda=args.semiLambda,
                               ensemble_path=args.ensemblePath,
                               tau=args.tau,  # Confidence threshold tuple
                               psi=args.psi,  # Uncertainty threshold tuple
                               device=args.device )
    else:
        raise NotImplementedError
