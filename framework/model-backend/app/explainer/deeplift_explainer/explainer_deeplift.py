from explainer.explainer_base import Explainer
import torch
from captum.attr import DeepLift
import torch.nn.functional as F
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import visualization as viz


class DeepLiftExplainer(Explainer):

    def __init__(self):

        self.explainer_name = 'DeepLift'


    def explain(self, model, data, params):
        # get prediction
        output = model(data)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)  # returns value & index of top 1 label
        pred_label_idx.squeeze_()

        # use deeplift attribute method to get explanation
        baselines = params['baselines']
        eps = params['eps']
        multiply_by_inputs = params['multiply_by_inputs']
        return_convergence_delta = params['return_convergence_delta']
        dl = DeepLift(model, eps=eps, multiply_by_inputs=multiply_by_inputs)
        attribution = dl.attribute(data, baselines=baselines, target=pred_label_idx, return_convergence_delta=return_convergence_delta)

        # create saliency map from attribution tensor
        default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                         [(0, '#ffffff'),
                                                          (0.25, '#000000'),
                                                          (1, '#000000')], N=256)

        _ = viz.visualize_image_attr(np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                     np.transpose(data.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                     method='blended_heat_map',
                                     cmap=default_cmap,
                                     show_colorbar=True,
                                     sign='absolute_value',
                                     outlier_perc=1)
        return attribution, _
