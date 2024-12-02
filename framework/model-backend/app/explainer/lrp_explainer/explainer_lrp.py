from explainer.explainer_base import Explainer
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import visualization as viz
import numpy as np
from captum.attr import LRP

# problem with resnet:
# https://github.com/pytorch/captum/issues/1138

class LRPExplainer(Explainer):

    def __init__(self):

        self.explainer_name = 'LRP'


    def explain(self, model, data, params):
        # get prediction
        output = model(data)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)  # returns value & index of top 1 label
        pred_label_idx.squeeze_()

        # use lrp attribute method to get explanation
        return_convergence_delta = params['return_convergence_delta']
        lrp = LRP(model)
        attribution = lrp.attribute(data, target=pred_label_idx, return_convergence_delta=return_convergence_delta)

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
