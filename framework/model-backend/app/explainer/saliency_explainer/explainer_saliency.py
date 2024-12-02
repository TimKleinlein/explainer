from explainer.explainer_base import Explainer
import torch
from captum.attr._core.saliency import Saliency
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import visualization as viz
import numpy as np



class SaliencyExplainer(Explainer):

    def __init__(self):

        self.explainer_name = 'Saliency Maps'


    def explain(self, model, data, params):
        # get prediction
        output = model(data)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)  # returns value & index of top 1 label
        pred_label_idx.squeeze_()

        # use saliency attribute method to get explanation
        abs = params['abs']
        saliency = Saliency(model)
        attribution = saliency.attribute(data, target=pred_label_idx, abs=abs)

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
