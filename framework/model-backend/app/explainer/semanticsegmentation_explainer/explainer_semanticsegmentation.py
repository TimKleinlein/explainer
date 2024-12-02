from explainer.explainer_base import Explainer
from captum.attr import LayerGradCam, LayerAttribution
from captum.attr import visualization as viz

class SemanticSegmentationExplainer(Explainer):

    def __init__(self):

        self.explainer_name = 'Semantic Segmentation'


    def explain(self, model, data, params):
        # get prediction
        output = model(data)['out']
        # Find most likely segmentation class for each pixel.
        # out_max = torch.argmax(output, dim=1, keepdim=True)

        # get explanation
        layer = 'model.' + params['layer']
        target = params['target']

        def agg_segmentation_wrapper(inp):
            return model(inp)['out'].sum(dim=(2, 3))

        LGC = LayerGradCam(agg_segmentation_wrapper, eval(layer))
        attribution = LGC.attribute(data, target=target)

        upsampled_gc_attr = LayerAttribution.interpolate(attribution, data.shape[2:])
        _ = viz.visualize_image_attr_multiple(upsampled_gc_attr[0].cpu().permute(1, 2, 0).detach().numpy(),
                                          original_image=data.squeeze(0).permute(1, 2, 0).numpy(),
                                          signs=["all", "positive", "negative"],
                                          methods=["original_image", "blended_heat_map", "blended_heat_map"])

        return attribution, _
