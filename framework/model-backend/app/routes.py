import numpy as np
import sys
from fastapi import APIRouter
from fastapi import Form, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from io import BytesIO
from typing import List, Tuple
import warnings
import json
import os, glob
import importlib.util
from onnx2torch import convert
import zipfile
from zipfile import ZipFile
import shutil
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from captum.attr import LayerIntegratedGradients
from captum.concept import Concept
from captum.concept import TCAV
import torchvision
from torchvision import models

from explainer.lrp_explainer.explainer_lrp import LRPExplainer
from explainer.lime_explainer.explainer_lime import LimeExplainer
from explainer.gradient_explainer.explainer_gradient import GradientExplainer
from explainer.deeplift_explainer.explainer_deeplift import DeepLiftExplainer
from explainer.saliency_explainer.explainer_saliency import SaliencyExplainer
from explainer.deconvnet_explainer.explainer_deconvnet import DeConvNetExplainer
from explainer.occlusion_explainer.explainer_occlusion import OcclusionExplainer
from explainer.guidedgradcam_explainer.explainer_guidedgradcam import GuidedGradCAMExplainer
from explainer.semanticsegmentation_explainer.explainer_semanticsegmentation import SemanticSegmentationExplainer



router = APIRouter()


@router.get("/")
async def home():
    return {"Welcome to the application"}


# TODO: cache model and data

@router.post('/saliency')
async def saliency(data: List[UploadFile], model: UploadFile, architecture: UploadFile = None, abs: bool = Form(True), heatmap: bool = Form(False), torchscript: bool = Form(False)):

    if architecture is None:  # model is uploaded via a single .onnx file
        file_path = await save_uploaded_file(model)
        if torchscript:
            file_path_with_extension = file_path + ".pt"
        else:
            file_path_with_extension = file_path + ".onxx"
        os.rename(file_path, file_path_with_extension)
        model = file_path_with_extension

        params = {'abs': abs}
        explanations = []
        predictions = []
        heatmaps = []
        for d in data:
            d.file.seek(0)
            result = await get_explanation(SaliencyExplainer, data=d, model=model, params=params, torchscript=torchscript)
            explanations.append(result[0].tolist())
            predictions.append(result[1].tolist())
            heatmaps.append(result[2])
        if not heatmap:
            return JSONResponse(content={'explanations': explanations, 'predictions': predictions})
        else:
            return heatmaps

    else:  # model is uploaded via two files: a model statedict .pth file (the model parameter) and a architecture .py file
        file_path = await save_uploaded_file(architecture)
        file_path_with_extension = file_path + ".py"
        os.rename(file_path, file_path_with_extension)

        params = {'abs': abs}
        explanations = []
        predictions = []
        heatmaps = []
        for d in data:
            d.file.seek(0)
            model.file.seek(0)
            result = await get_explanation_with_architecture(SaliencyExplainer, data=d, model_dic=model,
                                           architecture_path=file_path_with_extension, params=params)
            explanations.append(result[0].tolist())
            predictions.append(result[1].tolist())
            heatmaps.append(result[2])
        if not heatmap:
            return JSONResponse(
                content={'explanations': explanations, 'predictions': predictions})
        else:
            return heatmaps

@router.post('/saliency_heatmap')
async def saliency_heatmap(data: List[UploadFile], model: UploadFile, architecture: UploadFile = None, abs: bool = Form(True), torchscript: bool = Form(False)):
    heatmaps = await saliency(data, model, architecture, abs, heatmap=True, torchscript=torchscript)
    if os.path.exists('heatmaps'):
        shutil.rmtree('heatmaps')
    os.makedirs('heatmaps')
    
    for index, i in enumerate(heatmaps):
        fig, ax = i
        fig.savefig(f'heatmaps/figure{index}.pdf')

    # create zip file
    with zipfile.ZipFile('heatmaps.zip', "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add all files in the directory to the zip
        for filename in os.listdir('heatmaps'):
            file_path = os.path.join('heatmaps', filename)
            zipf.write(file_path, arcname=filename)
    return FileResponse('heatmaps.zip')


@router.post('/deeplift')
async def deeplift(data: List[UploadFile], model: UploadFile, architecture: UploadFile = None, baselines: str = Form(None), multiply_by_inputs: bool = Form(True), eps: float = Form(1e-10), return_convergence_delta: bool = Form(False), heatmap: bool = Form(False), torchscript: bool = Form(False)):

    if architecture is None:  # model is uploaded via a single .onnx file
        file_path = await save_uploaded_file(model)
        if torchscript:
            file_path_with_extension = file_path + ".pt"
        else:
            file_path_with_extension = file_path + ".onxx"
        os.rename(file_path, file_path_with_extension)
        model = file_path_with_extension

        if baselines is not None:
            baselines = json.loads(baselines)  # Deserialize the string representation back into a list
            if not isinstance(baselines, list):
                warnings.warn("Invalid baselines format. Expected a PyTorch tensor.")
            baselines = torch.tensor(baselines)  # convert the list to a tensor
        params = {'baselines': baselines,
                  'multiply_by_inputs': multiply_by_inputs,
                  'eps': eps,
                  'return_convergence_delta': return_convergence_delta}
        explanations = []
        predictions = []
        heatmaps = []
        for d in data:
            d.file.seek(0)
            result = await get_explanation(DeepLiftExplainer, data=d, model=model, params=params, torchscript=torchscript)
            explanations.append(result[0].tolist())
            predictions.append(result[1].tolist())
            heatmaps.append(result[2])
        if not heatmap:
            return JSONResponse(
                content={'explanations': explanations, 'predictions': predictions})
        else:
            return heatmaps

    else: # model is uploaded via two files: a model statedict .pth file (the model parameter) and a architecture .py file
        file_path = await save_uploaded_file(architecture)
        file_path_with_extension = file_path + ".py"
        os.rename(file_path, file_path_with_extension)

        if baselines is not None:
            baselines = json.loads(baselines)  # Deserialize the string representation back into a list
            if not isinstance(baselines, list):
                warnings.warn("Invalid baselines format. Expected a PyTorch tensor.")
            baselines = torch.tensor(baselines)  # convert the list to a tensor
        params = {'baselines': baselines,
                  'multiply_by_inputs': multiply_by_inputs,
                  'eps': eps,
                  'return_convergence_delta': return_convergence_delta}
        explanations = []
        predictions = []
        heatmaps = []
        for d in data:
            d.file.seek(0)
            model.file.seek(0)
            result = await get_explanation_with_architecture(DeepLiftExplainer, data=d, model_dic=model,
                                           architecture_path=file_path_with_extension, params=params)
            explanations.append(result[0].tolist())
            predictions.append(result[1].tolist())
            heatmaps.append(result[2])
        if not heatmap:
            return JSONResponse(
                content={'explanations': explanations, 'predictions': predictions})
        else:
            return heatmaps


@router.post('/deeplift_heatmap')
async def deeplift_heatmap(data: List[UploadFile], model: UploadFile, architecture: UploadFile = None, baselines: str = Form(None), multiply_by_inputs: bool = Form(True), eps: float = Form(1e-10), return_convergence_delta: bool = Form(False), torchscript: bool = Form(False)):
    heatmaps = await deeplift(data, model, architecture, baselines, multiply_by_inputs, eps, return_convergence_delta, heatmap=True, torchscript=torchscript)
    if os.path.exists('heatmaps'):
        shutil.rmtree('heatmaps')
    os.makedirs('heatmaps')

    for index, i in enumerate(heatmaps):
        fig, ax = i
        fig.savefig(f'heatmaps/figure{index}.pdf')

    # create zip file
    with zipfile.ZipFile('heatmaps.zip', "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add all files in the directory to the zip
        for filename in os.listdir('heatmaps'):
            file_path = os.path.join('heatmaps', filename)
            zipf.write(file_path, arcname=filename)
    return FileResponse('heatmaps.zip')

@router.post('/lime')
async def lime(data: List[UploadFile], model: UploadFile, architecture: UploadFile = None, baselines: str = Form(None), feature_mask: str = Form(None), n_samples: int = Form(50), perturbations_per_eval: int = Form(1), return_input_shape: bool = Form(True), heatmap: bool = Form(False), torchscript: bool = Form(False)):

    if architecture is None:  # model is uploaded via a single .onnx file
        file_path = await save_uploaded_file(model)
        if torchscript:
            file_path_with_extension = file_path + ".pt"
        else:
            file_path_with_extension = file_path + ".onxx"
        os.rename(file_path, file_path_with_extension)
        model = file_path_with_extension

        if baselines is not None:
            baselines = json.loads(baselines)  # Deserialize the string representation back into a list
            if not isinstance(baselines, list):
                warnings.warn("Invalid baselines format. Expected a PyTorch tensor.")
            baselines = torch.tensor(baselines)  # convert the list to a tensor
        if feature_mask is not None:
            feature_mask = json.loads(feature_mask)  # Deserialize the string representation back into a list
            if not isinstance(feature_mask, list):
                warnings.warn("Invalid baselines format. Expected a PyTorch tensor.")
            feature_mask = torch.tensor(feature_mask)  # convert the list to a tensor
        params = {'baselines': baselines,
                  'feature_mask': feature_mask,
                  'n_samples': n_samples,
                  'perturbations_per_eval': perturbations_per_eval,
                  'return_input_shape': return_input_shape}
        explanations = []
        predictions = []
        heatmaps = []
        for d in data:
            d.file.seek(0)
            result = await get_explanation(LimeExplainer, data=d, model=model, params=params, torchscript=torchscript)
            explanations.append(result[0].tolist())
            predictions.append(result[1].tolist())
            heatmaps.append(result[2])
        if not heatmap:
            return JSONResponse(
                content={'explanations': explanations, 'predictions': predictions})
        else:
            return heatmaps
    else: # model is uploaded via two files: a model statedict .pth file (the model parameter) and a architecture .py file
        file_path = await save_uploaded_file(architecture)
        file_path_with_extension = file_path + ".py"
        os.rename(file_path, file_path_with_extension)

        if baselines is not None:
            baselines = json.loads(baselines)  # Deserialize the string representation back into a list
            if not isinstance(baselines, list):
                warnings.warn("Invalid baselines format. Expected a PyTorch tensor.")
            baselines = torch.tensor(baselines)  # convert the list to a tensor
        if feature_mask is not None:
            feature_mask = json.loads(feature_mask)  # Deserialize the string representation back into a list
            if not isinstance(feature_mask, list):
                warnings.warn("Invalid baselines format. Expected a PyTorch tensor.")
            feature_mask = torch.tensor(feature_mask)  # convert the list to a tensor
        params = {'baselines': baselines,
                  'feature_mask': feature_mask,
                  'n_samples': n_samples,
                  'perturbations_per_eval': perturbations_per_eval,
                  'return_input_shape': return_input_shape}
        explanations = []
        predictions = []
        heatmaps = []
        for d in data:
            d.file.seek(0)
            model.file.seek(0)
            result = await get_explanation_with_architecture(LimeExplainer, data=d, model_dic=model, architecture_path=file_path_with_extension, params=params)
            explanations.append(result[0].tolist())
            predictions.append(result[1].tolist())
            heatmaps.append(result[2])
    if not heatmap:
        return JSONResponse(content={'explanations': explanations, 'predictions': predictions})
    else:
        return heatmaps


@router.post('/lime_heatmap')
async def lime_heatmap(data: List[UploadFile], model: UploadFile, architecture: UploadFile = None, baselines: str = Form(None), feature_mask: str = Form(None), n_samples: int = Form(50), perturbations_per_eval: int = Form(1), return_input_shape: bool = Form(True), torchscript: bool = Form(False)):
    heatmaps = await lime(data, model, architecture, baselines, feature_mask, n_samples, perturbations_per_eval, return_input_shape, heatmap=True, torchscript=torchscript)
    if os.path.exists('heatmaps'):
        shutil.rmtree('heatmaps')
    os.makedirs('heatmaps')

    for index, i in enumerate(heatmaps):
        fig, ax = i
        fig.savefig(f'heatmaps/figure{index}.pdf')

    # create zip file
    with zipfile.ZipFile('heatmaps.zip', "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add all files in the directory to the zip
        for filename in os.listdir('heatmaps'):
            file_path = os.path.join('heatmaps', filename)
            zipf.write(file_path, arcname=filename)
    return FileResponse('heatmaps.zip')

@router.post('/lrp')
async def lrp(data: List[UploadFile], model: UploadFile, architecture: UploadFile = None, return_convergence_delta: bool = Form(True), heatmap: bool = Form(False), torchscript: bool = Form(False)):

    if architecture is None:  # model is uploaded via a single .onnx file
        file_path = await save_uploaded_file(model)
        if torchscript:
            file_path_with_extension = file_path + ".pt"
        else:
            file_path_with_extension = file_path + ".onxx"
        os.rename(file_path, file_path_with_extension)
        model = file_path_with_extension

        params = {'return_convergence_delta': return_convergence_delta}
        explanations = []
        predictions = []
        heatmaps = []
        for d in data:
            d.file.seek(0)
            result = await get_explanation(LRPExplainer, data=d, model=model, params=params, torchscript=torchscript)
            explanations.append(result[0].tolist())
            predictions.append(result[1].tolist())
            heatmaps.append(result[2])
        if not heatmap:
            return JSONResponse(
                content={'explanations': explanations, 'predictions': predictions})
        else:
            return heatmaps

    else: # model is uploaded via two files: a model statedict .pth file (the model parameter) and a architecture .py file
        file_path = await save_uploaded_file(architecture)
        file_path_with_extension = file_path + ".py"
        os.rename(file_path, file_path_with_extension)

        params = {'return_convergence_delta': return_convergence_delta}
        explanations = []
        predictions = []
        heatmaps = []
        for d in data:
            d.file.seek(0)
            model.file.seek(0)
            result = await get_explanation_with_architecture(LRPExplainer, data=d, model_dic=model, architecture_path=file_path_with_extension, params=params)
            explanations.append(result[0].tolist())
            predictions.append(result[1].tolist())
            heatmaps.append(result[2])
        if not heatmap:
            return JSONResponse(
                content={'explanations': explanations, 'predictions': predictions})
        else:
            return heatmaps


@router.post('/lrp_heatmap')
async def lrp_heatmap(data: List[UploadFile], model: UploadFile, architecture: UploadFile = None, return_convergence_delta: bool = Form(True), torchscript: bool = Form(False)):
    heatmaps = await lrp(data, model, architecture, return_convergence_delta, heatmap=True, torchscript=torchscript)
    if os.path.exists('heatmaps'):
        shutil.rmtree('heatmaps')
    os.makedirs('heatmaps')

    for index, i in enumerate(heatmaps):
        fig, ax = i
        fig.savefig(f'heatmaps/figure{index}.pdf')

    # create zip file
    with zipfile.ZipFile('heatmaps.zip', "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add all files in the directory to the zip
        for filename in os.listdir('heatmaps'):
            file_path = os.path.join('heatmaps', filename)
            zipf.write(file_path, arcname=filename)
    return FileResponse('heatmaps.zip')

@router.post('/occlusion')
async def occlusion(data: List[UploadFile], model: UploadFile, architecture: UploadFile = None, baselines: str = Form(None), sliding_window_shapes: Tuple[int, ...] = Form((3, 15, 15)), strides: Tuple[int, ...] = Form((3, 8, 8)), perturbations_per_eval: int = 1, heatmap: bool = Form(False), torchscript: bool = Form(False)):

    if architecture is None:  # model is uploaded via a single .onnx file
        file_path = await save_uploaded_file(model)
        if torchscript:
            file_path_with_extension = file_path + ".pt"
        else:
            file_path_with_extension = file_path + ".onxx"
        os.rename(file_path, file_path_with_extension)
        model = file_path_with_extension

        if baselines is not None:
            baselines = json.loads(baselines)  # Deserialize the string representation back into a list
            if not isinstance(baselines, list):
                warnings.warn("Invalid baselines format. Expected a PyTorch tensor.")
            baselines = torch.tensor(baselines)  # convert the list to a tensor
        params = {'baselines': baselines,
                  'sliding_window_shapes': sliding_window_shapes,
                  'strides': strides,
                  'perturbations_per_eval': perturbations_per_eval}
        explanations = []
        predictions = []
        heatmaps = []
        for d in data:
            d.file.seek(0)
            result = await get_explanation(OcclusionExplainer, data=d, model=model, params=params, torchscript=torchscript)
            explanations.append(result[0].tolist())
            predictions.append(result[1].tolist())
            heatmaps.append(result[2])
        if not heatmap:
            return JSONResponse(
                content={'explanations': explanations, 'predictions': predictions})
        else:
            return heatmaps

    else: # model is uploaded via two files: a model statedict .pth file (the model parameter) and a architecture .py file
        file_path = await save_uploaded_file(architecture)
        file_path_with_extension = file_path + ".py"
        os.rename(file_path, file_path_with_extension)

        if baselines is not None:
            baselines = json.loads(baselines)  # Deserialize the string representation back into a list
            if not isinstance(baselines, list):
                warnings.warn("Invalid baselines format. Expected a PyTorch tensor.")
            baselines = torch.tensor(baselines)  # convert the list to a tensor
        params = {'baselines': baselines,
                  'sliding_window_shapes': sliding_window_shapes,
                  'strides': strides,
                  'perturbations_per_eval': perturbations_per_eval}
        explanations = []
        predictions = []
        heatmaps = []
        for d in data:
            d.file.seek(0)
            model.file.seek(0)
            result = await get_explanation_with_architecture(OcclusionExplainer, data=d, model_dic=model, architecture_path=file_path_with_extension, params=params)
            explanations.append(result[0].tolist())
            predictions.append(result[1].tolist())
            heatmaps.append(result[2])
        if not heatmap:
            return JSONResponse(
                content={'explanations': explanations, 'predictions': predictions})
        else:
            return heatmaps


@router.post('/occlusion_heatmap')
async def occlusion_heatmap(data: List[UploadFile], model: UploadFile, architecture: UploadFile = None, baselines: str = Form(None), sliding_window_shapes: Tuple[int, ...] = Form((3, 15, 15)), strides: Tuple[int, ...] = Form((3, 8, 8)), perturbations_per_eval: int = 1, torchscript: bool = Form(False)):
    heatmaps = await occlusion(data, model, architecture, baselines, sliding_window_shapes, strides, perturbations_per_eval, heatmap=True, torchscript=torchscript)
    if os.path.exists('heatmaps'):
        shutil.rmtree('heatmaps')
    os.makedirs('heatmaps')

    for index, i in enumerate(heatmaps):
        fig, ax = i
        fig.savefig(f'heatmaps/figure{index}.pdf')

    # create zip file
    with zipfile.ZipFile('heatmaps.zip', "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add all files in the directory to the zip
        for filename in os.listdir('heatmaps'):
            file_path = os.path.join('heatmaps', filename)
            zipf.write(file_path, arcname=filename)
    return FileResponse('heatmaps.zip')

@router.post('/gradient')
async def gradient(data: List[UploadFile], model: UploadFile, architecture: UploadFile = None, baselines: str = Form(None), n_steps: int = Form(50), method: str = Form('gausslegendre'), internal_batch_size: int = Form(None), multiply_by_inputs: bool = Form(True), return_convergence_delta: bool = Form(False), heatmap: bool = Form(False), torchscript: bool = Form(False)):

    if architecture is None:  # model is uploaded via a single .onnx file
        file_path = await save_uploaded_file(model)
        if torchscript:
            file_path_with_extension = file_path + ".pt"
        else:
            file_path_with_extension = file_path + ".onxx"
        os.rename(file_path, file_path_with_extension)
        model = file_path_with_extension

        if baselines is not None:
            baselines = json.loads(baselines)  # Deserialize the string representation back into a list
            if not isinstance(baselines, list):
                warnings.warn("Invalid baselines format. Expected a PyTorch tensor.")
            baselines = torch.tensor(baselines)  # convert the list to a tensor

        params = {'n_steps': n_steps,
                  'method': method,
                  'baselines': baselines,
                  'internal_batch_size': internal_batch_size,
                  'multiply_by_inputs': multiply_by_inputs,
                  'return_convergence_delta': return_convergence_delta}
        explanations = []
        predictions = []
        heatmaps = []
        for d in data:
            d.file.seek(0)
            result = await get_explanation(GradientExplainer, data=d, model=model, params=params, torchscript=torchscript)
            explanations.append(result[0].tolist())
            predictions.append(result[1].tolist())
            heatmaps.append(result[2])
        if not heatmap:
            return JSONResponse(
                content={'explanations': explanations, 'predictions': predictions})
        else:
            return heatmaps
    else: # model is uploaded via two files: a model statedict .pth file (the model parameter) and a architecture .py file
        file_path = await save_uploaded_file(architecture)
        file_path_with_extension = file_path + ".py"
        os.rename(file_path, file_path_with_extension)

        if baselines is not None:
            baselines = json.loads(baselines)  # Deserialize the string representation back into a list
            if not isinstance(baselines, list):
                warnings.warn("Invalid baselines format. Expected a PyTorch tensor.")
            baselines = torch.tensor(baselines)  # convert the list to a tensor

        params = {'n_steps': n_steps,
                  'method': method,
                  'baselines': baselines,
                  'internal_batch_size': internal_batch_size,
                  'multiply_by_inputs': multiply_by_inputs,
                  'return_convergence_delta': return_convergence_delta}
        explanations = []
        predictions = []
        heatmaps = []
        for d in data:
            d.file.seek(0)
            model.file.seek(0)
            result = await get_explanation_with_architecture(GradientExplainer, data=d, model_dic=model, architecture_path=file_path_with_extension, params=params)
            explanations.append(result[0].tolist())
            predictions.append(result[1].tolist())
            heatmaps.append(result[2])
        if not heatmap:
            return JSONResponse(
                content={'explanations': explanations, 'predictions': predictions})
        else:
            return heatmaps


@router.post('/gradient_heatmap')
async def gradient_heatmap(data: List[UploadFile], model: UploadFile, architecture: UploadFile = None, baselines: str = Form(None), n_steps: int = Form(50), method: str = Form('gausslegendre'), internal_batch_size: int = Form(None), multiply_by_inputs: bool = Form(True), return_convergence_delta: bool = Form(False), torchscript: bool = Form(False)):
    heatmaps = await gradient(data, model, architecture, baselines, n_steps, method, internal_batch_size, multiply_by_inputs, return_convergence_delta, heatmap=True, torchscript=torchscript)
    if os.path.exists('heatmaps'):
        shutil.rmtree('heatmaps')
    os.makedirs('heatmaps')

    for index, i in enumerate(heatmaps):
        fig, ax = i
        fig.savefig(f'heatmaps/figure{index}.pdf')

    # create zip file
    with zipfile.ZipFile('heatmaps.zip', "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add all files in the directory to the zip
        for filename in os.listdir('heatmaps'):
            file_path = os.path.join('heatmaps', filename)
            zipf.write(file_path, arcname=filename)
    return FileResponse('heatmaps.zip')

@router.post('/deconvnet')
async def deconvnet(data: List[UploadFile], model: UploadFile, architecture: UploadFile = None, heatmap: bool = Form(False), torchscript: bool = Form(False)):

    if architecture is None:  # model is uploaded via a single .onnx file
        file_path = await save_uploaded_file(model)
        if torchscript:
            file_path_with_extension = file_path + ".pt"
        else:
            file_path_with_extension = file_path + ".onxx"
        os.rename(file_path, file_path_with_extension)
        model = file_path_with_extension

        explanations = []
        predictions = []
        heatmaps = []
        for d in data:
            d.file.seek(0)
            result = await get_explanation(DeConvNetExplainer, data=d, model=model, params={}, torchscript=torchscript)
            explanations.append(result[0].tolist())
            predictions.append(result[1].tolist())
            heatmaps.append(result[2])
        if not heatmap:
            return JSONResponse(
                content={'explanations': explanations, 'predictions': predictions})
        else:
            return heatmaps
    else: # model is uploaded via two files: a model statedict .pth file (the model parameter) and a architecture .py file
        file_path = await save_uploaded_file(architecture)
        file_path_with_extension = file_path + ".py"
        os.rename(file_path, file_path_with_extension)

        explanations = []
        predictions = []
        heatmaps = []
        for d in data:
            d.file.seek(0)
            model.file.seek(0)
            result = await get_explanation_with_architecture(DeConvNetExplainer, data=d, model_dic=model, architecture_path=file_path_with_extension, params={})
            explanations.append(result[0].tolist())
            predictions.append(result[1].tolist())
            heatmaps.append(result[2])
        if not heatmap:
            return JSONResponse(
                content={'explanations': explanations, 'predictions': predictions})
        else:
            return heatmaps


@router.post('/deconvnet_heatmap')
async def deconvnet_heatmap(data: List[UploadFile], model: UploadFile, architecture: UploadFile = None, torchscript: bool = Form(False)):
    heatmaps = await deconvnet(data, model, architecture, heatmap=True, torchscript=torchscript)
    if os.path.exists('heatmaps'):
        shutil.rmtree('heatmaps')
    os.makedirs('heatmaps')

    for index, i in enumerate(heatmaps):
        fig, ax = i
        fig.savefig(f'heatmaps/figure{index}.pdf')

    # create zip file
    with zipfile.ZipFile('heatmaps.zip', "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add all files in the directory to the zip
        for filename in os.listdir('heatmaps'):
            file_path = os.path.join('heatmaps', filename)
            zipf.write(file_path, arcname=filename)
    return FileResponse('heatmaps.zip')

@router.post('/guidedgradcam')
async def guidedgradcam(data: List[UploadFile], model: UploadFile, architecture: UploadFile = None, layer: str = Form(None), device_ids: List[int] = Form(None), interpolate_mode: str = Form('nearest'), attribute_to_layer_input: bool = Form(False), heatmap: bool = Form(False), torchscript: bool = Form(False)):

    if architecture is None:  # model is uploaded via a single .onnx file
        file_path = await save_uploaded_file(model)
        if torchscript:
            file_path_with_extension = file_path + ".pt"
        else:
            file_path_with_extension = file_path + ".onxx"
        os.rename(file_path, file_path_with_extension)
        model = file_path_with_extension


        if layer is None:
            warnings.warn('Please define a layer!')
            return
        params = {'layer': layer,
                  'device_ids': device_ids,
                  'interpolate_mode': interpolate_mode,
                  'attribute_to_layer_input': attribute_to_layer_input}
        explanations = []
        predictions = []
        heatmaps = []
        for d in data:
            d.file.seek(0)
            result = await get_explanation(GuidedGradCAMExplainer, data=d, model=model, params=params, torchscript=torchscript)
            explanations.append(result[0].tolist())
            predictions.append(result[1].tolist())
            heatmaps.append(result[2])
        if not heatmap:
            return JSONResponse(
                content={'explanations': explanations, 'predictions': predictions})
        else:
            return heatmaps

    else:  # model is uploaded via two files: a model statedict .pth file (the model parameter) and a architecture .py file
        file_path = await save_uploaded_file(architecture)
        file_path_with_extension = file_path + ".py"
        os.rename(file_path, file_path_with_extension)

        if layer is None:
            warnings.warn('Please define a layer!')
            return
        params = {'layer': layer,
                  'device_ids': device_ids,
                  'interpolate_mode': interpolate_mode,
                  'attribute_to_layer_input': attribute_to_layer_input}
        explanations = []
        predictions = []
        heatmaps = []
        for d in data:
            d.file.seek(0)
            model.file.seek(0)
            result = await get_explanation_with_architecture(GuidedGradCAMExplainer, data=d, model_dic=model, architecture_path=file_path_with_extension, params=params)
            explanations.append(result[0].tolist())
            predictions.append(result[1].tolist())
            heatmaps.append(result[2])
        if not heatmap:
            return JSONResponse(
                content={'explanations': explanations, 'predictions': predictions})
        else:
            return heatmaps


@router.post('/guidedgradcam_heatmap')
async def guidedgradcam_heatmap(data: List[UploadFile], model: UploadFile, architecture: UploadFile = None, layer: str = Form(None), device_ids: List[int] = Form(None), interpolate_mode: str = Form('nearest'), attribute_to_layer_input: bool = Form(False), torchscript: bool = Form(False)):
    heatmaps = await guidedgradcam(data, model, architecture, layer, device_ids, interpolate_mode, attribute_to_layer_input, heatmap=True, torchscript=torchscript)
    if os.path.exists('heatmaps'):
        shutil.rmtree('heatmaps')
    os.makedirs('heatmaps')

    for index, i in enumerate(heatmaps):
        fig, ax = i
        fig.savefig(f'heatmaps/figure{index}.pdf')

    # create zip file
    with zipfile.ZipFile('heatmaps.zip', "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add all files in the directory to the zip
        for filename in os.listdir('heatmaps'):
            file_path = os.path.join('heatmaps', filename)
            zipf.write(file_path, arcname=filename)
    return FileResponse('heatmaps.zip')


@router.post('/semanticsegmentation')
async def semanticsegmentation(data: List[UploadFile], model: UploadFile, architecture: UploadFile = None, layer: str = Form(None), target: int = Form(None), heatmap: bool = Form(False), torchscript: bool = Form(False)):

    if architecture is None:  # model is uploaded via a single .onnx file
        file_path = await save_uploaded_file(model)
        if torchscript:
            file_path_with_extension = file_path + ".pt"
        else:
            file_path_with_extension = file_path + ".onxx"
        os.rename(file_path, file_path_with_extension)
        model = file_path_with_extension


        if layer is None:
            warnings.warn('Please define a layer!')
            return
        if target is None:
            warnings.warn('Please define a target!')
            return
        params = {'layer': layer,
                  'target': target}
        explanations = []
        predictions = []
        heatmaps = []
        for d in data:
            d.file.seek(0)
            result = await get_explanation(SemanticSegmentationExplainer, data=d, model=model, params=params, torchscript=torchscript)
            explanations.append(result[0].tolist())
            predictions.append(result[1].tolist())
            heatmaps.append(result[2])
        if not heatmap:
            return JSONResponse(
                content={'explanations': explanations, 'predictions': predictions})
        else:
            return heatmaps

    else:  # model is uploaded via two files: a model statedict .pth file (the model parameter) and a architecture .py file
        file_path = await save_uploaded_file(architecture)
        file_path_with_extension = file_path + ".py"
        os.rename(file_path, file_path_with_extension)

        if layer is None:
            warnings.warn('Please define a layer!')
            return
        if target is None:
            warnings.warn('Please define a target!')
            return
        params = {'layer': layer,
                  'target': target}
        explanations = []
        predictions = []
        heatmaps = []
        for d in data:
            d.file.seek(0)
            model.file.seek(0)
            result = await get_explanation_with_architecture(SemanticSegmentationExplainer, data=d, model_dic=model, architecture_path=file_path_with_extension, params=params)
            explanations.append(result[0].tolist())
            predictions.append(result[1].tolist())
            heatmaps.append(result[2])
        if not heatmap:
            return JSONResponse(
                content={'explanations': explanations, 'predictions': predictions})
        else:
            return heatmaps



@router.post('/semanticsegmentation_heatmap')
async def semanticsegmentation_heatmap(data: List[UploadFile], model: UploadFile, architecture: UploadFile = None, layer: str = Form(None), target: int = Form(None), torchscript: bool = Form(False)):
    heatmaps = await semanticsegmentation(data, model, architecture, layer, target, heatmap=True, torchscript=torchscript)
    if os.path.exists('heatmaps'):
        shutil.rmtree('heatmaps')
    os.makedirs('heatmaps')

    for index, i in enumerate(heatmaps):
        fig, ax = i
        fig.savefig(f'heatmaps/figure{index}.pdf')

    # create zip file
    with zipfile.ZipFile('heatmaps.zip', "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add all files in the directory to the zip
        for filename in os.listdir('heatmaps'):
            file_path = os.path.join('heatmaps', filename)
            zipf.write(file_path, arcname=filename)
    return FileResponse('heatmaps.zip')

async def save_uploaded_file(file: UploadFile):
    file_path = os.path.join("/tmp", file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return file_path

async def assemble_concept(name, id, concepts_path):
    def transform(img):
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )(img)
    def get_tensor_from_filename(filename):
        img = Image.open(filename).convert("RGB")
        return transform(img)

    concept_path = os.path.join(concepts_path, name) + "/"
    dataset = CustomIterableDataset(get_tensor_from_filename, concept_path)
    concept_iter = dataset_to_dataloader(dataset)
    return Concept(id=id, name=name, data_iter=concept_iter)


@router.post('/tcav')
# input zip file has to be named input_images.zip and concepts zip file has to be named concepts.zip!
async def tcav(input_images: UploadFile, concepts: UploadFile, model: UploadFile = None, architecture: UploadFile = None, layers: List[str] = Form(None), n_steps: int = 5):
    if layers is None:
        warnings.warn('Please define a layer!')
        return

    # set up input
    # delete directory if still existing from last tcav call
    if os.path.exists('/tmp/input_images'):
        shutil.rmtree('/tmp/input_images')
    # unzip input: store images as list of UploadFiles
    zip_path = await save_uploaded_file(input_images)
    zip_path_with_extension = zip_path + '.zip'
    os.rename(zip_path, zip_path_with_extension)
    with ZipFile(zip_path_with_extension, 'r') as zObject:
        zObject.extractall(zip_path)
    input_images_path = (os.path.join(zip_path, input_images.filename))
    filenames = glob.glob(f'{input_images_path}/*.jpg')
    tensors = []
    for filename in filenames:
        img = Image.open(filename).convert('RGB')
        tensors.append(img)

    def transform(img):
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )(img)
    input_tensors = torch.stack([transform(img) for img in tensors])


    # set up concepts
    # delete directory if still existing from last tcav call
    if os.path.exists('/tmp/concepts'):
        shutil.rmtree('/tmp/concepts')
    # unzip concepts: store files as list of UploadFiles
    zip_path = await save_uploaded_file(concepts)
    zip_path_with_extension = zip_path + '.zip'
    os.rename(zip_path, zip_path_with_extension)
    with ZipFile(zip_path_with_extension, 'r') as zObject:
        zObject.extractall(zip_path)
    concepts_list = os.listdir(os.path.join(zip_path, concepts.filename))
    if '.DS_Store' in concepts_list:  # mac automatically creates such a file
        concepts_list.remove('.DS_Store')
    concepts_dic = {}
    concepts_explainer_dic = {}
    for index, i in enumerate(concepts_list):
        concepts_dic[f'concept_{i}'] = await assemble_concept(i, index, concepts_path=os.path.join(zip_path, concepts.filename))
        concepts_explainer_dic[index] = i

    # set up model
    model = torchvision.models.googlenet(pretrained=True)
    model = model.eval()
    """
    if architecture is None:  # model is uploaded via a single .onnx file
        file_path = await save_uploaded_file(model)
        file_path_with_extension = file_path + ".onxx"
        os.rename(file_path, file_path_with_extension)
        model = file_path_with_extension
        model = convert(model)
        model = model.eval()
        print('Got model', file=sys.stderr)

    else:  # model is uploaded via statedict and architecture.py file
        model_dic = model  # in this case uploaded model is the state dict
        file_path = await save_uploaded_file(architecture)
        file_path_with_extension = file_path + ".py"
        os.rename(file_path, file_path_with_extension)
        # Load the models basic architecture
        # Import the Model class from the uploaded file
        spec = importlib.util.spec_from_file_location("architecture", file_path_with_extension)
        architecture_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(architecture_module)
        # Instantiate an object from the Model class
        model = architecture_module.Model()
        # Load the state_dict
        state_dict = model_dic.file.read()
        state_dict = torch.load(BytesIO(state_dict))
        # Load the state_dict into the model and set to evaluation mode
        model.load_state_dict(state_dict)
        model = model.eval()
        print('Got model', file=sys.stderr)
    """


    # set up tcav
    tcav = TCAV(model=model,
                  layers=layers,
                  layer_attr_method=LayerIntegratedGradients(
                      model, None, multiply_by_inputs=False))


    # create experimental sets as all possible pair combinations of the concepts
    keys = list(concepts_dic.keys())
    experimental_sets = [[concepts_dic[keys[i]], concepts_dic[keys[j]]] for i in range(len(keys)) for j in range(i + 1, len(keys))]

    # get model prediction for the input data
    predictions = []
    for i in range(len(input_tensors)):
        pred = model(input_tensors[i].unsqueeze(0))
        output = F.softmax(pred, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)  # returns value & index of top 1 label
        pred_label_idx.squeeze_()
        predictions.append(pred_label_idx.item())

    # get explanations for all the concepts pairs
    tcav_scores = tcav.interpret(inputs=input_tensors,
                                 experimental_sets=experimental_sets,
                                 target=predictions,
                                 n_steps=n_steps)

    def tensor_to_list(item):
        if isinstance(item, torch.Tensor):
            return item.tolist()
        elif isinstance(item, dict):
            return {key: tensor_to_list(value) for key, value in item.items()}
        return item

    def convert_dict(d):
        return {key: tensor_to_list(value) for key, value in d.items()}

    return JSONResponse(content={'concepts_to_id_explanation': concepts_explainer_dic, 'explanations': convert_dict(tcav_scores), 'predictions': predictions})



async def get_explanation(explainer, data: UploadFile, model, params, torchscript):
    fail_return_data = []
    try:
        # Load the model
        if torchscript:
            model = torch.jit.load(model)
        else:
            model = convert(model)

        model = model.eval()
        print('Got model', file=sys.stderr)

        # Load image data to use
        data = Image.open(BytesIO(data.file.read()))
        print('Got Data', file=sys.stderr)

        # Transform image data to tensor
        transform = transforms.ToTensor()
        data = transform(data)
        data = data.unsqueeze(0)

        # Start explainer and analyze input with set params
        explain = explainer()
        print('Use', explain.explainer_name)
        if explain.explainer_name == 'Semantic Segmentation':  # because onnx models can not be used with semantic segmentation at the moment
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = models.segmentation.fcn_resnet50(pretrained=True).to(device).eval()
        img, heatmap = explain.explain(model, data, params)
        print('Finished Explanation', file=sys.stderr)

        if explain.explainer_name == 'Semantic Segmentation':
            pred = model(data)['out']
        else:
            pred = model(data)
        print('Finished Predictions', file=sys.stderr)

        return [img, pred, heatmap]

    except ValueError:
        print("Value error:", sys.exc_info()[1], file=sys.stderr)
    except:
        print("Unexpected error:", sys.exc_info()[1], file=sys.stderr)
        # traceback.print_exc()

    return [fail_return_data[0], np.array([-1])]


async def get_explanation_with_architecture(explainer, data: UploadFile, model_dic: UploadFile, architecture_path: str, params):
    fail_return_data = []
    try:
        # Load the models basic architecture
        # Import the Model class from the uploaded file
        spec = importlib.util.spec_from_file_location("architecture", architecture_path)
        architecture_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(architecture_module)
        # Instantiate an object from the Model class
        model = architecture_module.Model()

        # Load the state_dict
        state_dict = model_dic.file.read()
        state_dict = torch.load(BytesIO(state_dict))

        # Load the state_dict into the model and set to evaluation mode
        model.load_state_dict(state_dict)
        model = model.eval()
        print('Got model', file=sys.stderr)

        # Load image data to use
        data = Image.open(BytesIO(data.file.read()))
        print('Got Data', file=sys.stderr)

        # Transform image data to tensor
        transform = transforms.ToTensor()
        data = transform(data)
        data = data.unsqueeze(0)

        # Start explainer and analyze input with set params
        explain = explainer()
        print('Use', explain.explainer_name)
        img, heatmap = explain.explain(model, data, params)
        print('Finished Explanation', file=sys.stderr)

        if explain.explainer_name == 'Semantic Segmentation':
            pred = model(data)['out']
        else:
            pred = model(data)
        print('Finished Predictions', file=sys.stderr)

        return [img, pred, heatmap]

    except ValueError:
        print("Value error:", sys.exc_info()[1], file=sys.stderr)
    except:
        print("Unexpected error:", sys.exc_info()[1], file=sys.stderr)
        # traceback.print_exc()

    return [fail_return_data[0], np.array([-1])]


