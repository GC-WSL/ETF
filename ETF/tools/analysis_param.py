# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from typing import Any, Dict
import torch
import torch.nn as nn
from mmengine.config import Config, DictAction
from mmengine.logging import print_log, MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.runner import Runner
from fvcore.nn import FlopCountAnalysis
from mmseg.registry import RUNNERS
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Count model parameters')
    parser.add_argument('config', help='model config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config settings')
    args = parser.parse_args()
    return args


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def compute_flops_with_fvcore(
    model: nn.Module, 
    inputs: Any, 
    estimate_backward: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Args:
        model: Model to be analyzed  
        inputs: Input tensor for the model (compatible with the x parameter of wrapped_model.forward)   
        estimate_backward: Whether to estimate backward propagation FLOPs (during training)
    
    Returns:
        dict: {
            'forward_flops': int,        
            'train_flops': int,          
            'params_total': int,         
            'params_trainable': int,     
            'per_layer': List[Dict],     
            'summary': Dict              
        }
    """
    

    class WrappedModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            
            return self.model(x,)
    
    wrapped_model = WrappedModel(model)
    

    flops_analyzer = FlopCountAnalysis(wrapped_model, inputs)
    total_forward_flops = flops_analyzer.total()
    flops_by_module = flops_analyzer.by_module() 
    

    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    

    layer_stats = []
    trainable_flops = 0
    frozen_flops = 0
    nonparam_flops = 0  
    
    
    for name, module in model.named_modules():

        if len(list(module.children())) > 0:
            continue
        

        module_flops = flops_by_module.get(name, 0)
        if module_flops == 0 and name not in flops_by_module:

            wrapped_name = f"model.{name}" if name else "model"
            module_flops = flops_by_module.get(wrapped_name, 0)
        

        has_params = len(list(module.parameters())) > 0
        requires_grad = any(p.requires_grad for p in module.parameters()) if has_params else False
        

        always_backward = isinstance(module, (
            nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.ELU, nn.Sigmoid, nn.Tanh,
            nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.Dropout
        ))
        
        is_trainable = requires_grad or (not has_params and always_backward)
        

        num_params = sum(p.numel() for p in module.parameters())
        num_trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

        if is_trainable:
            trainable_flops += module_flops
        elif has_params:  
            frozen_flops += module_flops
        else:  
            nonparam_flops += module_flops
        

        layer_stats.append({
            'name': name if name else 'root',
            'module_type': type(module).__name__,
            'flops': module_flops,
            'is_trainable': is_trainable,
            'has_parameters': has_params,
            'requires_grad': requires_grad,
            'num_params': num_params,
            'num_trainable_params': num_trainable_params,
            'always_backward': always_backward if not has_params else False
        })
    
    if estimate_backward:

        train_flops = (
            total_forward_flops +
            trainable_flops * 2 +    
            # frozen_flops * 1 +       
            nonparam_flops * 1       
        )
    else:
        train_flops = total_forward_flops
    
    summary = {
        'trainable_layers_flops': trainable_flops,
        'frozen_layers_flops': frozen_flops,
        'nonparam_layers_flops': nonparam_flops,
        'trainable_params': params_trainable,
        'frozen_params': params_total - params_trainable,
        'total_forward_flops': total_forward_flops,
        'estimated_train_flops': train_flops
    }
    
    return {
        'forward_flops': total_forward_flops,
        'train_flops': train_flops,
        'params_total': params_total,
        'params_trainable': params_trainable,
        'per_layer': layer_stats,
        'summary': summary
    }



def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    runner = Runner.from_cfg(cfg)
    model = runner.model
    model = revert_sync_batchnorm(model)
    model.eval()  


    input_tensor = torch.randn(1, 3, 512, 512).cuda()


    from fvcore.nn import FlopCountAnalysis, parameter_count


    def model_wrapper(x):
        return model(x, mode='loss')  

    try:
        summary = compute_flops_with_fvcore(model, input_tensor)
        gflops_inference = summary['forward_flops'] / 1e9
        gflops_training = summary['train_flops'] /1e9  
        mparams = summary['params_total'] / 1e6
    except Exception as e:
        print(f"Direct analysis failed: {e}")
        print("Trying with wrapper...")
        flops = FlopCountAnalysis(model_wrapper, input_tensor)

    params = parameter_count(model)

    total_params = params['']

    print(f"Params: {total_params / 1e6:.2f} M")


    total, trainable, frozen = count_parameters(model)
    logger = runner.logger

    msg = "\n" + "="*50 + "\n"
    msg += "MODEL PARAMETER AND FLOPs COUNT\n"
    msg += "="*50 + "\n"
    msg += f"Total Parameters     : {total:,} ({total / 1e6:.2f} M)\n"
    msg += f"Trainable Parameters : {trainable:,}\n"
    msg += f"Frozen Parameters    : {frozen:,}\n"
    msg += f"Inference FLOPs        : {gflops_inference:.2f} GFLOPs\n"
    msg += f"Training FLOPs   : {gflops_training:.2f} GFLOPs\n"
    msg += f"Model Type           : {cfg.model.type}\n"
    msg += "="*50

    logger.info(msg)


if __name__ == '__main__':
    main()