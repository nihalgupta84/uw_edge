#File: models/test_wavelet_model.py

import torch
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis
from torch.utils.benchmark import Timer
import numpy as np
from models.wavelet_model_v1 import *

# ======================================================
# 1. Unit Test Framework for Module Validation
# ======================================================
class ModelValidator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.dummy_input = torch.randn(2, 3, 256, 256).to(device)  # Batch of 2x256x256 RGB
        
    def generate_module_summary(self, module, input_shape=None):
        """Generate detailed summary of module architecture and parameters"""
        if input_shape is None:
            x = self.dummy_input
        else:
            x = torch.randn(input_shape).to(self.device)
            
        # Count parameters
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        # Get module structure
        def get_module_structure(mod, prefix=''):
            structure = []
            for name, child in mod.named_children():
                child_name = f"{prefix}.{name}" if prefix else name
                params = sum(p.numel() for p in child.parameters())
                structure.append({
                    'name': child_name,
                    'type': child.__class__.__name__,
                    'params': params,
                    'shape': 'Dynamic' if isinstance(child, (nn.ModuleList, nn.Sequential)) else str(child)
                })
                structure.extend(get_module_structure(child, child_name))
            return structure
        
        # Forward pass for shape analysis
        module.eval()
        with torch.no_grad():
            try:
                out = module(x)
                if isinstance(out, tuple):
                    output_shape = tuple(o.shape if isinstance(o, torch.Tensor) else 'Non-tensor' for o in out)
                else:
                    output_shape = tuple(out.shape) if isinstance(out, torch.Tensor) else 'Non-tensor'
            except Exception as e:
                output_shape = f"Error: {str(e)}"
            
        module.train()
        
        # Safely get device
        try:
            device_type = next(module.parameters()).device.type
        except StopIteration:
            device_type = self.device  # Use validator's device if module has no parameters
        
        return {
            'module_name': module.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_shape': tuple(x.shape),
            'output_shape': output_shape,
            'structure': get_module_structure(module),
            'device': device_type,
            'has_parameters': total_params > 0
        }
    
    def print_module_summary(self, summary_dict):
        """Print formatted module summary"""
        print(f"\n{'='*50}")
        print(f"Module Summary: {summary_dict['module_name']}")
        print(f"{'='*50}")
        
        if summary_dict['has_parameters']:
            print(f"Total Parameters: {summary_dict['total_parameters']:,}")
            print(f"Trainable Parameters: {summary_dict['trainable_parameters']:,}")
        else:
            print("Note: This module has no trainable parameters")
            
        print(f"Input Shape: {summary_dict['input_shape']}")
        print(f"Output Shape: {summary_dict['output_shape']}")
        print(f"Device: {summary_dict['device']}")
        
        print("\nModule Structure:")
        print("-" * 50)
        for module in summary_dict['structure']:
            indent = '  ' * len(module['name'].split('.'))
            print(f"{indent}{module['name']} ({module['type']})")
            if module['params'] > 0:
                print(f"{indent}Parameters: {module['params']:,}")

    def validate_module(self, module, input_shape=None, custom_input=None):
        """Fundamental validation of module: shapes, gradients, numerics"""
        try:
            # Input preparation
            if custom_input is None:
                x = torch.randn(input_shape).to(self.device) if input_shape else self.dummy_input
            else:
                x = custom_input
            
            # Forward Pass
            out = module(x)
            
            # Handle modules with no parameters
            has_params = any(p.requires_grad for p in module.parameters())
            
            if has_params:
                # Backward Pass (only for modules with parameters)
                loss = out.sum() if isinstance(out, torch.Tensor) else sum(o.sum() for o in out if isinstance(o, torch.Tensor))
                loss.backward()
                
                # Parameter Update Check
                has_gradients = any(p.grad is not None and not torch.all(p.grad == 0) 
                                  for p in module.parameters())
            else:
                has_gradients = None
            
            # Numerical Stability
            if isinstance(out, torch.Tensor):
                nan_check = not (torch.isnan(out).any() or torch.isinf(out).any())
            else:
                nan_check = all(not (torch.isnan(o).any() or torch.isinf(o).any()) 
                              for o in out if isinstance(o, torch.Tensor))
            
            return {
                "input_shape": tuple(x.shape),
                "output_shape": tuple(out.shape) if isinstance(out, torch.Tensor) else "multiple outputs",
                "has_parameters": has_params,
                "gradient_flow": has_gradients,
                "numerical_stable": nan_check
            }
            
        except Exception as e:
            return {"error": str(e)}

    def benchmark_module(self, module):
        """Computational performance analysis"""
        # FLOPs Calculation
        flops = FlopCountAnalysis(module, self.dummy_input)
        
        # Timing Benchmark
        timer = Timer(
            stmt='module(x)',
            globals={'module': module, 'x': self.dummy_input},
            num_threads=torch.get_num_threads()
        )
        
        return {
            "flops": flops.total(),
            "activations": ActivationCountAnalysis(module, self.dummy_input).total(),
            "time_ms": timer.timeit(100).mean * 1000  # Avg over 100 runs
        }
    def test_module_registration(model):
        """Verify all modules are properly registered"""
        # Get all modules
        module_dict = dict(model.named_modules())
        
        # Check critical modules
        critical_modules = [
            'color_reduce',
            'spec_reduce',
            'qfeat',
            'cross_block1',
            'cross_block2',
            'detail_restorer',
            'spp',
            'final_out',
            'final_harmonizer'
        ]
        
        for module_name in critical_modules:
            assert hasattr(model, module_name), f"Missing module: {module_name}"
            assert module_name in module_dict, f"Module not registered: {module_name}"
        
        # Verify no dynamic creation
        def check_forward_creates_no_modules(model, x):
            modules_before = set(dict(model.named_modules()).keys())
            _ = model(x)
            modules_after = set(dict(model.named_modules()).keys())
            assert modules_before == modules_after, "New modules created during forward pass"
        
        # Test with dummy input
        x = torch.randn(1, 3, 64, 64)
        check_forward_creates_no_modules(model, x)
        
        return True
# ======================================================
# 2. Core Module Tests (Mathematical Validation)
# ======================================================
def test_quaternion_conv(validator):
    """Test complete quaternion pipeline including channel adaptation"""
    
    # First test the channel adapter
    adapter = QuaternionChannelAdapter().to(validator.device)
    
    # Create RGB input
    rgb_input = torch.randn(2, 3, 256, 256).to(validator.device)
    
    # Test adapter
    adapted = adapter(rgb_input)
    assert adapted.shape[1] == 4, f"Channel adapter should output 4 channels, got {adapted.shape[1]}"
    
    # Now test quaternion conv with adapted input
    qconv = QuaternionConv(
        in_channels=4,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1
    ).to(validator.device)
    
    # Process through quaternion conv
    with torch.enable_grad():
        out = qconv(adapted)
        loss = out.mean()
        loss.backward()
    
    # Validation
    has_gradients = any(p.grad is not None for p in qconv.parameters())
    numerical_stable = not (torch.isnan(out).any() or torch.isinf(out).any())
    
    return {
        "channel_adapter_test": {
            "input_shape": rgb_input.shape,
            "adapted_shape": adapted.shape
        },
        "quaternion_conv_test": {
            "input_shape": adapted.shape,
            "output_shape": out.shape,
            "gradient_flow": has_gradients,
            "numerical_stable": numerical_stable
        }
    }

def test_color_prior(validator):
    """Validate color correction physics"""
    module = MultiStageColorPrior().to(validator.device)
    results = validator.validate_module(module)
    
    # Physical Constraint Check: Output should be same size as input
    assert results['output_shape'] == tuple(validator.dummy_input.shape), "Color prior changed image dimensions"
    
    # Numerical Stability for Color Offset
    out = module(validator.dummy_input)
    color_offset = out - validator.dummy_input
    assert torch.abs(color_offset).max() < 1.0, "Color offset exceeds physical limits"
    
    return results

def test_wavelet_decomp(validator):
    """Validate frequency domain decomposition properties"""
    module = WaveletFFTDecomposition().to(validator.device)
    
    # Test input
    x = torch.randn(2, 3, 256, 256).to(validator.device)
    
    try:
        # Forward pass
        LL, LH, HL, HH, mag, phase = module(x)
        
        # Validate shapes
        assert LL.shape == (2, 3, 128, 128), f"Invalid LL shape: {LL.shape}"
        assert LH.shape == (2, 3, 128, 128), f"Invalid LH shape: {LH.shape}"
        assert HL.shape == (2, 3, 128, 128), f"Invalid HL shape: {HL.shape}"
        assert HH.shape == (2, 3, 128, 128), f"Invalid HH shape: {HH.shape}"
        
        # Validate FFT components
        assert mag.shape == x.shape, f"Invalid magnitude shape: {mag.shape}"
        assert phase.shape == x.shape, f"Invalid phase shape: {phase.shape}"
        
        # Validate energy conservation (approximately)
        input_energy = torch.sum(x**2)
        wavelet_energy = torch.sum(LL**2) + torch.sum(LH**2) + torch.sum(HL**2) + torch.sum(HH**2)
        relative_error = torch.abs(input_energy - wavelet_energy) / input_energy
        assert relative_error < 0.1, f"Energy not conserved, relative error: {relative_error}"
        
        return {
            "input_shape": x.shape,
            "output_shapes": {
                "LL": LL.shape,
                "LH": LH.shape,
                "HL": HL.shape,
                "HH": HH.shape,
                "magnitude": mag.shape,
                "phase": phase.shape
            },
            "energy_conservation": relative_error.item()
        }
        
    except Exception as e:
        return {"error": str(e)}

# ======================================================
# 3. Cross-Module Integration Tests
# ======================================================
def test_cross_attention(validator):
    """Test cross attention with appropriate dual inputs"""
    dim = 64
    module = CrossAttentionBlock(dim).to(validator.device)
    
    # Create paired test inputs
    x = torch.randn(2, dim, 64, 64).to(validator.device)
    y = torch.randn(2, dim, 64, 64).to(validator.device)
    
    # Test forward pass with gradient tracking
    with torch.enable_grad():
        out = module(x, y)
        loss = out.mean()
        loss.backward()
    
    # Analyze attention patterns
    attention_score = (out - y).abs().mean().item()
    
    # Validate parameters and gradients
    param_count = sum(p.numel() for p in module.parameters())
    grad_flow = all(p.grad is not None for p in module.parameters() if p.requires_grad)
    
    return {
        "input_shapes": {
            "x": x.shape,
            "y": y.shape
        },
        "output_shape": out.shape,
        "attention_score": attention_score,
        "parameters": param_count,
        "gradient_flow": grad_flow
    }

# ======================================================
# 4. Full Model Validation
# ======================================================

def validate_quaternion_convs():
    from inspect import signature
    required_params = ['in_channels', 'out_channels', 'kernel_size', 'stride']
    
    for name, module in WaveletModel.named_modules():
        if isinstance(module, QuaternionConv):
            sig = signature(module.__init__)
            assert all(p in sig.parameters for p in required_params), \
                f"QuaternionConv {name} missing parameters"
            
def test_fft_components(validator):
    """Test FFT branch functionality with proper frequency patterns"""
    model = WaveletModel(use_fft_branch=True).to(validator.device)
    
    # Create frequency-rich test input using numpy/math
    def create_test_pattern(size=256):
        x = torch.zeros(2, 3, size, size).to(validator.device)
        frequencies = torch.linspace(0, math.pi/4, size).to(validator.device)
        mesh_x, mesh_y = torch.meshgrid(frequencies, frequencies, indexing='ij')
        
        # Create different frequency patterns for each channel
        patterns = [
            torch.sin(2 * mesh_x) * torch.cos(2 * mesh_y),
            torch.sin(4 * mesh_x) * torch.sin(4 * mesh_y),
            torch.cos(3 * mesh_x + mesh_y)
        ]
        
        for b in range(2):
            for c in range(3):
                x[b, c] = patterns[c]
        return x
    
    # Ensure model is in eval mode for testing
    model.eval()
    test_input = create_test_pattern()
    
    # Track both forward and backward passes
    activation_counts = {}
    gradient_flows = {}
    
    def track_activation(name):
        def hook(module, input, output):
            activation_counts[name] = activation_counts.get(name, 0) + 1
            if module.training:
                grad_tensors = [p.grad for p in module.parameters() if p.requires_grad]
                gradient_flows[name] = any(g is not None and g.abs().sum() > 0 for g in grad_tensors)
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if 'fft_branch' in name:
            hooks.append(module.register_forward_hook(track_activation(name)))
    
    try:
        # Test training mode
        model.train()
        with torch.enable_grad():
            output = model(test_input)
            loss = output.mean()
            loss.backward()
        
        # Test eval mode
        model.eval()
        with torch.no_grad():
            eval_output = model(test_input)
        
        # Analyze frequency components
        input_fft = torch.fft.fft2(test_input)
        output_fft = torch.fft.fft2(output)
        
        # Compute frequency response
        input_magnitude = torch.abs(input_fft)
        output_magnitude = torch.abs(output_fft)
        
        # Analyze preserved frequency bands
        stacked = torch.stack([
            input_magnitude.view(-1),
            output_magnitude.view(-1)
        ], dim=0)  # shape [2, ...flattened...]

        corr_matrix = torch.corrcoef(stacked)  # shape [2, 2]
        overall_correlation = corr_matrix[0, 1].item()

        freq_analysis = {
            'low_freq_ratio': (input_magnitude[:, :, :32, :32].mean() / 
                             output_magnitude[:, :, :32, :32].mean()).item(),
            'high_freq_ratio': (input_magnitude[:, :, -32:, -32:].mean() / 
                              output_magnitude[:, :, -32:, -32:].mean()).item(),
            'overall_preservation': overall_correlation
        }
        
        return {
            'module_activations': activation_counts,
            'gradient_flows': gradient_flows,
            'frequency_analysis': freq_analysis,
            'shapes': {
                'input': test_input.shape,
                'output': output.shape,
                'eval_output': eval_output.shape
            }
        }
        
    finally:
        for hook in hooks:
            hook.remove()

def full_model_test():
    validator = ModelValidator()
    model = WaveletModel().to(validator.device)

    # Get model summary
    print("\n📊 Model Architecture Summary:")
    model_summary = validator.generate_module_summary(model)
    validator.print_module_summary(model_summary)

    modules = {
        "QuaternionConv": test_quaternion_conv,
        "ColorPrior": test_color_prior,
        "WaveletDecomp": test_wavelet_decomp,
        "CrossAttention": test_cross_attention,
        "FFTBranch": test_fft_components
    }
    
    results = {}
    for name, test in modules.items():
        try:
            results[name] = test(validator)
            print(f"✅ {name} passed")
            
            # Print component summary if applicable
            if name != "FFTBranch":  # FFTBranch is tested differently
                component = get_component_for_summary(model, name)
                if component is not None:
                    print(f"\n📊 {name} Component Summary:")
                    component_summary = validator.generate_module_summary(component)
                    validator.print_module_summary(component_summary)
                    
        except AssertionError as e:
            print(f"❌ {name} failed: {str(e)}")
            results[name] = {"error": str(e)}
    
    # Computational performance
    print("\n⚙️ Computational Performance:")
    bench = validator.benchmark_module(model)
    print(f"FLOPs: {bench['flops']/1e9:.2f} G")
    print(f"Inference Time: {bench['time_ms']:.2f} ms")
    print(f"Activations: {bench['activations']/1e6:.2f} Million")
    
    return results

def get_component_for_summary(model, component_name):
    """Helper function to get component for summary generation"""
    component_mapping = {
        "QuaternionConv": model.qfeat.qconv1,
        "ColorPrior": model.color_prior,
        "WaveletDecomp": model.spectral_decomp,
        "CrossAttention": model.cross_block1
    }
    return component_mapping.get(component_name)






class ModelTestSuite:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.activation_counts = {}
        self.hooks = []
    
    def _register_activation_hooks(self, model):
        """Register hooks to track module activations"""
        def make_hook(name):
            def hook(module, input, output):
                self.activation_counts[name] = self.activation_counts.get(name, 0) + 1
            return hook
        
        for name, module in model.named_modules():
            if any(key in name for key in ['fft_branch', 'norm', 'qfeat']):
                self.hooks.append(module.register_forward_hook(make_hook(name)))
    
    def _cleanup_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def test_full_model(self, model):
        """Comprehensive model testing"""
        model = model.to(self.device)
        test_input = torch.randn(2, 3, 256, 256).to(self.device)
        
        # Register activation tracking
        self._register_activation_hooks(model)
        
        try:
            # Training mode test
            model.train()
            output = model(test_input)
            loss = output.mean()
            loss.backward()
            
            # Evaluation mode test
            model.eval()
            with torch.no_grad():
                eval_output = model(test_input)
            
            # Collect results
            results = {
                'activation_counts': self.activation_counts,
                'train_output_shape': output.shape,
                'eval_output_shape': eval_output.shape,
                'has_gradients': all(p.grad is not None for p in model.parameters() if p.requires_grad),
                'module_activation_status': {
                    name: count > 0 
                    for name, count in self.activation_counts.items()
                }
            }
            
            # Check FFT branch specifically
            if hasattr(model, 'fft_branch'):
                fft_results = self._test_fft_branch(model, test_input)
                results['fft_analysis'] = fft_results
            
            # Check quaternion feature extractor
            if hasattr(model, 'qfeat'):
                qfeat_results = self._test_qfeat(model, test_input)
                results['qfeat_analysis'] = qfeat_results
            
            return results
            
        finally:
            self._cleanup_hooks()
    
    def _test_fft_branch(self, model, x):
        """Test FFT branch specifically"""
        # Ensure input is on same device as model
        freq_input = create_frequency_pattern(size=x.shape[-1]).to(self.device)  # Move to correct device
        
        with torch.no_grad():
            # Get FFT components
            _, _, _, _, mag, phase = model.spectral_decomp(freq_input)
            
            # Process through FFT branch
            if model.use_fft_branch:
                fft_output = model.fft_branch(mag, phase)
                
                # Analyze frequency preservation
                input_spectrum = torch.fft.fft2(freq_input)
                output_spectrum = torch.fft.fft2(fft_output)
                
                freq_preservation = {
                    'input_magnitude': torch.abs(input_spectrum).mean().item(),
                    'output_magnitude': torch.abs(output_spectrum).mean().item(),
                    'shape_maintained': fft_output.shape[-2:] == freq_input.shape[-2:]
                }
                
                return freq_preservation
        
        return None
    
    def _test_qfeat(self, model, x):
        """Test quaternion feature extractor"""
        with torch.no_grad():
            # Track intermediate shapes
            shapes = {}
            
            # Get channel adapter output
            adapted = model.qfeat.channel_adapter(x)
            shapes['post_adaptation'] = adapted.shape
            
            # Get normalization stats if available
            if hasattr(model.qfeat, 'get_activation_stats'):
                norm_stats = model.qfeat.get_activation_stats()
            else:
                norm_stats = None
            
            return {
                'shapes': shapes,
                'normalization_stats': norm_stats
            }

def create_frequency_pattern(size=256, frequencies=[2, 5, 10]):
    """Create input with known frequency components"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.zeros(2, 3, size, size).to(device)  # Create tensor on correct device
    for b in range(2):
        for c in range(3):
            for freq in frequencies:
                grid = torch.linspace(0, 1, size, device=device)  # Create linspace on correct device
                x[b, c] += torch.sin(2 * np.pi * freq * grid.view(1, -1))
                x[b, c] += torch.sin(2 * np.pi * freq * grid.view(-1, 1))
    return x / len(frequencies)

if __name__ == "__main__":
    print("\n🔍 Starting Enhanced Model Testing")
    
    # Initialize validators
    validator = ModelValidator()
    test_suite = ModelTestSuite()
    
    # Create model with explicit configuration
    model = WaveletModel(
        base_ch=64,
        use_fft_branch=True,
        use_subband_attn=True,
        deeper_detail=True
    ).to(validator.device)
    
    # 1. Model Architecture Summary
    print("\n📊 Model Architecture Summary:")
    model_summary = validator.generate_module_summary(model)
    validator.print_module_summary(model_summary)
    
    # 2. Component Testing
    test_results = {}
    
    print("\n🧪 Component Tests:")
    for name, test_fn in {
        "QuaternionConv": test_quaternion_conv,
        "ColorPrior": test_color_prior,
        "WaveletDecomp": test_wavelet_decomp,
        "CrossAttention": test_cross_attention,
        "FFTBranch": test_fft_components
    }.items():
        try:
            results = test_fn(validator)
            test_results[name] = results
            print(f"✅ {name}: Passed")
            
            # Print component details
            if isinstance(results, dict):
                for key, value in results.items():
                    if key not in ['error', 'structure']:
                        print(f"  • {key}: {value}")
        except Exception as e:
            print(f"❌ {name}: Failed - {str(e)}")
            test_results[name] = {"error": str(e)}
    
    # 3. Full Model Validation
    print("\n🔍 Full Model Validation:")
    model_test_results = test_suite.test_full_model(model)
    
    # Print activation status
    print("\nModule Activation Status:")
    for name, active in model_test_results['module_activation_status'].items():
        status = "✅" if active else "❌"
        print(f"{status} {name}")
    
    # 4. Performance Metrics
    print("\n⚙️ Performance Metrics:")
    bench = validator.benchmark_module(model)
    print(f"• FLOPs: {bench['flops']/1e9:.2f} G")
    print(f"• Inference Time: {bench['time_ms']:.2f} ms")
    print(f"• Activations: {bench['activations']/1e6:.2f} M")
    
    # Final validation
    all_passed = all(
        'error' not in results for results in test_results.values()
    )
    if all_passed:
        print("\n🎉 All components validated successfully!")
    else:
        print("\n⚠️ Some components require attention!")