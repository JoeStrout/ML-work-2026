#!/usr/bin/env python3
"""
CUDA Availability Diagnostic Script
Tests whether PyTorch can access CUDA and provides detailed information
"""

import sys
import torch

print("=" * 70)
print("CUDA AVAILABILITY DIAGNOSTIC")
print("=" * 70)

# 1. Basic CUDA availability
print("\n1. CUDA Available:", torch.cuda.is_available())

# 2. PyTorch version
print("\n2. PyTorch Version:", torch.__version__)

# 3. CUDA version that PyTorch was built with
print("\n3. PyTorch Built With CUDA:", torch.version.cuda)

# 4. cuDNN version
print("\n4. cuDNN Version:", torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A")

# 5. Number of CUDA devices
if torch.cuda.is_available():
    print("\n5. Number of CUDA Devices:", torch.cuda.device_count())
    
    # 6. Device details
    print("\n6. CUDA Device Details:")
    for i in range(torch.cuda.device_count()):
        print(f"   Device {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"      Compute Capability: {props.major}.{props.minor}")
        print(f"      Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"      Multi Processors: {props.multi_processor_count}")
    
    # 7. Current device
    print("\n7. Current CUDA Device:", torch.cuda.current_device())
    
    # 8. Simple tensor test
    print("\n8. Tensor Test:")
    try:
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = x + y
        print("   ✓ Successfully created tensors on CUDA and performed operation")
        print(f"   Result device: {z.device}")
    except Exception as e:
        print(f"   ✗ Failed to create/operate on CUDA tensors: {e}")
else:
    print("\n5-8. [SKIPPED - CUDA not available]")
    print("\nPOSSIBLE ISSUES:")
    print("  • PyTorch may not be installed with CUDA support")
    print("  • NVIDIA drivers may not be installed")
    print("  • CUDA toolkit version mismatch")
    print("\nTo check your system:")
    print("  • Run: nvidia-smi")
    print("  • Check PyTorch installation with: pip show torch")

print("\n" + "=" * 70)
print("SYSTEM INFORMATION")
print("=" * 70)
print(f"Python Version: {sys.version}")
print(f"Python Executable: {sys.executable}")

# Check if we can run nvidia-smi
print("\nTrying to run nvidia-smi...")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("\n" + result.stdout)
    else:
        print("nvidia-smi failed to run")
except FileNotFoundError:
    print("nvidia-smi not found in PATH")
except Exception as e:
    print(f"Could not run nvidia-smi: {e}")

print("=" * 70)
