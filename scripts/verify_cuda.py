"""Verify that the server's PyTorch can execute a kernel on the actual GPU."""
import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Run this with .venv-server/bin/python after server setup.")
        return 1
    print(f"PyTorch: {torch.__version__}; built for CUDA: {torch.version.cuda}")
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {available}")
    if not available:
        print("CUDA is unavailable to PyTorch. Check nvidia-smi inside Ubuntu and the CUDA PyTorch wheel.")
        return 1
    try:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
        value = (torch.ones(32, device="cuda") * 2).sum().item()
        torch.cuda.synchronize()
        if value != 64:
            print(f"Unexpected CUDA computation: {value}")
            return 1
        print(f"CUDA tensor sum: {value}")
    except RuntimeError as exc:
        print(f"CUDA kernel execution failed: {exc}")
        return 1
    print("CUDA tensor computation passed. local-jev automatically selects CUDA when available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
