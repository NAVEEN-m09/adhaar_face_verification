import os
import sys

def setup_gpu_dlls():
    """
    Registers python-loaded NVIDIA DLL directories for onnxruntime-gpu on Windows.
    This ensures that CUDAExecutionProvider finds necessary DLL dependencies.
    """
    if sys.platform == "win32":
        py_ver = f"Python{sys.version_info.major}{sys.version_info.minor}"
        app_data = os.environ.get("APPDATA")
        if app_data:
            user_site = os.path.join(app_data, "Python", py_ver, "site-packages")
            nvidia_packages = ["cudnn", "cublas", "cuda_nvrtc"]
            for package in nvidia_packages:
                bin_dir = os.path.join(user_site, "nvidia", package, "bin")
                if os.path.exists(bin_dir):
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
                    try:
                        os.add_dll_directory(bin_dir)
                    except Exception:
                        pass
