import os
import sys
import subprocess
import shutil
import platform
import requests
import ctypes

def run_command(cmd, env=None):
    print(f"[setup.py] Running command: {cmd}")
    result = subprocess.run(cmd, shell=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def download_file(url, destination):
    """
    Downloads a file from 'url' to 'destination' using requests.
    """
    print(f"[setup.py] Downloading from {url} to {destination}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(destination, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("[setup.py] Download complete.")

def check_cuda_installed():
    """
    Check if 'nvcc' is on PATH and can run. If so, we assume CUDA is installed.
    """
    try:
        subprocess.check_output(["nvcc", "--version"])
        print("[setup.py] Detected CUDA (nvcc).")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[setup.py] CUDA not found (nvcc is missing).")
        return False
def check_torch_installed():
    try:
        import torch
        print("[setup.py] Detected PyTorch.")
        return True
    except ImportError:
        print("[setup.py] PyTorch not installed or cannot be imported.")
        return False
def check_cudnn_installed():
    """
    Check if cuDNN is installed. For simplicity, we try to import a PyPI package
    (like 'nvidia-cudnn-cu11' or 'cudnn') referenced in requirements.txt.
    If that fails, we assume cuDNN is not installed.
    """
    #if torch not installed, install torch
    if not check_torch_installed():
        print("[setup.py] PyTorch not installed, installing torch + torchvision + torchaudio for CUDA 12.4...")
        run_command(f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    try:
        import torch
        cudnn_ver = torch.backends.cudnn.version()
        if cudnn_ver is None:
            print("[setup.py] PyTorch says cuDNN is not available or not configured.")
            return False
        else:
            print(f"[setup.py] Found cuDNN via PyTorch. Version = {cudnn_ver}")
            return True
    except ImportError:
        print("[setup.py] PyTorch not installed or cannot be imported. cuDNN check failed.")
        return False

def clone_dlib_repo(repo_url="https://github.com/davisking/dlib.git", destination="dlib_src"):
    """
    Clones the dlib GitHub repository if not already present.
    """
    if os.path.exists(destination):
        print(f"[setup.py] dlib_src directory already exists, skipping clone.")
        return
    print(f"[setup.py] Cloning dlib repository from {repo_url} ...")
    run_command(f"git clone {repo_url} {destination}")

def build_dlib(with_cuda=True, dlib_path="dlib_src"):
    """
    Builds dlib from source, optionally with GPU support, depending on with_cuda.
    """
    build_mode = "GPU" if with_cuda else "CPU"
    print(f"[setup.py] Building dlib with {build_mode} support...")

    original_dir = os.getcwd()
    try:
        os.chdir(dlib_path)

        # dlib’s own setup.py no longer accepts these as command arguments.
        # They’re automatically enabled if supported on your system.
        # So just call “install” with no extra flags:
        uninstall_cmd = f"{sys.executable} -m pip uninstall dlib -y"
        run_command(uninstall_cmd)
        cmd_install = f"{sys.executable} setup.py install"
        run_command(cmd_install)

    finally:
        os.chdir(original_dir)

def install_cuda_windows():
    """
    Example function to download and silently install CUDA on Windows.
    Skips download & install if nvcc is already present.
    """
    # 1) Check if nvcc is present (i.e., CUDA is already installed)
    try:
        subprocess.check_output(["nvcc", "--version"])
        print("[setup.py] It appears CUDA is already installed (nvcc found). Skipping download & install.")
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass  # No nvcc found, proceed with install

    # 2) If no CUDA, proceed: download & install
    cuda_url = (
        "https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/"
        "cuda_12.4.0_551.61_windows.exe"
    )
    installer_name = "cuda_installer.exe"

    if os.path.exists(installer_name):
        print("[setup.py] Found existing cuda_installer.exe, skipping download.")
    else:
        print("[setup.py] No existing cuda_installer.exe found; downloading.")
        download_file(cuda_url, installer_name)
        print("[setup.py] CUDA installer downloaded.")

    print("[setup.py] Running silent install of CUDA...")
    run_command(f"{installer_name} /S")
    print("[setup.py] CUDA installer finished.")
    
def install_cudnn_windows():
    # (edit_1) If cuDNN is recognized by Python, skip
    if check_cudnn_installed():
        print("[setup.py] cuDNN is already installed, skipping installation.")
        return

    print("[setup.py] Installing cuDNN on Windows...")
    cudnn_url = "https://developer.download.nvidia.com/compute/cudnn/9.6.0/local_installers/cudnn_9.6.0_windows.exe"
    cudnn_installer_name = "cudnn_installer.exe"
    
    if os.path.exists(cudnn_installer_name):
        print("[setup.py] Found existing cudnn_installer.exe, skipping download.")
    else:
        print("[setup.py] No existing cudnn_installer.exe found; downloading.")
        download_file(cudnn_url, cudnn_installer_name)
        print("[setup.py] cuDNN installer downloaded.")

    print("[setup.py] Copying cuDNN to CUDA toolkit bin folder...")
    src_bin = r"C:\Program Files\NVIDIA\cudnn\v9.6\bin\11.8"
    dest_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"

    if os.path.exists(src_bin):
        os.makedirs(dest_bin, exist_ok=True)
        for filename in os.listdir(src_bin):
            shutil.copy2(os.path.join(src_bin, filename), os.path.join(dest_bin, filename))
        print("[setup.py] cuDNN files copied successfully!")
    else:
        print(f"[setup.py] Source cuDNN path not found: {src_bin}")

def ensure_windows_admin():
    """
    Checks if running on Windows and if the process is elevated.
    If not elevated, attempt to re-run with Administrator in
    a new cmd.exe window that remains open after the script finishes.
    """
    if platform.system().lower().startswith("win"):
        try:
            # IsUserAnAdmin returns a non-zero value if admin
            if not ctypes.windll.shell32.IsUserAnAdmin():
                print("[setup.py] Not running with admin privileges. Attempting elevation.")
                # Re-launch the script with Administrator rights
                script = os.path.abspath(sys.argv[0])
                # Capture original arguments
                params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
                
                # Launch an elevated cmd.exe window using /k to keep it open
                # so that the window doesn't close automatically after finishing
                command = (
                    f'powershell -Command "Start-Process cmd -ArgumentList'
                    f' \'/k \"{sys.executable}\" \"{script}\" {params}\''
                    f' -Verb RunAs"'
                )
                
                run_command(command)
                # Exit the current (non-elevated) process
                sys.exit(0)
        except Exception as e:
            print(f"[setup.py] Warning: could not check or elevate admin rights: {e}")

def main():
    ensure_windows_admin()

    print("[setup.py] Starting setup...")
    current_os = platform.system().lower()
    print(f"[setup.py] Detected OS: {current_os}")

    if current_os.startswith("win"):
        install_cuda_windows()
        install_cudnn_windows()
    else:
        print("[setup.py] On non-Windows OS, please modify code to install CUDA/cuDNN or skip auto-install.")

    # Check if CUDA + cuDNN is installed
    cuda_ok = check_cuda_installed()
    cudnn_ok = check_cudnn_installed()
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    run_command(f"{sys.executable} -m pip install --upgrade certifi")
    # 1) Clone dlib
    clone_dlib_repo()

    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    run_command(f"{sys.executable} -m pip install --force-reinstall -r \"{requirements_path}\"")

    # 2) Build dlib. If both CUDA & cuDNN are installed, build with GPU support
    if cuda_ok and cudnn_ok:
        print("[setup.py] Building dlib with GPU support...")
        build_dlib(with_cuda=True)
    else:
        print("[setup.py] Building dlib without GPU support...")
        build_dlib(with_cuda=False)

    print("[setup.py] dlib build complete!")
    
    # Install Python dependencies (including cuDNN if listed) from requirements.txt

    # (edit_1) Wait for user input so window won't close immediately after finishing
    input("\nPress Enter to exit setup.py...")

if __name__ == "__main__":
    main()
