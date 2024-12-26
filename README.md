# Face Recognition System

This repository provides a facial recognition pipeline built in Python. It includes:
1. A backend engine for face detection, recognition, and classification (in engine.py).
2. A GUI (in ui.py) for convenient, interactive usage.
3. A setup script (in setup.py) for installing CUDA/CuDNN on Windows systems and building dependencies such as dlib with optional GPU support.

Below is a detailed overview of the system's purpose, how to install and run the software, hardware/software prerequisites, and more.

---

## Table of Contents
1. [Purpose](#purpose)
2. [Key Features](#key-features)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
    - [1) Install Dependencies](#1-install-dependencies)
    - [2) Optional GPU Setup on Windows](#2-optional-gpu-setup-on-windows)
    - [3) Clone and Build dlib](#3-clone-and-build-dlib)
    - [4) Python Packages Installation](#4-python-packages-installation)
    - [5) Configure Known Images / Dataset Paths](#5-configure-known-images--dataset-paths)
5. [Usage](#usage)
    - [Starting the GUI](#starting-the-gui)
    - [Processing Images](#processing-images)
    - [Output Structure](#output-structure)
6. [How It Works](#how-it-works)
    - [1) GUI](#1-gui)
    - [2) Recognition Engine](#2-recognition-engine)
7. [Tips](#tips)
8. [Troubleshooting](#troubleshooting)
9. [License](#license)

---

## Purpose
The purpose of this system is to:
• Recognize faces in images.  
• Categorize them into matching identities or an "unknown" group.  
• Handle both single-face and multi-face images.  
• (Optional) Build and run with GPU acceleration on Windows using CUDA/CuDNN for faster inference.

---

## Key Features
• Single-button start in a user-friendly GUI.  
• Capability to dynamically blur non-primary faces in multiface images.  
• Automatic classification with a threshold for face matching.  
• Optionally build and run dlib with CUDA for faster face recognition if a compatible GPU is present.  
• Organized output directories for recognized and unknown faces.  
• Real-time progress updates, including intermediate thumbnail previews.

---

## System Requirements
1. Python 3.8+ recommended.
2. Operating System:
   - Windows 10 or newer if you plan on using GPU auto-install features.
   - Linux/macOS can be used, but manual installation of CUDA/CuDNN or skipping GPU steps is expected.
3. Adequate CPU/RAM resources to process images, especially if dealing with large datasets.
4. For GPU acceleration:
   - Compatible NVIDIA GPU.
   - Up-to-date GPU drivers.
   - (Optional) Automatic download and install of CUDA 12.x on Windows.

---

## Installation

### 1) Install Dependencies
• Make sure Python 3 is installed:  
  - On Windows, you can download from the official Python website.
  - On Linux/macOS, use your package manager (e.g., apt-get or brew).

• Install pip if not present:
  - pip usually comes with Python by default.  
  - Verify via:  
    ```
    python -m ensurepip --upgrade
    ```

### 2) Optional GPU Setup on Windows
If you are on Windows and want to use GPU acceleration:
1. Run the following from an **Administrator** shell (or let the script prompt you to elevate):
   ```
   python setup.py
   ```
2. The script attempts to:
   - Check if CUDA is installed, if not, download and install CUDA 12.4.  
   - Check if cuDNN is installed, if not, download a matching version and copy relevant libraries.  
   - Build dlib with GPU support if it detects CUDA and cuDNN.

### 3) Clone and Build dlib
If not using the automated step from the setup script:
• The repository includes a small function that clones the official [dlib GitHub repo](https://github.com/davisking/dlib).  
• The build process is triggered automatically in setup.py, but you can skip if you already have dlib installed or prefer CPU-only usage.

### 4) Python Packages Installation
• For base usage (without the setup script), install Python dependencies:
  ```
  pip install -r requirements.txt
  ```
• Or manually if you skip the provided requirements.txt:
  - psutil
  - face_recognition
  - pillow (for Image/PIL manipulation)
  - numpy
  - tkinter (usually included on Windows for Python, or else install through system package manager)
  - etc.

### 5) Configure Known Images / Dataset Paths
• Inside the GUI, you'll specify:
  - Dataset folder (the folder containing images you want to classify).
  - A known images folder (containing references of known individuals).
  - An output base folder (where recognized or unknown faces are copied).

---

## Usage

### Starting the GUI
1. Make sure all dependencies are installed.  
2. Launch the main GUI:
   ```
   python main.py
   ```
3. You'll see a window with fields to enter or browse for the dataset folder, known folder, and output location.

### Processing Images
1. In the GUI:
   - "Dataset" = the folder containing images to be recognized.  
   - "Known Folder" = the folder containing known images (one face per image recommended). The name of each file (without extension) is taken as the identity label.  
   - "Output Base" = the main location to store results.  
   - Adjust the threshold slider to decide how strict or loose the recognition matching should be.  
2. Click "Start" to begin. The progress bar updates as images are processed.  
3. Real-time thumbnail previews appear as each image is done.

### Output Structure
When the engine processes images, it organizes them like so under "Output Base":
• "unknown" folder - for unrecognized or corrupted images.  
• "output_group" folder - for pictures containing multiple faces, but at least one recognized identity.  
• Sub-folders named by recognized identity for single or dominant faces.  

---

## How It Works

### 1) GUI
The GUI (ui.py) allows you to:
• Select folders for input, known references, and output.  
• Set a threshold that influences whether an unknown face is recognized as a match.  
• Watch progress and results in real-time.

### 2) Recognition Engine
Powered by the face_recognition library:
• Each known image is loaded, encoded, and stored in an array for comparison.  
• Each target image is opened via PIL, checked for corruption, resized, then processed with face_recognition.  
• The engine detects faces, attempts to find the closest known match, and blurs other faces if multiple.  
• The recognized identity (if any) is used to sort/copy the original file into the correct folder.

---

## Tips
• GPU usage can dramatically speed up face recognition if you have a supported NVIDIA card.  
• Lower the unknown threshold if you see too many false positives.  
• Increase the threshold if you see too many unknowns of people you do have references for.

---

## Troubleshooting
1. "No module named ...": Make sure you ran "pip install -r requirements.txt".  
2. Windows GPU driver or admin permission issues: See README or run setup.py from an elevated shell.  
3. If face_recognition fails to install on older systems, upgrade pip or Python.  
4. Dlib build issues (on Linux/macOS):
   - Check you have necessary build tools (cmake, make, gcc, etc.).  
   - Install dependencies (like X11 dev libraries) for X11/GUI.  
5. Corrupted images:
   - The system tries to identify and move them to an "unknown/corrupted_files" folder.

---

## License
This project is distributed under the MIT License.
