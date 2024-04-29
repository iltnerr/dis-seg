# important: use a separate venv for gsam, because it seems to be incompatible otherwise.

export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda/

python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
pip install --upgrade diffusers[torch]
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel seaborn
