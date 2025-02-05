FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.1.0_3.10_cuda_12.1

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

ENV PJRT_DEVICE=CUDA

# Install pip requirements
RUN pip install scipy==1.11.4

# RUN pip install torchaudio==2.4.0 # this would load incompatible GPU/CUDA configuration
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install espnet_model_zoo==0.1.7
RUN pip install soundfile==0.12.1
RUN pip uninstall -y espnet
RUN pip install -U git+https://github.com/toby-fotherby/espnet-tts-with-xla.git

WORKDIR /app

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["sleep", "36000"]