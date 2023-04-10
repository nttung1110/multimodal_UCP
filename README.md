# Unified implementation for multimodal UCP

## Installation

```
conda create --name multimodal_ucp python==3.8
conda env update -n multimodal_ucp --file environment.yml

# These are installed independently for avoiding conflicts
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio===0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip install -r requirements.txt

```
