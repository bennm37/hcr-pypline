# pymorphogen
Morphogen Profile Extraction In Python

# Installation
Clone into the repository \
```git clone https://github.com/bennm37/pymorphogen.git```\
If not already set up, set up a Cellpose Conda envirnoment following the instructions here (for mac)\
https://forum.image.sc/t/cellpose-on-macos-m1-pro-apple-silicon-arm64/68018/4 \
Activate the cellpose environment. \
```conda activate cellpose-env```\
Install the hcrp package and dependencies \
```pip install -e .```

For some reason, a Segfault is obtained unless Cellpose is the first import of the python process. So every script must start with 
```from cellpose.models import Cellpose```
<!-- 
# Setup 
Create a file in data called image location.\
```data/image_location```\
hcrp will then use to look for images.
   -->