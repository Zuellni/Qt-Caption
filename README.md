# Qt Caption
A simple image captioning GUI using [Florence-2](https://huggingface.co/collections/microsoft/florence-6669f44df0d87d9c3bfb76de).

## Installation
Create a new environment with mamba:
```bat
mamba create -n qt-caption git python pytorch pytorch-cuda torchvision -c conda-forge -c nvidia -c pytorch
mamba activate qt-caption
```

Clone the repository and install requirements:
```bat
git clone https://github.com/zuellni/qt-caption
cd qt-caption
pip install -r requirements.txt
```

## Usage
Start the GUI with:
```bat
python . -c config.json
```

Or with a script like this to hide the console:
```bat
@echo off
call mamba activate qt-caption
start pythonw . -c config.json
```

## Preview
![Preview](assets/preview.png)
