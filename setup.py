from setuptools import setup

setup(
    name="ale",
    packages=["ale"],
    install_requires=[
        'librosa >= 0.8',
        'torchaudio_augmentations==0.2.1', # for augmentation
        'numpy',
        'pandas',
        'einops',
        'sklearn',
        'wandb',
        'jupyter',
        'matplotlib',
        'omegaconf',
        'transformers',
        "tqdm",
        'visdom==0.1.8.9',
        'termcolor==1.1.0',
        'ftfy==6.1.1',
        "pytorch-ignite==0.3.0"
    ]
)