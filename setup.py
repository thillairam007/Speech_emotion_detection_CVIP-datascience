from setuptools import setup, find_packages

# Define project metadata
name = "Speech_emotion_detection"
version = "1.0.0"

# Specify project dependencies
install_requires = [
    "absl-py==1.4.0",
    "aiohttp==3.8.6",
    "albumentations==1.3.1",
    "astunparse==1.6.3",
    "async-timeout==4.0.3",
    "attrs==23.1.0",
    "audioread==3.0.1",
    "beautifulsoup4==4.11.2",
    "bleach==6.1.0",
    "certifi==2023.7.22",
    "cffi==1.16.0",
    "chardet==5.2.0",
    "charset-normalizer==3.3.0",
    "colorcet==3.0.1",
    "colorlover==0.3.0",
    "cryptography==41.0.4",
    "cycler==0.12.1",
    "decorator==4.4.2",
    "docutils==0.18.1",
    "entrypoints==0.4",
    "idna==3.4",
    "imageio==2.31.5",
    "importlib-metadata==6.8.0",
    "itsdangerous==2.1.2",
    "lxml==4.9.3",
    "numpy==1.23.5",
    "oauthlib==3.2.2",
    "packaging==23.2",
    "pandas==1.5.3",
    "pandas-gbq==0.17.9",
    "prometheus-client==0.17.1",
    "pyasn1==0.5.0",
    "pyasn1-modules==0.3.0",
    "pyparsing==3.1.1",
    "pyproj==3.6.1",
    "requests==2.31.0",
    "requests-oauthlib==1.3.1",
    "scikit-image==0.19.3",
    "scikit-learn==1.2.2",
    "scipy==1.11.3",
    "seaborn==0.12.2",
    "six==1.16.0",
    "text-unidecode==1.3",
    "urllib3==2.0.6",
]

# Define setup configuration
setup(
    name=name,
    version=version,
    packages=find_packages(),
    install_requires=install_requires,
)
