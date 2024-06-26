from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['absl-py==2.1.0', 'aiohttp==3.9.5', 'aiosignal==1.3.1', 'astunparse==1.6.3', 'attrs==23.2.0', 'awscli==1.32.88', 'boto3==1.34.88', 'botocore==1.34.88', 'certifi==2024.2.2', 'cffi==1.16.0', 'charset-normalizer==3.3.2', 'click==8.1.7', 'cloudpickle==2.2.1', 'colorama==0.4.4', 'contextlib2==21.6.0', 'contourpy==1.2.1', 'cryptography==42.0.5', 'cycler==0.12.1', 'datasets==2.19.0', 'dill==0.3.8', 'docker==7.0.0', 'docutils==0.16', 'evaluate==0.4.1', 'filelock==3.13.4', 'flatbuffers==24.3.25', 'flwr==1.8.0', 'flwr-datasets==0.1.0', 'fonttools==4.51.0', 'frozenlist==1.4.1', 'fsspec==2024.3.1', 'gast==0.5.4', 'google-pasta==0.2.0', 'grpcio==1.62.2', 'h5py==3.11.0', 'huggingface-hub==0.22.2', 'idna==3.7', 'importlib-metadata==6.11.0', 'iterators==0.0.2', 'Jinja2==3.1.3', 'jmespath==1.0.1', 'joblib==1.4.0', 'jsonschema==4.21.1', 'jsonschema-specifications==2023.12.1', 'keras==3.2.1', 'kiwisolver==1.4.5', 'libclang==18.1.1', 'Markdown==3.6', 'markdown-it-py==3.0.0', 'MarkupSafe==2.1.5', 'matplotlib==3.8.4', 'mdurl==0.1.2', 'ml-dtypes==0.3.2', 'mpmath==1.3.0', 'msgpack==1.0.8', 'multidict==6.0.5', 'multiprocess==0.70.16', 'namex==0.0.8', 'networkx==3.3', 'nltk==3.8.1', 'numpy==1.26.4', 'OpenAttack==2.1.1', 'opt-einsum==3.3.0', 'optree==0.11.0', 'packaging==24.0', 'pandas==2.2.2', 'pathos==0.3.2', 'pillow==10.3.0', 'platformdirs==4.2.0', 'pox==0.3.4', 'ppft==1.7.6.8', 'protobuf==4.25.3', 'psutil==5.9.8', 'pyarrow==15.0.2', 'pyarrow-hotfix==0.6', 'pyasn1==0.6.0', 'pycparser==2.22', 'pycryptodome==3.20.0', 'pydantic==1.10.15', 'Pygments==2.17.2', 'pyparsing==3.1.2', 'python-dateutil==2.9.0.post0', 'pytz==2024.1', 'PyYAML==6.0.1', 'ray==2.6.3', 'referencing==0.34.0', 'regex==2024.4.16', 'requests==2.31.0', 'responses==0.18.0', 'rich==13.7.1', 'rpds-py==0.18.0', 'rsa==4.7.2', 's3transfer==0.10.1', 'safetensors==0.4.3', 'sagemaker==2.216.0', 'schema==0.7.5', 'shellingham==1.5.4', 'six==1.16.0', 'smdebug-rulesconfig==1.0.1', 'sympy==1.12', 'tblib==3.0.0', 'tensorboard==2.16.2', 'tensorboard-data-server==0.7.2', 'tensorflow==2.16.1', 'tensorflow-io-gcs-filesystem==0.36.0', 'termcolor==2.4.0', 'tf_keras==2.16.0', 'tokenizers==0.19.1', 'tomli==2.0.1', 'torch==2.2.2', 'tqdm==4.66.2', 'transformers==4.40.0', 'typer==0.9.4', 'typing_extensions==4.11.0', 'tzdata==2024.1', 'urllib3==2.2.1', 'Werkzeug==3.0.2', 'wrapt==1.16.0', 'xxhash==3.4.1', 'yarl==1.9.4', 'zipp==3.18.1']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application.'
)

