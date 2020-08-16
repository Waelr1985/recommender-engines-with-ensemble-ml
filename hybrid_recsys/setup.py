
from setuptools import setup, find_packages

REQUIRED_PACKAGES = ['tensorflow-hub==0.8.0']

setup(name='trainer',
    version='0.1',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    description='Setup dependencies for trainer package.'
)
