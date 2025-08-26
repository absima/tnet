from setuptools import setup, find_packages

setup(
    name='tnet_analysis',
    version='1.0.1',
    packages=find_packages(include=['tnet_analysis', 'tnet_analysis.*']),
    install_requires=[
        'numpy',
        # 'networkx',
        'scikit-learn',
        # 'python-louvain',
    ],
)