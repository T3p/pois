from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))


setup(name='baselines',
      packages=[package for package in find_packages()
                if package.startswith('baselines')],
      install_requires=[
	  'gym[classic_control]',
          'scipy',
          'tqdm',
          'joblib',
          'zmq',
          'dill',
          'azure==1.0.3',
          'progressbar2',
          'mpi4py',
          'path.py',
          'theano',
          'cached_property',
          'Box2D',
          'mako',
          'pygame'
      ],
      description="POIS implementation based on openai/baselines",
      author="A. M. Metelli, M. Papini, N. Montali, F. Faccio, M. Restelli",
      url='https://github.com/T3p/pois',
      author_email="matteo.papini@polimi.it",
      version="0.1.1")
