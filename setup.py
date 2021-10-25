"""Setup evaler."""

import sys

import os
import os.path as op

from setuptools import setup, find_packages

# Give setuptools a hint to complain if it's too old a version
# 30.3.0 allows us to put most metadata in setup.cfg
# 38.3.0 contains most setup.cfg bugfixes
# Should match pyproject.toml
SETUP_REQUIRES = ["setuptools >= 38.3.0"]
# This enables setuptools to install wheel on-the-fly
SETUP_REQUIRES += ["wheel"] if "bdist_wheel" in sys.argv else []

# get the version (don't import mne here, so dependencies are not needed)
version = None
with open(op.join('evaler', '_version.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

descr = ('Evaler.')

DISTNAME = 'evaler'
DESCRIPTION = descr
MAINTAINER = 'John GW Samuelsson'
MAINTAINER_EMAIL = 'johnsam@mit.edu'
URL = 'https://github.com/johnsam7/evaler'
LICENSE = 'MIT License'
DOWNLOAD_URL = 'https://github.com/johnsam7/evaler'
VERSION = version

if __name__ == "__main__":
    if op.exists('MANIFEST'):
        os.remove('MANIFEST')

    with open('README.md', 'r') as fid:
        long_description = fid.read()

    install_requires = list()
    with open('requirements.txt', 'r') as fid:
        for line in fid:
            req = line.strip()
            install_requires.append(req)
            
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=long_description,
          long_description_content_type='text/x-rst',
          include_package_data=True,
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
          ],
          keywords='neuroscience neuroimaging MEG EEG brain',
          project_urls={
          'Source': 'https://github.com/johnsam7/evaler/',
          'Tracker': 'https://github.com/johnsam7/evaler/issues/',
          },
          platforms='any',
          python_requires='>=3.7',
          install_requires=install_requires,
          setup_requires=SETUP_REQUIRES,
          packages=find_packages()
          )
