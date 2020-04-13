"""Setup evaler."""
from setuptools import setup, find_packages

descr = ('Evaler.')

DISTNAME = 'evaler'
DESCRIPTION = descr
MAINTAINER = 'John GW Samuelsson'
MAINTAINER_EMAIL = 'johnsam@mit.edu'
VERSION = '0.1.dev0'

if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          version=VERSION,
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
          platforms='any',
          packages=find_packages()
          )
