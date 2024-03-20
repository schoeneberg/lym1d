from setuptools import setup

setup(name="lym1d",
      description='Lyman-Alpha 1D power spectrum analysis',
      author='Nils Schöneberg',
      author_email='nils.science@gmail.com',
      url="https://github.com/schoeneberg/lym1d",
      install_requires=["numpy", "scipy", "george"],
      packages=['lym1d'],
      package_dir={'':'src'},
      readme="README.md",
      setuptools_git_versioning={'enabled':True},
      setup_requires=["setuptools","setuptools-git-versioning"],
      classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8']
      )
