from setuptools import setup

setup(
    name='lym1d',
    version='0.1.0',
    description='Lyman-Alpha 1D power spectrum analysis',
    url='https://github.com/schoeneberg/lym1d',
    author='Nils Schöneberg',
    author_email='nils.science@gmail.com',
    packages=['lym1d'],
    package_dir={'':'src'},
    install_requires=[
                      'numpy',
                      'scipy',
                      'george'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
    ],
)
