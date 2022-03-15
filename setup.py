import setuptools

def readme():
    with open('README.md') as f:
        return f.read()

setuptools.setup(name='propel_loss',
      version='0.0.2',
      description='Probabilistic Parameteric Regression Loss (PROPEL)',
      long_description=readme(),
      classifiers=[
        'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
      ],
      keywords='regression probabilistic neural networks machine learning',
      url='http://github.com/masadcv/PROPEL',
      author='Muhammad Asad',
      author_email='muhammad.asad@kcl.ac.uk',
      license='BSD-3-Clause',
      packages=['propel_loss'],
      install_requires=[
          'torch',
      ],
      zip_safe=False)
