import setuptools

def readme():
    with open('README.md') as f:
        return f.read()

setuptools.setup(name='torchpropel',
      version='0.0.2',
      description='Probabilistic Parameteric Regression Loss (PROPEL)',
      long_description=readme(),
      long_description_content_type="text/markdown",
      classifiers=[
        'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
      ],
      keywords='regression probabilistic neural networks machine learning',
      url='http://github.com/masadcv/torchpropel',
      author='Muhammad Asad',
      author_email='muhammad.asad@kcl.ac.uk',
      license='BSD-3-Clause',
      packages=['torchpropel'],
      install_requires=[
          'torch',
      ],
      zip_safe=False)
