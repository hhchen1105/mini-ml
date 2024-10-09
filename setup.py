from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='miniml',
      version='0.1',
      description='The miniml package implements key machine learning algorithms.',
      long_description='The minimal package provides a backbone implementation of the key machine learning algorithms for educational purposes. We focus on code readability over efficiency.',
      classifiers=[
          'Development Status :: 1 - Planning',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.12',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      keywords='machine learning',
      url='https://github.com/hhchen1105/mini-ml/',
      author='Hung-Hsuan Chen',
      author_email='hhchen1105@gmail.com',
      license='MIT',
      packages=['miniml'],
      install_requires=[
          'numpy',
          'scikit-learn'
      ],
      include_package_data=True,
      zip_safe=False,
      setup_requires=['pytest-runner'],
      tests_require=[
          'pytest',
      ],
)
