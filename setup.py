import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='kerastroke',
    version='2.0.3',
    scripts=['./kerastroke/__init__.py'],
    author="Charles Averill",
    author_email="charlesaverill20@gmail.com",
    description="A suite of the generalization-improvement techniques Stroke, Pruning, and NeuroPlast",
    long_description=long_description,
    install_requires=['keras', 'numpy'],
    long_description_content_type="text/markdown",
    url="https://github.com/CharlesAverill/kerastroke/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ]
)
