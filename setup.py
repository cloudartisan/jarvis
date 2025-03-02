from setuptools import setup, find_packages

setup(
    name="jarvis",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'jarvis.face': ['cascades/*.xml'],
    },
    install_requires=[
        'opencv-contrib-python>=4.0.0',
        'google-auth>=2.3.0',
        'google-cloud-core>=2.0.0',
        'google-cloud-speech>=2.0.0',
        'pyaudio>=0.2.11',
        'scipy>=1.7.0',
        'pillow>=8.0.0',
        'matplotlib>=3.4.0',
        'numpy>=1.20.0',
        'PyQt5>=5.15.0',
    ],
    entry_points={
        'console_scripts': [
            'jarvis=jarvis:main',
        ],
    },
    author="Cloud Artisan",
    author_email="david@cloudartisan.com",
    description="Computer vision and face detection application",
    keywords="computer vision, face detection, video processing",
    url="https://github.com/cloudartisan/jarvis",
)
