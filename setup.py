from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="hotel-booking-pred",
    version="0.0.1",
    author="",
    author_email="",
    packages=find_packages(),
    install_requires=requirements,
)
