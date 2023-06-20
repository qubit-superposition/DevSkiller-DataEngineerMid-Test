from setuptools import find_packages, setup

requirements = [
    "pytest<7,>=5",
    "pytest-timeout",
]

setup(
    name="dimensional-modelling",
    version="1.0.0",
    author="DevSkiller",
    author_email="support@devskiller.com",
    packages=find_packages(),
    package_data={"app": ["tables/*.csv"], "verify_pack": ["tables/*.csv"]},
    install_requires=["numpy", "pandas"],
    tests_require=requirements,
    setup_requires=["pytest-runner"],
    extras_require={"dev": ["black"]},
)
