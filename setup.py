from typing import List

from setuptools import find_packages, setup

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    this function will return the list of requirements
    """
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="student-performance-analysis",
    version="0.0.2",
    author="Sunnythesage",
    author_email="Sundaypius2000gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
