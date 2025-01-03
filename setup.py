from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path, 'r', encoding='utf-8') as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='student_performance_prediction',
    version='0.0.1',
    author='Suman Moni',
    author_email='sumanmoni27@gmail.com',
    description='A package to predict student performance',
    license='MIT',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)