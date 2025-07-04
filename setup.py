from setuptools import setup, find_packages

# Read requirements from requirements.txt
def read_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="vqa-ir-qa",
    version="0.1.0",
    description="Visual Question Answering Information Retrieval and Question Answering",
    author="Omar",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=read_requirements(),
    include_package_data=True,
    zip_safe=False,
)