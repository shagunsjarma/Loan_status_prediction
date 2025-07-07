from setuptools import setup, find_packages

setup(
    name = "loan_status_prediction",
    version = "0.0.1",
    author = "Shagun",
    author_email = "Shagunsharma029@gmail.com",
    packages = find_packages(),
    install_requires = ["pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "joblib"],
    description = "A package for loan status prediction",
    long_description = "A package for loan status prediction",
    long_description_content_type = "text/markdown",
)