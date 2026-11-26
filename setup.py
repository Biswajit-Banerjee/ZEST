from setuptools import setup, find_packages

setup(
    name="zest",
    version="0.2.0",
    description="LEMON-ZEST: Evolution-Informed Tokenization for Efficient Protein Language Modeling",
    packages=find_packages(include=["zest", "zest.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "pyarrow>=13.0.0",
        "tokenizers>=0.15.0",
        "matplotlib>=3.8.0",
    ],
    extras_require={
        "hf": [
            "transformers>=4.44.0",
            "safetensors>=0.4.0",
            "huggingface-hub>=0.24.0",
        ],
        "dev": [
            "pytest>=7.0.0",
        ],
    },
)
