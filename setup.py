from setuptools import setup, find_packages

setup(
    name="zest",
    version="0.1.0",
    description="ZEST: Injecting Inductive Bias at Tokenizer level",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "pyarrow>=13.0.0",
        "gdown>=5.0.0",
        "matplotlib>=3.8.0",
    ],
    extras_require={
        "code": [
            "tree-sitter>=0.25.0",
            "tree-sitter-c",
            "tree-sitter-cpp",
            "tree-sitter-go",
            "tree-sitter-java",
            "tree-sitter-javascript",
            "tree-sitter-php",
            "tree-sitter-python",
            "tree-sitter-ruby",
            "gensim>=4.0.0",
            "tokenizers>=0.15.0",
        ],
    },
)
