from setuptools import setup, find_packages

setup(
    name="customer-support-env",
    version="1.0.0",
    description="Customer Support Triage — OpenEnv Environment",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "pydantic>=2.6.0",
        "openai>=1.25.0",
        "pyyaml>=6.0.1",
        "httpx>=0.27.0",
    ],
)
