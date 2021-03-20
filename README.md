# Arabidopsis Rosette Analysis

Author: Suxing Liu

![CI](https://github.com/Computational-Plant-Science/arabidopsis-rosette-analysis/workflows/CI/badge.svg)

Robust and parameter-free plant image segmentation and trait extraction.

1. Process with plant image top view, including whole tray plant image, this tool will segment it into individual images.
2. Robust segmentation based on parameter-free color clustering method.
3. Extract individual plant gemetrical traits, and write output into excel file.

## Requirements

Either [Docker](https://www.docker.com/) or [Singularity ](https://sylabs.io/singularity/) is required to run this project in a Unix environment.

## Usage

### Docker

```bash
docker run -v "$(pwd)":/opt/arabidopsis-rosette-analysis -w /opt/arabidopsis-rosette-analysis computationalplantscience/arabidopsis-rosette-analysis python3 /opt/arabidopsis-rosette-analysis/trait_extract_parallel.py -i input -o output -ft "jpg,png"
```

### Singularity

```bash
singularity exec docker://computationalplantscience/arabidopsis-rosette-analysis python3 trait_extract_parallel.py -i input -o output -ft "jpg,png"
```