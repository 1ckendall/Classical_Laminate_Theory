# Classical Laminate Theory & Progressive Failure Toolkit

A Python toolkit for the analysis of composite materials. This project implements Classical Laminate Theory (CLT) and Progressive Failure Analysis (PFA) using various failure criteria and material degradation models.

## Key Features

*   **Core CLT Engine:** Automated calculation of A, B, and D stiffness matrices and through-thickness stress/strain distributions.
*   **Advanced Failure Criteria:** Includes validated implementations of **Puck (1996)**, **Hashin (1980)**, **Tsai-Hill**, and **Maximum Stress** criteria.
*   **Non-Destructive PFA:** Uses a Continuum Damage Mechanics (CDM) approach with damage factors, allowing for iterative failure simulation without destroying initial material data.
*   **Visualization Suite:** Generate publication-quality failure envelopes using the `cmcrameri` scientific colormaps.
*   **Module-Based Architecture:** Fully typed and structured as a proper Python module for easy integration into other engineering workflows.

## Installation

This project uses `uv` for high-performance dependency management.

```powershell
# Clone the repository
git clone https://github.com/1char/Classical_Laminate_Theory.git
cd Classical_Laminate_Theory

# Sync dependencies and create virtual environment
uv sync
```

## Quick Start

### 1. Progressive Failure Analysis (PFA)
Run a comparison of failure models on a sample layup:
```powershell
uv run clt-pfa
```

### 2. Failure Envelope Visualization
Visualize the $\sigma_2 - \tau_{12}$ failure boundary for a specific material:
```powershell
uv run clt-viz
```

### 3. Basic API Usage
```python
from classical_laminate_theory import Laminate, Puck

# Define material properties
props = {
    "E1": 140e9, "E2": 10e9, "G12": 5e9, "v12": 0.3, "t": 0.15e-3,
    "Xt": 1000e6, "Xc": 800e6, "Yt": 40e6, "Yc": 120e6, "S12": 40e6
}

# Create a failure model (Puck with standard weakening)
model = Puck(**props, weakening="puck")

# Assemble laminate from layup string
lam = Laminate.from_layup("[0/90/45/-45]_s", failure_model=model, **props)

# Access ABD matrix
print(lam.ABD)
```

## Documentation

Comprehensive documentation is available via Sphinx. To build the HTML docs locally:

```powershell
cd docs
.\make html
```
The output will be located in `docs/build/index.html`.

## Testing

The toolkit includes a validation suite of 25+ tests covering CLT mechanics, recursive layup parsing, and failure model accuracy.

```powershell
uv run pytest
```

## References

*   Puck, A. and Schürmann, H., "Failure analysis of FRP laminates by means of physically based phenomenological models", *Composites Science and Technology*, 1998.
*   Hashin, Z., "Failure Criteria for Unidirectional Fiber Composites", *Journal of Applied Mechanics*, 1980.
