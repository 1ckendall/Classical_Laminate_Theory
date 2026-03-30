# GEMINI.md - Project Context

## Project Overview
**Classical_Laminate_Theory** is a Python-based engineering toolkit for the analysis of composite materials using Classical Laminate Theory (CLT). It provides high-fidelity calculations for lamina and laminate stiffness matrices (ABD matrices), stress/strain distributions through the laminate thickness, and Progressive Failure Analysis (PFA).

### Main Technologies
- **Python (>=3.14):** Modern Python features (type hints, properties).
- **NumPy & SciPy:** core linear algebra and matrix operations.
- **Matplotlib & Seaborn:** Visualization of stress distributions and failure curves.
- **ezdxf:** Generation of CAD-compatible profiles for isotensoid vessels.
- **uv:** Modern package management.

### Architecture
- **Structures:** `Lamina` and `Laminate` classes manage geometric and material properties, with dynamic stiffness recalculation during failure simulation.
- **Failure Analysis:** A pluggable architecture for failure criteria (Max Stress, Tsai-Hill, Hashin, Puck) that identifies specific failure modes (fiber tension, matrix compression, etc.).
- **Progressive Failure:** Iterative load-stepping that degrades material properties based on identified failure modes until global structural failure occurs.
- **Isotensoid Design:** Specialized module for generating optimal winding profiles for pressure vessels.

## Building and Running

### Environment Setup
The project uses `uv` for dependency management.
```powershell
# Sync dependencies
uv sync
```

### Running Tests
Tests are located in the `test/` directory and can be run using `pytest`.
```powershell
# Run all tests
uv run pytest
```

### Key Commands
- **Interactive Development:** Use `pressure_vessel_development.ipynb` for exploring the API and developing new vessel designs.
- **Isotensoid Generation:** Run `isotensoid/isotensoid_generator.py` to generate vessel profiles and export to DXF.
- **PFA Comparison:** Run `classical_laminate_theory/progressive_failure_analysis.py` directly to see a comparison of different failure models on a sample layup.

## Development Conventions

### Coding Style
- **Type Safety:** Use comprehensive type hints for all class methods and functions.
- **Dynamic Stiffness:** The `Lamina` class uses `@property` for stiffness matrices (`Qmat`, `Qbarmat`). This ensures that if engineering constants (E1, E2, etc.) are degraded during PFA, the stiffness matrices are automatically updated.
- **Failure Modes:** Failure criteria must return a list of `FailureMode` enums to allow the PFA loop to apply targeted stiffness knockdowns.
- **Docstrings:** Follow standard Python docstring conventions for classes and complex methods.

### Testing Practices
- **Unit Tests:** New helpers or core logic should be accompanied by tests in the `test/` directory.
- **Empirical Validation:** The project includes PDF references (`Puck1996...`) which should be used to validate the implementation of failure models against established literature.

## Key Files
- `classical_laminate_theory/structures.py`: Core mechanics of plies and laminates.
- `classical_laminate_theory/failuremodels.py`: Implementations of composite failure criteria.
- `classical_laminate_theory/progressive_failure_analysis.py`: The simulation engine for structural degradation.
- `classical_laminate_theory/helpers.py`: Layup string parser and numerical utilities.
- `isotensoid/isotensoid_generator.py`: Generates optimal vessel geometries.
- `pyproject.toml`: Project metadata and dependencies.
