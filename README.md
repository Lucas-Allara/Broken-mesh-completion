# Broken mesh completion

Implementation of *Robust Inside-Outside Segmentation using Generalized Winding Numbers* (Jacobson, Kavan and Sorkine-Hornung, SIGGRAPH 2013), with cap fairing and symmetry completion.

```
Broken-mesh-completion/
├── assets/    the test models (Spot, Homer, Lucy, Ogre)
├── demos/     the scripts that draw the figures
├── figures/   the figures of the demos and the numbers of the damage experiment
├── gsd/       winding numbers, tetrahedralization, graph cut, fairing and symmetry
├── poster/    the poster, with the script that renders its figures
└── tests/     the pytest suite
```

```bash
pip install -r requirements.txt
python -m pytest
python demos/demo_segmentation_spot.py   # each demo writes its figure to figures/
```
