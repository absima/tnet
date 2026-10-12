Installation and first analysis
===============================

Install
-------

Use Python 3.9 or newer. From the source directory::

   python -m pip install .

For an editable installation with parallel, documentation and test tools::

   python -m pip install -e ".[parallel,docs,test]"

NumPy, SciPy, scikit-learn and Numba are core dependencies. Joblib and tqdm
are optional parallel tools. NetworkX is an optional test reference, not a
runtime requirement for modularity.

First network
-------------

The example keeps a zero-diagonal network for structure and a
waiting-diagonal copy for propagation and dynamism.

.. literalinclude:: examples/quick_start.py
   :language: python

Download :download:`quick_start.py <examples/quick_start.py>` and run it::

   python quick_start.py

The first compiled call includes Numba compilation. Subsequent calls are
usually faster. The distance vector has eight entries; :doc:`outputs`
explains their order.

Build and browse
----------------

.. only:: html

   :download:`Download the complete PDF for offline reading <../output/pdf/tnet_analysis-1.0.2-user-guide.pdf>`.

From the source directory::

   python -m pip install ".[docs]"
   python -m sphinx -W --keep-going -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` in a browser. The site includes search
and an API reference extracted from library docstrings. For a continuous
HTML page, use builder ``singlehtml`` and a different output directory.

To build the PDF using a TeX installation with XeLaTeX and latexmk::

   python -m sphinx -b latex docs docs/_build/latex
   latexmk -xelatex -interaction=nonstopmode -halt-on-error -cd docs/_build/latex/tnet_user_guide.tex
   mkdir -p output/pdf
   cp docs/_build/latex/tnet_user_guide.pdf output/pdf/tnet_analysis-1.0.2-user-guide.pdf

The PDF has a contents page, page numbers, bookmarks and the API reference.
HTML builds do not require TeX. Rebuild the PDF when guide content changes,
then rebuild HTML so its downloadable copy is refreshed.
