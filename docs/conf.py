"""Sphinx configuration for the temporal-network user guide."""
import ast
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
setup_tree = ast.parse((PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8"))
setup_call = next(node for node in ast.walk(setup_tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == "setup")
release = next(ast.literal_eval(item.value) for item in setup_call.keywords
               if item.arg == "version")
version = release
project = "Temporal Network Analysis Toolkit"
author = "Simachew Mengiste (with Demian Battaglia)"
copyright = "2026, tnet_analysis contributors"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon",
              "sphinx.ext.doctest", "sphinx.ext.mathjax"]
html_theme = "alabaster"
html_title = f"tnet_analysis {release} User Guide"
html_sidebars = {"**": ["about.html", "navigation.html", "searchfield.html"]}
exclude_patterns = ["_build", ".DS_Store", "examples"]
autodoc_member_order = "bysource"
autodoc_typehints = "none"
add_module_names = False
maximum_signature_line_length = 60
python_maximum_signature_line_length = 60
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False
latex_documents = [("index", "tnet_user_guide.tex",
                    f"Temporal Network Analysis Toolkit {release}: User Guide",
                    author, "manual")]
latex_engine = "xelatex"
latex_show_pagerefs = True
latex_domain_indices = False
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    "extraclassoptions": "oneside,openany",
    "sphinxsetup": "hmargin={25mm,25mm},vmargin={25mm,25mm}",
    "printindex": "",
    # Compact the contents only; keep the body typography unchanged.
    "tableofcontents": (
        r"\begingroup\linespread{0.9}\selectfont"
        r"\sphinxtableofcontents\endgroup"
    ),
}


def NormalizeNotes(app, what, name, obj, options, lines):
    """Indent Notes bullet continuations before Napoleon converts docstrings."""
    in_notes = False
    bullet_indent = None
    for index, line in enumerate(lines):
        if line and not line[0].isspace() and line.endswith(":"):
            in_notes = line == "Notes:"
            bullet_indent = None
        elif in_notes and line.strip():
            indent = len(line) - len(line.lstrip())
            if line.lstrip().startswith("- "):
                bullet_indent = indent
            elif bullet_indent is not None and indent == bullet_indent:
                lines[index] = "  " + line


def setup(app):
    app.connect("autodoc-process-docstring", NormalizeNotes, priority=400)
