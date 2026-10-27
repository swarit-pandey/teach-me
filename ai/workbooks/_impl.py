import importlib.util, os, sys


def load(test_file, name=None):
    """Load exercises.py (or solutions.py when COURSE_IMPL=solutions) next to test_file."""
    d = os.path.dirname(os.path.abspath(test_file))
    mod = name or os.environ.get("COURSE_IMPL", "exercises")
    path = os.path.join(d, mod + ".py")
    if d not in sys.path:
        sys.path.insert(0, d)  # lets a workbook import its own helper modules
    spec = importlib.util.spec_from_file_location(f"wb_{os.path.basename(d)}_{mod}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
