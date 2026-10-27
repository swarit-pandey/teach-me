"""Shared pytest setup for all workbooks.

Each test file does:   from _impl import load;  ex = load(__file__)
which imports exercises.py (default) or solutions.py (COURSE_IMPL=solutions)
from the test's own directory.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
