"""The published page: charts, template, and the build that turns metrics into HTML.

Deliberately empty of re-exports. ``from .build import build`` here would bind the name
``build`` in this package to the *function*, shadowing the module of the same name for every
importer — including the tests.
"""
