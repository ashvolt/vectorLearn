"""vectorLearn — a course compiler.

A technical book goes in; a dependency-graph course comes out: objectives, a
prerequisite DAG, lessons bound to the source passages they were generated
from, and gates a learner has to pass.

The IR in `ir.py` is the centre of the design. Everything else — the passes
that fill it, the evals that grade it, and eventually the map, reader and
playground that render it — is downstream of that one schema.
"""

from .ir import SCHEMA_VERSION, Course, Node, SourceDoc, Span, Step

__version__ = "0.0.1"

__all__ = ["Course", "Node", "Span", "Step", "SourceDoc", "SCHEMA_VERSION", "__version__"]
