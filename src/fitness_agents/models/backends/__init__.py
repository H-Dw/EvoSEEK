"""Optional heavyweight predictor backends.

Modules in this package are loaded lazily through ``backend_factory`` so the core package remains
CPU-friendly and does not import optional deep-learning dependencies unless selected by config.
"""

