"""Coming-night sleep forecasting experiments (version 1).

This package is separate from the deployed same-night estimator in
``ml/scripts`` and ``backend/app/services``. Nothing here is loaded by the API.

Every dataset keeps its own adapter, participant identifiers and target
definition. Features are computed only from observations recorded before a
per-night cutoff, so the target night never contributes to its own inputs.
"""

FORECASTING_VERSION = "forecasting_v1"
