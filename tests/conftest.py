"""Global test fixtures to mock all jolpica API calls.

No test should hit the live Jolpica API. All jolpica functions are mocked at the
module level so no test accidentally hits the network.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

# Mock data - defined at module level so they're available at import time
_mock_schedule = pd.DataFrame({
    "season": [2026, 2026, 2026],
    "round": [16, 17, 18],
    "race_name": ["Bahrain Grand Prix", "Singapore Grand Prix", "Japanese Grand Prix"],
    "circuit_id": ["sepang", "marina_bay", "suzuka"],
    "race_datetime": pd.to_datetime([
        "2026-10-04T07:00:00Z",
        "2026-10-11T12:00:00Z",
        "2026-10-25T06:00:00Z"
    ]),
})

_mock_results = pd.DataFrame({
    "season": [2024, 2024, 2025, 2025],
    "round": [17, 17, 17, 17],
    "driver_id": ["max_verstappen", "oscar_piastri", "max_verstappen", "oscar_piastri"],
    "constructor_id": ["red_bull", "mclaren", "red_bull", "mclaren"],
    "circuit_id": ["marina_bay", "marina_bay", "marina_bay", "marina_bay"],
    "position": [1, 2, 2, 1],
    "points": [25, 18, 25, 18],
    "dnf": [0, 0, 0, 0],
    "grid": [1, 2, 2, 1],
    "goals": [1, 0, 1, 0],
    "assists": [0, 1, 0, 1],
    "dnf": [0, 0, 0, 0],
})

_mock_schedule_hist = pd.DataFrame({
    "season": [2024, 2025, 2026],
    "round": [17, 17, 17],
    "circuit_id": ["marina_bay", "marina_bay", "marina_bay"],
})

_mock_results = pd.DataFrame({
    "season": [2024, 2024, 2025, 2025],
    "round": [17, 17, 17, 17],
    "driver_id": ["max_verstappen", "oscar_piastri", "max_verstappen", "oscar_piastri"],
    "constructor_id": ["red_bull", "mclaren", "red_bull", "mclaren"],
    "circuit_id": ["marina_bay", "marina_bay", "marina_bay", "marina_bay"],
    "position": [1, 2, 2, 1],
    "points": [25, 18, 25, 18],
    "dnf": [0, 0, 0, 0],
    "grid": [1, 2, 2, 1],
    "goals": [1, 0, 1, 0],
    "assists": [0, 1, 0, 1],
    "dnf": [0, 0, 0, 0],
})

_mock_schedule_hist = pd.DataFrame({
    "season": [2024, 2025, 2026],
    "round": [17, 17, 17],
    "circuit_id": ["marina_bay", "marina_bay", "marina_bay"],
})

def _mock_load_history(season, force_refresh=False):
    mock_results = pd.DataFrame({
        "season": [2024, 2024, 2025, 2025],
        "round": [17, 17, 17, 17],
        "driver_id": ["max_verstappen", "oscar_piastri", "max_verstappen", "oscar_piastri"],
        "constructor_id": ["red_bull", "mclaren", "red_bull", "mclaren"],
        "circuit_id": ["marina_bay", "marina_bay", "marina_bay", "marina_bay"],
        "position": [1, 2, 2, 1],
        "points": [25, 18, 25, 18],
        "dnf": [0, 0, 0, 0],
        "grid": [1, 2, 2, 1],
        "goals": [1, 0, 1, 0],
        "assists": [0, 1, 0, 1],
        "dnf": [0, 0, 0, 0],
    })
    mock_schedule_hist = pd.DataFrame({
        "season": [2024, 2025, 2026],
        "round": [17, 17, 17],
        "circuit_id": ["marina_bay", "marina_bay", "marina_bay"],
    })
    return (mock_results, mock_schedule_hist)

def _mock_fetch_season_schedule(season, force_refresh=False):
    mock_schedule = pd.DataFrame({
        "season": [2026, 2026, 2026],
        "round": [16, 17, 18],
        "race_name": ["Bahrain Grand Prix", "Singapore Grand Prix", "Japanese Grand Prix"],
        "circuit_id": ["sepang", "marina_bay", "suzuka"],
        "race_datetime": pd.to_datetime([
            "2026-10-04T07:00:00Z",
            "2026-10-11T12:00:00Z",
            "2026-10-25T06:00:00Z"
        ]),
    })
    return mock_schedule

def _mock_load_season_results(season, force_refresh=False):
    return pd.DataFrame()

def _mock_fetch_race_results(season, round_, force_refresh=False):
    return pd.DataFrame()

def _mock_fetch_driver_standings(season):
    return pd.DataFrame({
        "driver_id": ["max_verstappen", "oscar_piastri"],
        "points": [200, 150],
        "position": [1, 2]
    })

def _mock_fetch_constructor_standings(season):
    return pd.DataFrame({
        "constructor_id": ["red_bull", "mclaren"],
        "points": [400, 300],
        "position": [1, 2]
    })

def _mock_load_season_results(season, force_refresh=False):
    return pd.DataFrame()

def _mock_load_multi_season_results(seasons):
    return pd.DataFrame()

def _mock_fetch_qualifying(season, round_):
    return pd.DataFrame()

def _mock_load_season_qualifying(season, force_refresh=False):
    return pd.DataFrame()

def _mock_load_season_sprints(season, force_refresh=False):
    return pd.DataFrame()

def _mock_fetch_race_results(season, round_, force_refresh=False):
    return pd.DataFrame()

def _mock_load_multi_season_results(seasons):
    return pd.DataFrame()

def _mock_fetch_qualifying(season, round_):
    return pd.DataFrame()

def _mock_fetch_season_schedule(season):
    return pd.DataFrame()

def _mock_load_season_qualifying(season, force_refresh=False):
    return pd.DataFrame()

def _mock_load_season_sprints(season, force_refresh=False):
    return pd.DataFrame()

def _mock_fetch_race_results(season, round_, force_refresh=False):
    return pd.DataFrame()

def _mock_load_multi_season_results(seasons):
    return pd.DataFrame()

def _mock_fetch_qualifying(season, round_):
    return pd.DataFrame()

def _mock_fetch_season_schedule(season):
    return pd.DataFrame()

def _mock_load_season_qualifying(season, force_refresh=False):
    return pd.DataFrame()

def _mock_load_season_sprints(season, force_refresh=False):
    return pd.DataFrame()

def _mock_fetch_race_results(season, round_, force_refresh=False):
    return pd.DataFrame()

def _mock_load_multi_season_results(seasons):
    return pd.DataFrame()

def _mock_fetch_qualifying(season, round_):
    return pd.DataFrame()

def _mock_fetch_season_schedule(season):
    return pd.DataFrame({
        "season": [2026], "round": [16],
        "race_name": ["Test GP"],
        "race_datetime": pd.to_datetime(["2026-10-04T07:00:00Z"])
    })

def _mock_load_season_qualifying(season, force_refresh=False):
    return pd.DataFrame()

def _mock_load_season_sprints(season, force_refresh=False):
    return pd.DataFrame()

def _mock_fetch_race_results(season, round_, force_refresh=False):
    return pd.DataFrame()

def _mock_load_multi_season_results(seasons):
    return pd.DataFrame()

def _mock_fetch_qualifying(season, round_):
    return pd.DataFrame()

def _mock_fetch_season_schedule(season):
    return pd.DataFrame({
        "season": [2026], "round": [16],
        "race_name": ["Test GP"],
        "race_datetime": pd.to_datetime(["2026-10-04T07:00:00Z"])
    })

def _mock_load_season_qualifying(season, force_refresh=False):
    return pd.DataFrame()

def _mock_load_season_sprints(season, force_refresh=False):
    return pd.DataFrame()

# Apply patches BEFORE any test modules are imported
import f1_predictor.data.jolpica as jolpica_module
import f1_predictor.api.routes as routes_module

# Patch the actual module where functions are defined
import f1_predictor.data.jolpica as jolpica_module
jolpica_module.load_history = lambda season, force_refresh=False: (
    pd.DataFrame({
        "season": [2024, 2024, 2025, 2025],
        "round": [17, 17, 17, 17],
        "driver_id": ["max_verstappen", "oscar_piastri", "max_verstappen", "oscar_piastri"],
        "constructor_id": ["red_bull", "mclaren", "red_bull", "mclaren"],
        "circuit_id": ["marina_bay", "marina_bay", "marina_bay", "marina_bay"],
        "position": [1, 2, 2, 1],
        "points": [25, 18, 25, 18],
        "dnf": [0, 0, 0, 0],
        "grid": [1, 2, 2, 1],
        "goals": [1, 0, 1, 0],
        "assists": [0, 1, 0, 1],
        "dnf": [0, 0, 0, 0],
    }), pd.DataFrame({
        "season": [2024, 2025, 2026],
        "round": [17, 17, 17],
        "circuit_id": ["marina_bay", "marina_bay", "marina_bay"],
    })
)

import f1_predictor.data.jolpica as jolpica_module
jolpica_module.fetch_season_schedule = lambda season, force_refresh=False: pd.DataFrame({
    "season": [2026, 2026, 2026],
    "round": [16, 17, 18],
    "race_name": ["Bahrain Grand Prix", "Singapore Grand Prix", "Japanese Grand Prix"],
    "circuit_id": ["sepang", "marina_bay", "suzuka"],
    "race_datetime": pd.to_datetime([
        "2026-10-04T07:00:00Z",
        "2026-10-11T12:00:00Z",
        "2026-10-25T06:00:00Z"
    ]),
})

import f1_predictor.data.jolpica as jolpica_module
jolpica_module.load_season_results = lambda season, force_refresh=False: pd.DataFrame()
jolpica_module.fetch_race_results = lambda season, round_, force_refresh=False: pd.DataFrame()
jolpica_module.fetch_driver_standings = lambda season: pd.DataFrame({
    "driver_id": ["max_verstappen", "oscar_piastri"],
    "points": [200, 150],
    "position": [1, 2]
})
jolpica_module.fetch_constructor_standings = lambda season: pd.DataFrame({
    "constructor_id": ["red_bull", "mclaren"],
    "points": [400, 300],
    "position": [1, 2]
})
jolpica_module.load_season_results = lambda season, force_refresh=False: pd.DataFrame()
jolpica_module.load_multi_season_results = lambda seasons: pd.DataFrame()
jolpica_module.fetch_qualifying = lambda season, round_: pd.DataFrame()
jolpica_module.load_season_qualifying = lambda season, force_refresh=False: pd.DataFrame()
jolpica_module.load_season_sprints = lambda season, force_refresh=False: pd.DataFrame()
jolpica_module.fetch_race_results = lambda season, round_, force_refresh=False: pd.DataFrame()
jolpica_module.load_multi_season_results = lambda seasons: pd.DataFrame()
jolpica_module.fetch_qualifying = lambda season, round_: pd.DataFrame()
jolpica_module.load_season_qualifying = lambda season, force_refresh=False: pd.DataFrame()
jolpica_module.load_season_sprints = lambda season, force_refresh=False: pd.DataFrame()

# Also patch the routes module's jolpica reference for any direct access
import f1_predictor.api.routes as routes_module
routes_module.jolpica.load_history = lambda season, force_refresh=False: (
    pd.DataFrame({
        "season": [2024, 2024, 2025, 2025],
        "round": [17, 17, 17, 17],
        "driver_id": ["max_verstappen", "oscar_piastri", "max_verstappen", "oscar_piastri"],
        "constructor_id": ["red_bull", "mclaren", "red_bull", "mclaren"],
        "circuit_id": ["marina_bay", "marina_bay", "marina_bay", "marina_bay"],
        "position": [1, 2, 2, 1],
        "points": [25, 18, 25, 18],
        "dnf": [0, 0, 0, 0],
        "grid": [1, 2, 2, 1],
        "goals": [1, 0, 1, 0],
        "assists": [0, 1, 0, 1],
        "dnf": [0, 0, 0, 0],
    }), pd.DataFrame({
        "season": [2024, 2025, 2026],
        "round": [17, 17, 17],
        "circuit_id": ["marina_bay", "marina_bay", "marina_bay"],
    })
)

# Also patch the data.jolpica module for any direct imports
import f1_predictor.data.jolpica as jolpica_module
jolpica_module.fetch_season_schedule = lambda season, force_refresh=False: pd.DataFrame({
    "season": [2026, 2026, 2026],
    "round": [16, 17, 18],
    "race_name": ["Bahrain Grand Prix", "Singapore Grand Prix", "Japanese Grand Prix"],
    "circuit_id": ["sepang", "marina_bay", "suzuka"],
    "race_datetime": pd.to_datetime([
        "2026-10-04T07:00:00Z",
        "2026-10-11T12:00:00Z",
        "2026-10-25T06:00:00Z"
    ]),
})

# Also patch the data.jolpica module for any direct imports
import f1_predictor.data.jolpica as jolpica_module
jolpica_module.load_season_results = lambda season, force_refresh=False: pd.DataFrame()
jolpica_module.fetch_race_results = lambda season, round_, force_refresh=False: pd.DataFrame()
jolpica_module.fetch_driver_standings = lambda season: pd.DataFrame({
    "driver_id": ["max_verstappen", "oscar_piastri"],
    "points": [200, 150],
    "position": [1, 2]
})
jolpica_module.fetch_constructor_standings = lambda season: pd.DataFrame({
    "constructor_id": ["red_bull", "mclaren"],
    "points": [400, 300],
    "position": [1, 2]
})
jolpica_module.load_season_results = lambda season, force_refresh=False: pd.DataFrame()
jolpica_module.load_multi_season_results = lambda seasons: pd.DataFrame()
jolpica_module.fetch_qualifying = lambda season, round_: pd.DataFrame()
jolpica_module.load_season_qualifying = lambda season, force_refresh=False: pd.DataFrame()
jolpica_module.load_season_sprints = lambda season, force_refresh=False: pd.DataFrame()
jolpica_module.fetch_race_results = lambda season, round_, force_refresh=False: pd.DataFrame()
jolpica_module.load_multi_season_results = lambda seasons: pd.DataFrame()
jolpica_module.fetch_qualifying = lambda season, round_: pd.DataFrame()
jolpica_module.load_season_qualifying = lambda season, force_refresh=False: pd.DataFrame()
jolpica_module.load_season_sprints = lambda season, force_refresh=False: pd.DataFrame()

# Also patch the routes module's jolpica reference for any direct access
import f1_predictor.api.routes as routes_module
routes_module.jolpica.load_history = lambda season, force_refresh=False: (
    pd.DataFrame({
        "season": [2024, 2024, 2025, 2025],
        "round": [17, 17, 17, 17],
        "driver_id": ["max_verstappen", "oscar_piastri", "max_verstappen", "oscar_piastri"],
        "constructor_id": ["red_bull", "mclaren", "red_bull", "mclaren"],
        "circuit_id": ["marina_bay", "marina_bay", "marina_bay", "marina_bay"],
        "position": [1, 2, 2, 1],
        "points": [25, 18, 25, 18],
        "dnf": [0, 0, 0, 0],
        "grid": [1, 2, 2, 1],
        "goals": [1, 0, 1, 0],
        "assists": [0, 1, 0, 1],
        "dnf": [0, 0, 0, 0],
    }), pd.DataFrame({
        "season": [2024, 2025, 2026],
        "round": [17, 17, 17],
        "circuit_id": ["marina_bay", "marina_bay", "marina_bay"],
    })
)

# Also patch the data.jolpica module for any direct imports
import f1_predictor.data.jolpica as jolpica_module
jolpica_module.fetch_season_schedule = lambda season, force_refresh=False: pd.DataFrame({
    "season": [2026, 2026, 2026],
    "round": [16, 17, 18],
    "race_name": ["Bahrain Grand Prix", "Singapore Grand Prix", "Japanese Grand Prix"],
    "circuit_id": ["sepang", "marina_bay", "suzuka"],
    "race_datetime": pd.to_datetime([
        "2026-10-04T07:00:00Z",
        "2026-10-11T12:00:00Z",
        "2026-10-25T06:00:00Z"
    ]),
})

# Also patch the data.jolpica module for any direct imports
import f1_predictor.data.jolpica as jolpica_module
jolpica_module.load_season_results = lambda season, force_refresh=False: pd.DataFrame()
jolpica_module.fetch_race_results = lambda season, round_, force_refresh=False: pd.DataFrame()
jolpica_module.fetch_driver_standings = lambda season: pd.DataFrame({
    "driver_id": ["max_verstappen", "oscar_piastri"],
    "points": [200, 150],
    "position": [1, 2]
})
jolpica_module.fetch_constructor_standings = lambda season: pd.DataFrame({
    "constructor_id": ["red_bull", "mclaren"],
    "points": [400, 300],
    "position": [1, 2]
})
jolpica_module.load_season_results = lambda season, force_refresh=False: pd.DataFrame()
jolpica_module.load_multi_season_results = lambda seasons: pd.DataFrame()
jolpica_module.fetch_qualifying = lambda season, round_: pd.DataFrame()
jolpica_module.load_season_qualifying = lambda season, force_refresh=False: pd.DataFrame()
jolpica_module.load_season_sprints = lambda season, force_refresh=False: pd.DataFrame()
jolpica_module.fetch_race_results = lambda season, round_, force_refresh=False: pd.DataFrame()
jolpica_module.load_multi_season_results = lambda seasons: pd.DataFrame()
jolpica_module.fetch_qualifying = lambda season, round_: pd.DataFrame()
jolpica_module.load_season_qualifying = lambda season, force_refresh=False: pd.DataFrame()
jolpica_module.load_season_sprints = lambda season, force_refresh=False: pd.DataFrame()