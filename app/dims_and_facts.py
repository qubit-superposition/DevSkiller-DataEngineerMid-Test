from collections import namedtuple
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# List of all tables used in the original database
TABLES = [
    "addresses",
    "birthdates",
    "cities",
    "countries",
    "cuisines",
    "districts",
    "food",
    "orders",
    "promos",
    "restaurants",
    "states",
    "users",
]

# Path to the directory where tables' CSV files are stored
TABLES_DIR_PATH = Path(__file__).parent / "tables"

# Structure holding initial database
MultiDimDatabase = namedtuple(
    "MultiDimDatabase",
    [
        "addresses",
        "birthdates",
        "cities",
        "countries",
        "cuisines",
        "districts",
        "food",
        "orders",
        "promos",
        "restaurants",
        "states",
        "users",
    ],
)
ReducedDatabase = namedtuple(
    "ReducedDatabase", ["orders", "users", "food", "promos", "restaurants", "addresses"]
)


# --- Task #1 ---
def load_tables(tables_dir_path: Path, tables: List[str]) -> List[pd.DataFrame]:
    raise NotImplementedError()


# --- Task # 2 ---
def reduce_dims(db: MultiDimDatabase) -> ReducedDatabase:
    raise NotImplementedError()


# --- Task #3 ---
def create_orders_by_meal_type_age_cuisine_table(db: ReducedDatabase) -> pd.DataFrame:
    raise NotImplementedError()
