import unittest
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from app.dims_and_facts import (
    load_tables,
    reduce_dims,
    ReducedDatabase,
    MultiDimDatabase,
    create_orders_by_meal_type_age_cuisine_table,
)

TABLES_DIR_PATH = Path(__file__).parent.parent / "app" / "tables"

TABLES_SCHEMA = {
    "addresses": {"index": "address_id", "columns": ["district_id", "street"]},
    "birthdates": {"index": "birthdate_id", "columns": ["year", "month", "day"]},
    "cities": {"index": "city_id", "columns": ["name", "state_id"]},
    "countries": {"index": "country_id", "columns": ["name"]},
    "cuisines": {"index": "cuisine_id", "columns": ["name"]},
    "districts": {"index": "district_id", "columns": ["name", "city_id"]},
    "food": {"index": "food_id", "columns": ["name", "cuisine_id", "price"]},
    "orders": {
        "index": "order_id",
        "columns": [
            "user_id",
            "address_id",
            "restaurant_id",
            "food_id",
            "ordered_at",
            "promo_id",
        ],
    },
    "promos": {"index": "promo_id", "columns": ["discount"]},
    "restaurants": {"index": "restaurant_id", "columns": ["name", "address_id"]},
    "states": {"index": "state_id", "columns": ["name", "country_id"]},
    "users": {
        "index": "user_id",
        "columns": ["first_name", "last_name", "birthdate_id", "registred_at"],
    },
}

TABLES = sorted(TABLES_SCHEMA.keys())

REDUCED_TABLES_SCHEMA = {
    "addresses": {
        "index": "address_id",
        "columns": [
            ("city", "object"),
            ("country", "object"),
            ("district", "object"),
            ("state", "object"),
            ("street", "object"),
        ],
    },
    "food": {
        "index": "food_id",
        "columns": [("name", "object"), ("cuisine", "object"), ("price", "float64")],
    },
    "orders": {
        "index": "order_id",
        "columns": [
            ("user_id", "int64"),
            ("address_id", "int64"),
            ("restaurant_id", "int64"),
            ("food_id", "int64"),
            ("ordered_at", "datetime64[ns]"),
            ("promo_id", "object"),
        ],
    },
    "users": {
        "index": "user_id",
        "columns": [
            ("first_name", "object"),
            ("last_name", "object"),
            ("birthdate", "datetime64[ns]"),
            ("registred_at", "datetime64[ns]"),
        ],
    },
    "restaurants": {
        "index": "restaurant_id",
        "columns": [("name", "object"), ("address_id", "int64")],
    },
    "promos": {"index": "promo_id", "columns": [("discount", "float64")]},
}

REDUCED_TABLES = sorted(REDUCED_TABLES_SCHEMA.keys())

TABLE_ORDERS_BY_MEAL_TYPE_AGE_CUISINE = {
    "index": "order_id",
    "columns": [
        ("food_cuisine", "object"),
        ("user_age", "object"),
        ("meal_type", "object"),
    ],
}


def get_sorted_column_names_from_schema(schema: dict) -> List[str]:
    return [name for name, _ in sorted(schema["columns"], key=lambda p: p[0])]


def get_sorted_column_types_from_schema(schema: dict) -> List[np.dtype]:
    return [
        np.dtype(dtype) for _, dtype in sorted(schema["columns"], key=lambda p: p[0])
    ]


def get_sorted_column_names_from_df(df: pd.DataFrame) -> List[str]:
    return sorted(df.columns.tolist())


def get_sorted_column_types_from_df(df: pd.DataFrame) -> List[np.dtype]:
    return [
        dtype
        for dtype, _ in sorted(
            zip(df.dtypes.tolist(), df.columns.tolist()), key=lambda p: p[1]
        )
    ]


def load_single_table(
    name: str, tables_dir_path: Path = TABLES_DIR_PATH
) -> pd.DataFrame:
    res = load_tables(tables_dir_path, [name])
    return res[0]


def load_all_tables(tables_dir_path: Path = TABLES_DIR_PATH) -> List[pd.DataFrame]:
    res = load_tables(tables_dir_path, list(TABLES_SCHEMA.keys()))
    return res


def get_reduced_db() -> ReducedDatabase:
    return reduce_dims(MultiDimDatabase(*load_all_tables()))


def get_table() -> pd.DataFrame:
    return create_orders_by_meal_type_age_cuisine_table(get_reduced_db())


class TestCaseWithImplementationCheck(unittest.TestCase):
    def assertImplemented(self, fn):
        try:
            return fn()
        except NotImplementedError:
            self.fail("This should be implemented")
