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
    table_dataframes = []
    for table_name in tables:
        csv_file_path = tables_dir_path / (table_name + ".csv")
        df = pd.read_csv(csv_file_path)
        if "address_id" in df.columns:
            df.set_index("address_id", inplace=True)  # Establecer 'address_id' como índice
        table_dataframes.append(df)
    return table_dataframes

# --- Task # 2 ---
def reduce_dims(db: MultiDimDatabase) -> ReducedDatabase:
    reduced_users = db.users[["user_id", "first name", "last_name", "birthdate", "registred_at"]].copy()
    reduced_food = db.food[["food_id", "name", "cuisine", "price"]].copy()
    reduced_promos = db.promos[["promo_id", "discount"]].copy()
    reduced_addresses = db.addresses[["address_id", "country", "state", "city", "district", "street"]].copy()
    reduced_restaurants = db.restaurants[["restaurant_id", "name", "address_id"]].copy()
    reduced_orders = db.orders[
        ["order_id", "user_id", "address_id", "restaurant_id", "food_id", "ordered_at", "promo_id"]].copy()

    # Assign correct data types
    reduced_users = reduced_users.astype({
        "user_id": np.int64,
        "first name": np.object,
        "last_name": np.object,
        "birthdate": np.datetime64,
        "registred_at": np.datetime64
    })
    reduced_food = reduced_food.astype({
        "food_id": np.int64,
        "name": np.object,
        "cuisine": np.object,
        "price": np.float64
    })
    reduced_promos = reduced_promos.astype({
        "promo_id": np.object,
        "discount": np.float64
    })
    reduced_addresses = reduced_addresses.astype({
        "address_id": np.int64,
        "country": np.object,
        "state": np.object,
        "city": np.object,
        "district": np.object,
        "street": np.object
    })
    reduced_restaurants = reduced_restaurants.astype({
        "restaurant_id": np.int64,
        "name": np.object,
        "address_id": np.int64
    })
    reduced_orders = reduced_orders.astype({
        "order_id": np.int64,
        "user_id": np.int64,
        "address_id": np.int64,
        "restaurant_id": np.int64,
        "food_id": np.int64,
        "ordered_at": np.datetime64,
        "promo_id": np.object
    })

    reduced_db = ReducedDatabase(
        orders=reduced_orders,
        users=reduced_users,
        food=reduced_food,
        promos=reduced_promos,
        restaurants=reduced_restaurants,
        addresses=reduced_addresses
    )

    return reduced_db


# --- Task #3 ---
def create_orders_by_meal_type_age_cuisine_table(db: ReducedDatabase) -> pd.DataFrame:
    meal_type_map = {
        "breakfast": pd.Interval(pd.Timestamp("06:00:00"), pd.Timestamp("10:00:00"), closed="left"),
        "lunch": pd.Interval(pd.Timestamp("10:00:00"), pd.Timestamp("16:00:00"), closed="both")
    }

    user_age_map = {
        "young": pd.Interval(pd.Timestamp("1995-01-01"), pd.Timestamp.now(), closed="both"),
        "adult": pd.Interval(pd.Timestamp("1970-01-01"), pd.Timestamp("1995-01-01"), closed="both"),
        "old": pd.Interval(pd.Timestamp.min, pd.Timestamp("1969-12-31"), closed="both")
    }

    orders = db.orders.copy()
    orders["meal_type"] = pd.cut(orders["ordered_at"].dt.time,
                                 bins=[pd.Timestamp.min.time(), pd.Timestamp("06:00:00").time(),
                                       pd.Timestamp("10:00:00").time(), pd.Timestamp("16:00:00").time()],
                                 labels=["dinner", "breakfast", "lunch"], right=False)
    orders["user_age"] = pd.cut(db.users["birthdate"],
                                bins=[pd.Timestamp.min, pd.Timestamp("1970-01-01"), pd.Timestamp("1995-01-01"),
                                      pd.Timestamp.now()], labels=["old", "adult", "young"], right=False)

    orders_by_meal_type_age_cuisine = orders.merge(db.food[["food_id", "cuisine"]], on="food_id")
    orders_by_meal_type_age_cuisine = orders_by_meal_type_age_cuisine[
        ["order_id", "meal_type", "user_age", "cuisine"]].set_index("order_id")
    orders_by_meal_type_age_cuisine.sort_index(inplace=True)

    return orders_by_meal_type_age_cuisine


