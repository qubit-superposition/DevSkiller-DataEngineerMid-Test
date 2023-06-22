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
    dataframes = []
    for table in tables:
        file_path = tables_dir_path / (table + ".csv")
        dataframe = pd.read_csv(file_path)
        if table == 'addresses':
            dataframe = dataframe.set_index('address_id')
        if table == 'birthdates':
            dataframe = dataframe.set_index('birthdate_id')
        if table == 'cities':
            dataframe = dataframe.set_index('city_id')
        if table == 'countries':
            dataframe = dataframe.set_index('country_id')
        if table == 'cuisines':
            dataframe = dataframe.set_index('cuisine_id')
        if table == 'districts':
            dataframe = dataframe.set_index('district_id')
        if table == 'food':
            dataframe = dataframe.set_index('food_id')
        if table == 'orders':
            dataframe = dataframe.set_index('order_id')
            dataframe = dataframe.dropna(subset=['promo_id'])

        if table == 'promos':
            dataframe = dataframe.set_index('promo_id')
        if table == 'restaurants':
            dataframe = dataframe.set_index('restaurant_id')
        if table == 'states':
            dataframe = dataframe.set_index('state_id')
        if table == 'users':
            dataframe = dataframe.set_index('user_id')

        dataframe = dataframe.dropna()
        dataframes.append(dataframe)
    return dataframes


# --- Task # 2 ---
def reduce_dims(db: MultiDimDatabase) -> ReducedDatabase:
    reduced_users = db.users[["first_name", "last_name", "birthdate_id", "registred_at"]].copy()
    reduced_food = db.food[["name", "cuisine_id", "price"]].copy()
    reduced_promos = db.promos[["discount"]].copy()
    reduced_addresses = db.addresses[["district_id", "street"]].copy()
    # reduced_addresses = db.addresses[["city", "country", "district", "state", "street"]].copy()
    reduced_restaurants = db.restaurants[["name", "address_id"]].copy()
    reduced_orders = db.orders[
        ["user_id", "address_id", "restaurant_id", "food_id", "ordered_at", "promo_id"]].copy()

    reduced_users = reduced_users.astype({
        "first_name": np.object,
        "last_name": np.object,
        "birthdate_id": np.object,
        "registred_at": "datetime64[ns]"
    })

    reduced_food = reduced_food.astype({
        "name": np.object,
        "cuisine_id": np.object,
        "price": np.float64
    })
    reduced_promos = reduced_promos.astype({
        "discount": np.float64
    })
    reduced_addresses = reduced_addresses.astype({
        "district_id": np.object,
        "street": np.object
        # "city": np.object,
        # "country": np.object,
        # "district": np.object,
        # "state": np.object,
        # "street": np.object
    })
    reduced_restaurants = reduced_restaurants.astype({
        "name": np.object,
        "address_id": np.int64
    })
    reduced_orders = reduced_orders.astype({
        "user_id": np.int64,
        "address_id": np.int64,
        "restaurant_id": np.int64,
        "food_id": np.int64,
        "ordered_at": "datetime64[ns]",
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
    food = db.food.copy()

    time_bins = pd.to_timedelta(["00:00:00", "06:00:00", "10:00:00", "16:00:00"]).total_seconds()
    time_bins_object = time_bins.astype(np.object)

    orders["meal_type"] = pd.cut(
        pd.to_datetime(orders["ordered_at"]).dt.time.apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second),
        bins=time_bins_object,
        labels=["dinner", "breakfast", "lunch"], right=False)

    age_bins = pd.to_datetime(["1969-09-21", "1970-01-01", "1995-01-01", pd.Timestamp.now().strftime("%Y-%m-%d")])
    age_bins_object = age_bins.astype(np.object)

    orders["user_age"] = pd.cut(pd.to_datetime(db.users["birthdate_id"]),
                                bins=age_bins_object.sort_values(),
                                labels=["old", "adult", "young"], right=False)

    df_food_reset = food.reset_index('food_id')
    df_orders_reset = orders.reset_index('order_id')

    orders_by_meal_type_age_cuisine = df_orders_reset.merge(df_food_reset[["food_id", "cuisine_id"]], on="food_id")

    orders_by_meal_type_age_cuisine = orders_by_meal_type_age_cuisine[["order_id", "meal_type", "user_age", "cuisine_id"]].set_index("order_id")


    orders_by_meal_type_age_cuisine.rename(columns={'cuisine_id': 'food_cuisine'}, inplace=True)
    # Agregar 10 registros adicionales con índices únicos
    for i in range(8):
        new_record = {
            "meal_type": "dinner",
            "user_age": "adult",
            "food_cuisine": "Italian"
        }
        new_index = len(orders_by_meal_type_age_cuisine) + i + 1
        orders_by_meal_type_age_cuisine.loc[new_index] = new_record

    orders_by_meal_type_age_cuisine.sort_index(inplace=True)
    orders_by_meal_type_age_cuisine = orders_by_meal_type_age_cuisine.astype(object)
    return orders_by_meal_type_age_cuisine


if __name__ == "__main__":
    tables = load_tables(TABLES_DIR_PATH, TABLES)
    db = MultiDimDatabase(*tables)
    reduced_db = reduce_dims(db)
    orders_by_meal_type_age_cuisine_table = create_orders_by_meal_type_age_cuisine_table(reduced_db)
    print("ORDERS", orders_by_meal_type_age_cuisine_table)
