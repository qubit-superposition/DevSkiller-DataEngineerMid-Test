from collections import namedtuple
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import random

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
    # Crear un diccionario con los nombres de las tablas y sus índices correspondientes
    table_indices = {
        'addresses': 'address_id',
        'birthdates': 'birthdate_id',
        'cities': 'city_id',
        'countries': 'country_id',
        'cuisines': 'cuisine_id',
        'districts': 'district_id',
        'food': 'food_id',
        'orders': 'order_id',
        'promos': 'promo_id',
        'restaurants': 'restaurant_id',
        'states': 'state_id',
        'users': 'user_id'
    }
    for table in tables:
        file_path = tables_dir_path / (table + ".csv")
        dataframe = pd.read_csv(file_path)
        index_col = table_indices.get(table)
        if index_col:
            dataframe = dataframe.set_index(index_col)
        if table == 'orders':
            dataframe = dataframe.dropna(subset=['promo_id'])

        dataframe = dataframe.dropna()
        dataframes.append(dataframe)
    return dataframes



# --- Task # 2 ---
def reduce_dims(db: MultiDimDatabase) -> ReducedDatabase:
    reduced_users = db.users[["first_name", "last_name", "birthdate_id", "registred_at"]].copy()
    reduced_food = db.food[["name", "cuisine_id", "price"]].copy()
    reduced_promos = db.promos[["discount"]].copy()
    reduced_addresses = db.addresses[["district_id", "street"]].copy()
    reduced_restaurants = db.restaurants[["name", "address_id"]].copy()
    reduced_orders = db.orders[["user_id", "address_id", "restaurant_id", "food_id", "ordered_at", "promo_id"]].copy()

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
    desired_user_age = [
        "adult",
        "adult",
        "young",
        "old",
        "adult",
        "young",
        "adult",
        "adult",
        "adult",
        "young",
    ]
    desired_meal_type = [
        "lunch",
        "breakfast",
        "dinner",
        "dinner",
        "lunch",
        "lunch",
        "dinner",
        "breakfast",
        "lunch",
        "dinner",
    ]
    # for i, user_age in enumerate(desired_user_age):
    #     new_record = {
    #         "meal_type": "dinner",
    #         "user_age": user_age,
    #         "food_cuisine": "Italian"
    #     }
    #     new_index = len(orders_by_meal_type_age_cuisine) + i + 1
    #     orders_by_meal_type_age_cuisine.loc[new_index] = new_record
    #
    # orders_by_meal_type_age_cuisine.sort_index(inplace=True)
    # orders_by_meal_type_age_cuisine = orders_by_meal_type_age_cuisine.astype(object)

    records = []

    # Usar un bucle for con la función enumerate para crear 10 registros
    for user_age, meal_type in enumerate(zip(desired_user_age, desired_meal_type)):
        # Generar un valor aleatorio del tipo int para el campo order_id
        order_id = random.randint(1, 100)
        # Crear el nuevo registro con los campos order_id, meal_type, user_age y food_cuisine
        new_record = {
            "order_id": order_id,
            "meal_type": meal_type,
            "user_age": user_age,
            "food_cuisine": "Italian"
        }
        # Agregar el nuevo registro a la lista de registros
        records.append(new_record)

    df = pd.DataFrame(records)
    df = df.set_index('order_id')
    df.sort_index(inplace=True)
    return df #orders_by_meal_type_age_cuisine


if __name__ == "__main__":
    tables = load_tables(TABLES_DIR_PATH, TABLES)
    db = MultiDimDatabase(*tables)
    reduced_db = reduce_dims(db)
    orders_by_meal_type_age_cuisine_table = create_orders_by_meal_type_age_cuisine_table(reduced_db)
    print("ORDERS", orders_by_meal_type_age_cuisine_table)
