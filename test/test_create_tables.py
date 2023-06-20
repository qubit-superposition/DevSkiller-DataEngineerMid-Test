import unittest

from test.common import (
    get_sorted_column_names_from_schema,
    get_sorted_column_types_from_df,
    get_sorted_column_types_from_schema,
    get_sorted_column_names_from_df,
    get_table,
    TABLE_ORDERS_BY_MEAL_TYPE_AGE_CUISINE,
    TestCaseWithImplementationCheck,
)


class TestCreateOrdersByMealTypeAgeCuisineTable(TestCaseWithImplementationCheck):
    def setUp(self) -> None:
        self.table = self.assertImplemented(lambda: get_table())

        self.desired_row_count = 10
        self.desired_user_age = [
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
        self.desired_meal_type = [
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

    def test_table_should_be_of_proper_length(self):
        self.assertEqual(len(self.table), self.desired_row_count)

    def test_table_should_have_correct_index(self):
        self.assertEqual(
            self.table.index.name, TABLE_ORDERS_BY_MEAL_TYPE_AGE_CUISINE["index"]
        )

    def test_table_should_have_correct_column_names(self):
        self.assertEqual(
            get_sorted_column_names_from_df(self.table),
            get_sorted_column_names_from_schema(TABLE_ORDERS_BY_MEAL_TYPE_AGE_CUISINE),
        )

    def test_table_should_have_correct_column_types(self):
        self.assertEqual(
            get_sorted_column_types_from_df(self.table),
            get_sorted_column_types_from_schema(TABLE_ORDERS_BY_MEAL_TYPE_AGE_CUISINE),
        )

    def test_table_should_be_sorted_by_order_id(self):
        self.assertTrue(self.table.index.is_monotonic_increasing)

    def test_user_age_is_properly_mapped(self):
        self.assertListEqual(self.table["user_age"].tolist(), self.desired_user_age)

    def test_meal_type_is_properly_mapped(self):
        self.assertListEqual(self.table["meal_type"].tolist(), self.desired_meal_type)


if __name__ == "__main__":
    unittest.main()
