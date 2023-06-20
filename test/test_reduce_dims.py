import unittest

from test.common import (
    REDUCED_TABLES_SCHEMA,
    get_reduced_db,
    get_sorted_column_names_from_df,
    get_sorted_column_names_from_schema,
    get_sorted_column_types_from_df,
    get_sorted_column_types_from_schema,
    TestCaseWithImplementationCheck,
)


class TestReduceDims(TestCaseWithImplementationCheck):
    def test_no_none_tables(self):
        reduced_db = self.assertImplemented(lambda: get_reduced_db())
        self.assertIsNotNone(reduced_db.orders)
        self.assertIsNotNone(reduced_db.users)
        self.assertIsNotNone(reduced_db.food)
        self.assertIsNotNone(reduced_db.promos)
        self.assertIsNotNone(reduced_db.restaurants)
        self.assertIsNotNone(reduced_db.addresses)

    def test_reduced_tables_have_correct_indices(self):
        reduced_db = self.assertImplemented(lambda: get_reduced_db())
        for table, info in REDUCED_TABLES_SCHEMA.items():
            df = reduced_db.__getattribute__(table)
            self.assertEqual(df.index.name, info["index"])

    def test_reduced_tables_have_correct_column_names(self):
        reduced_db = self.assertImplemented(lambda: get_reduced_db())
        for table, info in REDUCED_TABLES_SCHEMA.items():
            df = reduced_db.__getattribute__(table)
            self.assertEqual(
                get_sorted_column_names_from_df(df),
                get_sorted_column_names_from_schema(info),
            )

    def test_reduced_tables_have_correct_column_types(self):
        reduced_db = self.assertImplemented(lambda: get_reduced_db())
        for table, info in REDUCED_TABLES_SCHEMA.items():
            df = reduced_db.__getattribute__(table)
            self.assertEqual(
                get_sorted_column_types_from_df(df),
                get_sorted_column_types_from_schema(info),
            )


if __name__ == "__main__":
    unittest.main()
