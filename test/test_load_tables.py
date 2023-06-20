import unittest

import pandas as pd

from app.dims_and_facts import load_tables
from test.common import (
    TABLES_DIR_PATH,
    TABLES_SCHEMA,
    load_all_tables,
    load_single_table,
    TestCaseWithImplementationCheck,
)


class TestLoadTables(TestCaseWithImplementationCheck):
    def test_no_tables_when_no_filepaths_given(self):
        res = self.assertImplemented(lambda: load_tables(TABLES_DIR_PATH, []))
        self.assertEqual(len(res), 0)

    def test_tables_count_equal_given_filepaths_count(self):
        res = self.assertImplemented(
            lambda: load_tables(TABLES_DIR_PATH, list(TABLES_SCHEMA.keys())[:5])
        )
        self.assertEqual(len(res), 5)

    def test_tables_are_dataframes(self):
        res = self.assertImplemented(lambda: load_all_tables())
        for df in res:
            self.assertIsInstance(df, pd.DataFrame)

    def test_tables_have_correct_indices(self):
        for table, info in TABLES_SCHEMA.items():
            df = self.assertImplemented(lambda: load_single_table(table))
            self.assertEqual(df.index.name, info["index"])

    def test_tables_have_correct_headers(self):
        for table, info in TABLES_SCHEMA.items():
            df = self.assertImplemented(lambda: load_single_table(table))
            self.assertEqual(df.columns.tolist(), info["columns"])

    def test_tables_have_records(self):
        for table, info in TABLES_SCHEMA.items():
            df = self.assertImplemented(lambda: load_single_table(table))
            self.assertGreater(len(df), 0)


if __name__ == "__main__":
    unittest.main()
