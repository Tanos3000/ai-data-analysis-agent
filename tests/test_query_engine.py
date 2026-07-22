import pandas as pd
import pytest

from query_engine import UnsafeQueryError, run_safe_query

SAMPLE = pd.DataFrame({
    "country": ["UK", "Germany", "France"],
    "quantity": [10, 5, 3],
    "unit_price": [2.5, 4.0, 1.0],
})


def test_select_is_allowed():
    result = run_safe_query(SAMPLE, "SELECT country, quantity FROM sales WHERE quantity > 4")
    assert list(result["country"]) == ["UK", "Germany"]


def test_drop_table_rejected():
    with pytest.raises(UnsafeQueryError):
        run_safe_query(SAMPLE, "DROP TABLE sales")


def test_delete_rejected():
    with pytest.raises(UnsafeQueryError):
        run_safe_query(SAMPLE, "DELETE FROM sales WHERE quantity > 4")


def test_insert_rejected():
    with pytest.raises(UnsafeQueryError):
        run_safe_query(SAMPLE, "INSERT INTO sales VALUES ('Spain', 1, 1.0)")


def test_update_rejected():
    with pytest.raises(UnsafeQueryError):
        run_safe_query(SAMPLE, "UPDATE sales SET quantity = 0")


def test_multiple_statements_rejected():
    with pytest.raises(UnsafeQueryError):
        run_safe_query(SAMPLE, "SELECT * FROM sales; DROP TABLE sales")


def test_double_dash_comment_rejected():
    with pytest.raises(UnsafeQueryError):
        run_safe_query(SAMPLE, "SELECT * FROM sales -- DROP TABLE sales")


def test_block_comment_rejected():
    with pytest.raises(UnsafeQueryError):
        run_safe_query(SAMPLE, "SELECT * FROM sales /* comment */ WHERE quantity > 0")
