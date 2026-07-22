"""Loads the Online Retail dataset into a clean, query-ready DataFrame."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "online_retail.xlsx"

COLUMN_RENAME = {
    "InvoiceNo": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "UnitPrice": "unit_price",
    "CustomerID": "customer_id",
    "Country": "country",
}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    return df.rename(columns=COLUMN_RENAME)


if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df.dtypes)
