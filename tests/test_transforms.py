from decimal import Decimal

import pytest
from pyspark.sql import types as T

from medicare_lakehouse.transforms import build_silver

STRING_COLUMNS = [
    "Prscrbr_NPI",
    "Prscrbr_City",
    "Prscrbr_State_Abrvtn",
    "Prscrbr_State_FIPS",
    "Prscrbr_Type",
    "Brnd_Name",
    "Gnrc_Name",
    "Tot_Clms",
    "Tot_30day_Fills",
    "Tot_Day_Suply",
    "Tot_Drug_Cst",
    "Tot_Benes",
]

BRONZE_SCHEMA = T.StructType(
    [T.StructField("reporting_year", T.IntegerType())]
    + [T.StructField(c, T.StringType()) for c in STRING_COLUMNS]
)

VALID_ROW = {
    "reporting_year": 2024,
    "Prscrbr_NPI": "1003000126",
    "Prscrbr_City": "Testville",
    "Prscrbr_State_Abrvtn": "AL",
    "Prscrbr_State_FIPS": "01",
    "Prscrbr_Type": "Internal Medicine",
    "Brnd_Name": "ExampleBrand",
    "Gnrc_Name": "example_generic",
    "Tot_Clms": "20",
    "Tot_30day_Fills": "22.5",
    "Tot_Day_Suply": "600",
    "Tot_Drug_Cst": "123.45",
    "Tot_Benes": "15",
}


def make_bronze(spark, *overrides):
    rows = [{**VALID_ROW, **o} for o in overrides]
    data = [tuple(r[f.name] for f in BRONZE_SCHEMA.fields) for r in rows]
    return spark.createDataFrame(data, schema=BRONZE_SCHEMA)


def test_normal_row_is_valid_and_typed(spark):
    valid, rejected = build_silver(make_bronze(spark, {}))

    assert rejected.count() == 0
    row = valid.collect()[0]
    assert row["total_claims"] == 20
    assert row["total_drug_cost"] == Decimal("123.45")
    assert row["total_30day_fills"] == Decimal("22.50")
    assert row["prescriber_state_fips"] == "01"
    assert row["beneficiaries_missing"] is False


def test_missing_beneficiaries_stay_null_and_valid(spark):
    valid, rejected = build_silver(make_bronze(spark, {"Tot_Benes": None}))

    assert rejected.count() == 0
    row = valid.collect()[0]
    assert row["total_beneficiaries"] is None
    assert row["beneficiaries_missing"] is True


@pytest.mark.parametrize(
    "override",
    [
        {"Prscrbr_NPI": None},
        {"Prscrbr_NPI": "12345"},
        {"Prscrbr_NPI": "ABCDEFGHIJ"},
        {"Tot_Clms": "abc"},
        {"Tot_Clms": None},
        {"Tot_Clms": "0"},
        {"Tot_Day_Suply": None},
        {"Tot_Drug_Cst": "-1.00"},
        {"Tot_Drug_Cst": "12.345"},
        {"Tot_30day_Fills": "1.234"},
        {"Tot_Benes": "abc"},
    ],
    ids=lambda o: "-".join(f"{k}={v}" for k, v in o.items()),
)
def test_bad_row_is_rejected_not_dropped(spark, override):
    valid, rejected = build_silver(make_bronze(spark, override))

    assert valid.count() == 0
    assert rejected.count() == 1


def test_every_input_row_lands_in_exactly_one_output(spark):
    bronze = make_bronze(
        spark,
        {"Brnd_Name": "good"},
        {"Brnd_Name": "null_npi", "Prscrbr_NPI": None},
        {"Brnd_Name": "bad_claims", "Tot_Clms": "abc"},
        {"Brnd_Name": "missing_benes", "Tot_Benes": None},
    )
    valid, rejected = build_silver(bronze)

    valid_names = {r["brand_name"] for r in valid.collect()}
    rejected_names = {r["brand_name"] for r in rejected.collect()}

    assert valid_names == {"good", "missing_benes"}
    assert rejected_names == {"null_npi", "bad_claims"}
    assert valid.count() + rejected.count() == bronze.count()
