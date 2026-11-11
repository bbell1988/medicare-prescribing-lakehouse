from pyspark.sql import DataFrame, functions as F

HELPER_COLUMNS = ["is_valid", "_cost_lossless", "_fills_lossless", "_benes_ok"]


def build_silver(bronze_df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Return (valid_rows, rejected_rows).

    Expects Bronze-shaped input: the CMS source columns as strings plus an
    integer reporting_year. Every input row lands in exactly one output.
    """
    parsed = bronze_df.select(
        "reporting_year",
        F.col("Prscrbr_NPI").alias("prescriber_npi"),
        F.col("Prscrbr_City").alias("prescriber_city"),
        F.col("Prscrbr_State_Abrvtn").alias("prescriber_state"),
        F.col("Prscrbr_State_FIPS").alias("prescriber_state_fips"),
        F.col("Prscrbr_Type").alias("prescriber_type"),
        F.col("Brnd_Name").alias("brand_name"),
        F.col("Gnrc_Name").alias("generic_name"),
        F.expr("try_cast(Tot_Clms AS BIGINT)").alias("total_claims"),
        F.expr("try_cast(Tot_30day_Fills AS DECIMAL(18,2))").alias("total_30day_fills"),
        F.expr("try_cast(Tot_Day_Suply AS BIGINT)").alias("total_day_supply"),
        F.expr("try_cast(Tot_Drug_Cst AS DECIMAL(18,2))").alias("total_drug_cost"),
        F.expr("try_cast(Tot_Benes AS BIGINT)").alias("total_beneficiaries"),
        F.col("Tot_Benes").isNull().alias("beneficiaries_missing"),
        # Rounding to 2 decimals must not change the value, or precision is lost.
        F.expr(
            "try_cast(Tot_Drug_Cst AS DECIMAL(38,10)) = try_cast(Tot_Drug_Cst AS DECIMAL(18,2))"
        ).alias("_cost_lossless"),
        F.expr(
            "try_cast(Tot_30day_Fills AS DECIMAL(38,10)) = try_cast(Tot_30day_Fills AS DECIMAL(18,2))"
        ).alias("_fills_lossless"),
        (
            F.col("Tot_Benes").isNull()
            | F.expr("try_cast(Tot_Benes AS BIGINT)").isNotNull()
        ).alias("_benes_ok"),
    )

    is_valid = (
        F.col("prescriber_npi").rlike(r"^[0-9]{10}$")
        & (F.col("total_claims") > 0)
        & F.col("total_day_supply").isNotNull()
        & F.col("total_30day_fills").isNotNull()
        & (F.col("total_drug_cost") >= 0)
        & F.col("_cost_lossless")
        & F.col("_fills_lossless")
        & F.col("_benes_ok")
    )

    # A rule evaluating to NULL matches neither is_valid nor NOT is_valid, so
    # coalesce forces every row into exactly one output.
    flagged = parsed.withColumn("is_valid", F.coalesce(is_valid, F.lit(False)))

    valid = flagged.where("is_valid").drop(*HELPER_COLUMNS)
    rejected = flagged.where("NOT is_valid")
    return valid, rejected