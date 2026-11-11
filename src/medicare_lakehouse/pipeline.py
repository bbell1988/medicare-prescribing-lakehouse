from pyspark.sql import SparkSession, functions as F

from medicare_lakehouse.transforms import build_silver

CATALOG = "dbw_medicare_lakehouse_dev"
BRONZE_TABLE = f"{CATALOG}.bronze.prescriber_drug"
SILVER_TABLE = f"{CATALOG}.silver.prescriber_drug"
REJECTS_TABLE = f"{CATALOG}.silver.prescriber_drug_rejects"


def _replace_year(df, table: str, reporting_year: int) -> None:
    """Atomically replace one reporting year, so reruns never duplicate rows."""
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("replaceWhere", f"reporting_year = {int(reporting_year)}")
        .saveAsTable(table)
    )


def ingest_bronze(spark: SparkSession, source_path: str, reporting_year: int) -> None:
    """Load the unchanged CMS CSV as strings, plus lineage columns."""
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.bronze")
    bronze = (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .csv(source_path)
        .withColumn("reporting_year", F.lit(int(reporting_year)))
        .withColumn("source_file", F.col("_metadata.file_path"))
        .withColumn("ingested_at", F.current_timestamp())
    )
    _replace_year(bronze, BRONZE_TABLE, reporting_year)


def build_silver_tables(
    spark: SparkSession, reporting_year: int, max_reject_ratio: float = 0.001
) -> dict:
    """Build Silver and rejects from Bronze, failing before publishing bad loads."""
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.silver")
    year_filter = F.col("reporting_year") == int(reporting_year)

    bronze = spark.table(BRONZE_TABLE).where(year_filter)
    valid, rejected = build_silver(bronze)

    # Rejects go first so a failed quality gate leaves Silver untouched.
    _replace_year(rejected, REJECTS_TABLE, reporting_year)
    n_bronze = bronze.count()
    n_rejected = spark.table(REJECTS_TABLE).where(year_filter).count()

    if n_bronze == 0:
        raise ValueError(f"No Bronze rows for reporting year {reporting_year}")
    if n_rejected / n_bronze > max_reject_ratio:
        raise ValueError(
            f"Rejected {n_rejected} of {n_bronze} rows, above the "
            f"{max_reject_ratio:.2%} threshold; Silver was not updated"
        )

    _replace_year(valid, SILVER_TABLE, reporting_year)
    n_valid = spark.table(SILVER_TABLE).where(year_filter).count()

    if n_valid + n_rejected != n_bronze:
        raise ValueError(
            f"Row counts do not reconcile: bronze={n_bronze}, "
            f"silver={n_valid}, rejected={n_rejected}"
        )
    return {"bronze": n_bronze, "silver": n_valid, "rejected": n_rejected}
