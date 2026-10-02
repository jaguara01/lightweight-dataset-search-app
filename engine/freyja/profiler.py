import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
logger = logging.getLogger(__name__)

import duckdb
import pandas as pd
from tqdm import tqdm
import ast
import time


@dataclass
class ProfilerConfig:
    """Configuration for the data profiler."""
    datalake_path: Path # Path were the CSVs are stored
    output_profiles_path: Path # Path were the profile file will be stored
    max_workers: int = 8 # Maximum amount of parallel workers
    varchar_only: bool = True # True -> only non-numerical columns are processed. False -> both numerical and string columns are processed


class ColumnProfiler:
    """Handles profiling of individual columns."""
    
    def __init__(self, con: duckdb.DuckDBPyConnection, table_name: str):
        self.con = con
        self.table_name = table_name
    
    def _execute_query(self, query: str) -> tuple:
        """Execute a query and return the first row."""
        return self.con.execute(query).fetchone()
    
    def _execute_query_as_dict(self, query: str) -> dict:
        cursor = self.con.execute(query)
        row = cursor.fetchone()
        if row is None:
            return {}
        # Use cursor.description to get column names
        return {desc[0]: value for desc, value in zip(cursor.description, row)}
    
    def _execute_query_all(self, query: str) -> List[tuple]:
        """Execute a query and return all rows."""
        return self.con.execute(query).fetchall()
    
    def _get_row_count(self) -> int:
        """Get total row count for the table."""
        return self._execute_query(f'SELECT COUNT(*) FROM "{self.table_name}"')[0]
    
    def _get_non_null_count(self, column: str) -> int:
        """Get count of non-null values in a column."""
        query = f'SELECT COUNT(*) FROM "{self.table_name}" WHERE "{column}" IS NOT NULL'
        return self._execute_query(query)[0]
    
    def _get_cardinality_metrics(self, column: str) -> Dict[str, Any]:
        from .metrics import CardinalityMetrics
        metrics = CardinalityMetrics()
        query = metrics.build_query(column, self.table_name)
        result = self._execute_query_as_dict(query)
        row_count = self._get_row_count()
        return metrics.process_result(result, row_count)

    
    def _get_distribution_metrics(self, column: str, row_count: int) -> Dict[str, Any]:
        """Calculate frequency distribution metrics."""
        from .metrics import DistributionMetrics
        metrics = DistributionMetrics()
        query = metrics.build_query(column, self.table_name)
        result = self._execute_query_as_dict(query)
        row_count = self._get_row_count()
        return metrics.process_result(result, row_count)
    
    def _get_common_values(self, column: str) -> Tuple[List[str], List[str]]:
        """Get most common values and their Soundex representations."""
        from .metrics import CommonValuesMetrics
        metrics = CommonValuesMetrics()
        query = metrics.build_query(column, self.table_name)
        result = self._execute_query_all(query)
        return metrics.process_result(result)
    
    def _get_length_metrics(self, column: str) -> Dict[str, Any]:
        """Calculate string length metrics."""
        from .metrics import LengthMetrics
        metrics = LengthMetrics()
        query = metrics.build_query(column, self.table_name)
        result = self._execute_query_as_dict(query)
        return metrics.process_result(result)
    
    def _get_word_count_metrics(self, column: str) -> Dict[str, Any]:
        """Calculate word count statistics."""
        from .metrics import WordCountMetrics
        metrics = WordCountMetrics()
        query = metrics.build_query(column, self.table_name)
        result = self._execute_query_as_dict(query)
        return metrics.process_result(result)
    
    def _get_boundary_values(self, column: str) -> Dict[str, Any]:
        """Get first and last values (lexicographically)."""
        from .metrics import BoundaryMetrics
        metrics = BoundaryMetrics()
        query = metrics.build_query(column, self.table_name)
        result = self._execute_query_as_dict(query)
        return metrics.process_result(result)
    
    def _get_column_type_flags(self, column: str) -> Dict[str, int]:
        """Determine if column is empty or binary."""
        from .metrics import ColumnFlagsMetrics
        metrics = ColumnFlagsMetrics()
        query = metrics.build_query(column, self.table_name)
        result = self._execute_query(query)
        return metrics.process_result(result)
    
    def profile_column(self, column: str) -> Optional[Dict[str, Any]]:
        """Generate a complete profile for a single column."""
        # Check for non-null values
        non_null_count = self._get_non_null_count(column)
        if non_null_count == 0:
            return None
        
        row_count = self._get_row_count()
        
        # Initialize profile
        profile = {
            'dataset_name': self.table_name,
            'attribute_name': column
        }
        
        # Gather all metrics
        profile.update(self._get_cardinality_metrics(column))
        profile.update(self._get_distribution_metrics(column, row_count))
        profile.update(self._get_common_values(column))
        profile.update(self._get_length_metrics(column))
        profile.update(self._get_word_count_metrics(column))
        profile.update(self._get_boundary_values(column))
        profile.update(self._get_column_type_flags(column))
        
        return self._round_floats(profile)
    
    @staticmethod
    def _round_floats(obj: Any, decimals: int = 9) -> Any:
        """Recursively round float values in nested structures."""
        if isinstance(obj, float):
            return round(obj, decimals)
        elif isinstance(obj, dict):
            return {k: ColumnProfiler._round_floats(v, decimals) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ColumnProfiler._round_floats(i, decimals) for i in obj]
        return obj


class TableProfiler:
    """Handles profiling of entire tables."""
    
    def __init__(self, config: ProfilerConfig):
        self.config = config
    
    def _get_columns(self, con: duckdb.DuckDBPyConnection, table_name: str) -> List[str]:
        """Get column names to profile."""
        if self.config.varchar_only:
            query = f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}' 
                AND data_type IN ('VARCHAR', 'BOOLEAN')
            """
        else:
            query = f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
            """
        
        return [col[0] for col in con.execute(query).fetchall()]
    
    def profile_table(self, con: duckdb.DuckDBPyConnection, table_name: str) -> pd.DataFrame:
        """Generate profiles for all columns in a table."""
        profiler = ColumnProfiler(con, table_name)
        columns = self._get_columns(con, table_name)

        # Fetch DuckDB-inferred data types from information_schema.
        # These are the raw SQL types (BIGINT, DOUBLE, VARCHAR, TIMESTAMP, …)
        # and are used downstream to compute column-type composition fractions
        # without any secondary read of the CSV.
        try:
            type_rows = con.execute(
                f"SELECT column_name, data_type "
                f"FROM information_schema.columns "
                f"WHERE table_name = '{table_name}'"
            ).fetchall()
            dtype_map = {col: dtype for col, dtype in type_rows}
        except Exception:
            dtype_map = {}

        profiles = []
        for column in columns:
            profile = profiler.profile_column(column)
            if profile is not None:
                profile['data_type'] = dtype_map.get(column, 'UNKNOWN')
                profiles.append(profile)

        return pd.DataFrame(profiles)


class DataProfilerWorker:
    """Worker for processing individual CSV files."""
    
    def __init__(self, config: ProfilerConfig):
        self.config = config
        self.table_profiler = TableProfiler(config)

    @staticmethod
    def clean_column_names(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
        """Clean column names by removing special characters."""
        cols = con.execute(f""" SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}';""").fetchall()
        
        for (col_name,) in cols:
            if not isinstance(col_name, str):
                continue
            
            clean_name = (
                col_name.encode("utf-8").decode("utf-8-sig")
                .replace("\ufeff", "").replace('"', "").replace("\n", "").replace("\r", "").strip()
            )
            
            if clean_name != col_name:
                escaped_col_name = col_name.replace('"', '""')
                escaped_clean_name = clean_name.replace('"', '""')
                con.execute(f'ALTER TABLE "{table_name}" RENAME COLUMN "{escaped_col_name}" TO "{escaped_clean_name}"')
    
    @staticmethod
    def clean_string_values(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
        """Clean string values by lowercasing, trimming, and removing line breaks."""
        cols = con.execute(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table_name}';
        """).fetchall()
        
        string_columns = [col for col, dtype in cols if dtype.upper() in ("VARCHAR", "STRING", "TEXT")]
        
        if string_columns:
            set_exprs = [
                f'"{col}" = lower(replace(replace(trim("{col}"), chr(10), \'\'), chr(13), \'\'))'
                for col in string_columns
            ]
            update_sql = f'UPDATE "{table_name}" SET {", ".join(set_exprs)}'
            con.execute(update_sql)
    
    def process_csv(self, csv_path: Path) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
        """Process a single CSV file and return its profile."""
        table_name = csv_path.name
        
        try:
            con = duckdb.connect(database=":memory:")
            try:
                # Load CSV into a DuckDB table.
                try:
                    con.execute(f""" CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM read_csv_auto('{csv_path.as_posix()}',
                            header = TRUE, ignore_errors = TRUE, sample_size = 100, strict_mode = false, parallel = TRUE);""")
                except Exception:
                    # Fallback: use pandas for problematic files. That is, read the data as a dataframe skipping the bad rows and then store in DuckDB
                    df = pd.read_csv(csv_path, dtype=str, on_bad_lines='skip')
                    con.register('temp_df', df)
                    con.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM temp_df')
                
                # Clean data
                self.clean_column_names(con, table_name)
                self.clean_string_values(con, table_name)
                
                # Generate profile
                profile_df = self.table_profiler.profile_table(con, table_name)
                
            finally:
                con.close()
            
            return (table_name, profile_df, None)
            
        except Exception as exc:
            tb = traceback.format_exc()
            return (table_name, None, f"{type(exc).__name__}: {exc}\n{tb}")


class DataProfiler:
    """Main profiler orchestrator."""
    
    def __init__(self, config: ProfilerConfig):
        self.config = config
        self.worker = DataProfilerWorker(config)
        # Create necessary directories.
        self.config.output_profiles_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize_profiles(profiles):
        """Apply Z-score normalization to an entire dataset of profiles, in order to prevent scale problems"""
        from .metrics import MetricProperties
        metrics_to_norm = MetricProperties().metrics_to_normalize
        profiles[metrics_to_norm] = (profiles[metrics_to_norm] - profiles[metrics_to_norm].mean()) / profiles[metrics_to_norm].std(ddof=1)
        profiles[metrics_to_norm] = profiles[metrics_to_norm].fillna(0.0)
        return profiles
    
    @staticmethod
    def preprocess_profiles(profiles):
        """Precompute numeric, set, and string versions of columns to avoid per-row conversions."""
        from .metrics import MetricProperties
        distance_patterns = MetricProperties().distance_patterns

        for metric, pattern in distance_patterns.items():
            if pattern == 'substraction':  # numeric
                profiles[metric] = pd.to_numeric(profiles[metric], errors="coerce")

            elif pattern == 'containment':  # set containment
                def to_set(value):
                    if isinstance(value, (list, set)):
                        return set(value)
                    if isinstance(value, str):
                        try:
                            parsed = ast.literal_eval(value)
                            return set(parsed) if isinstance(parsed, (list, set)) else {str(parsed)}
                        except (ValueError, SyntaxError):
                            return {value}
                    return {str(value)} if pd.notna(value) else set()

                profiles[metric] = profiles[metric].apply(to_set)

            elif pattern == 'levenshtein':  # string
                profiles[metric] = profiles[metric].apply(lambda v: str(v) if pd.notna(v) else None)

        return profiles
        
    
    def generate_profiles_for_datalake(self) -> None:
        """Run the profiler on all CSV files."""
        csv_files = list(self.config.datalake_path.glob("*.csv"))
        
        if not csv_files:
            error = f"No CSV files found at {self.config.datalake_path}"
            logger.error(error)
            return error
        
        futures = []
        results = []
        errors = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            for csv_path in csv_files:
                table_name = csv_path.name
                futures.append(executor.submit(self.worker.process_csv, csv_path))
            
            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tables"):
                try:
                    table_name, profile_df, error = future.result()
                    if error is not None:
                        text = f"[ERROR] Table {table_name} could not be profiled: {error}"
                        logger.error(text)
                        errors.append(text)
                    else:
                        results.append(profile_df)
                except Exception as e:
                    text = f"Unexpected error retrieving future result: {e}"
                    logger.error(text)
                    errors.append(text)

        total_time = time.time() - start_time

        if not results:
            error = "No profiles were produced successfully."
            logger.error(error)
            return error
        
        # Concatenate all profiles into a single dataframe
        try:
            all_profiles = pd.concat(results, ignore_index=True)
            all_profiles.to_csv(self.config.output_profiles_path, index=False)
        except Exception as e:
            logger.error(e)
            raise

        # Normalize the profiles
        try:
            all_profiles_normalized = self.normalize_profiles(all_profiles)
            output_profiles_path_normalized = self.config.output_profiles_path.with_stem(self.config.output_profiles_path.stem + "_normalized")
            all_profiles_normalized = all_profiles_normalized.round(6)
            all_profiles_normalized.to_csv(output_profiles_path_normalized, index=False)
        except Exception as e:
            logger.error(e)
            raise

        # Preprocess the profiles
        try:
            all_profiles_preprocessed = self.preprocess_profiles(all_profiles)
            output_profiles_path_preprocessed = output_profiles_path_normalized.with_suffix(".pkl")
            all_profiles_preprocessed.to_pickle(output_profiles_path_preprocessed)
        except Exception as e:
            logger.error(e)
            raise

        logger.info(f"Profiling completed (time -> {total_time:.4f})")
        if not errors:
            return f"Profiling completed (time -> {total_time:.4f}). Execution finished without errors"
        else:
            return f"Profiling completed (time -> {total_time:.4f}). Some tables could not be processed: {errors}"