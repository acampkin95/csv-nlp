"""
CSV Validation Engine for Message Processor
Provides comprehensive validation, encoding detection, and data quality checks
for input CSV files containing chat/message data.
"""

import csv
import chardet
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import re
import logging
from dataclasses import dataclass, field
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results"""
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    encoding: str = "utf-8"
    delimiter: str = ","
    column_mapping: Dict[str, str] = field(default_factory=dict)

    def add_error(self, message: str):
        """Add error and mark as invalid"""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        """Add warning (doesn't affect validity)"""
        self.warnings.append(message)

    def add_info(self, message: str):
        """Add informational message"""
        self.info.append(message)

    def get_summary(self) -> str:
        """Get validation summary as string"""
        lines = []
        lines.append(f"Validation {'PASSED' if self.is_valid else 'FAILED'}")
        lines.append(f"Encoding: {self.encoding}")
        lines.append(f"Delimiter: {repr(self.delimiter)}")

        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  ✗ {error}")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")

        if self.info:
            lines.append(f"\nInfo ({len(self.info)}):")
            for info in self.info:
                lines.append(f"  ℹ {info}")

        if self.statistics:
            lines.append("\nStatistics:")
            for key, value in self.statistics.items():
                lines.append(f"  • {key}: {value}")

        return "\n".join(lines)


class CSVValidator:
    """Comprehensive CSV validation for message data"""

    # Required columns (flexible naming)
    REQUIRED_COLUMNS = {
        'date': ['Date', 'date', 'DATE', 'Message Date', 'Timestamp'],
        'time': ['Time', 'time', 'TIME', 'Message Time'],
        'sender': ['Sender Name', 'Sender', 'From', 'Author', 'Speaker'],
        'text': ['Text', 'Message', 'Content', 'Body', 'Message Text'],
    }

    # Optional but useful columns
    OPTIONAL_COLUMNS = {
        'sender_number': ['Sender Number', 'Phone', 'Number', 'From Number'],
        'recipients': ['Recipients', 'To', 'Recipient'],
        'attachment': ['Attachment', 'Attachments', 'Media'],
        'type': ['Type', 'Message Type', 'Category'],
        'service': ['Service', 'Platform', 'Source'],
    }

    # Date format patterns to try
    DATE_FORMATS = [
        '%d/%m/%Y',  # DD/MM/YYYY
        '%m/%d/%Y',  # MM/DD/YYYY
        '%Y-%m-%d',  # YYYY-MM-DD
        '%d-%m-%Y',  # DD-MM-YYYY
        '%Y/%m/%d',  # YYYY/MM/DD
        '%d.%m.%Y',  # DD.MM.YYYY
    ]

    # Time format patterns
    TIME_FORMATS = [
        '%H:%M:%S',     # HH:MM:SS
        '%H:%M',        # HH:MM
        '%I:%M:%S %p',  # HH:MM:SS AM/PM
        '%I:%M %p',     # HH:MM AM/PM
    ]

    def __init__(self, auto_correct: bool = True):
        """Initialize validator

        Args:
            auto_correct: Whether to attempt auto-correction of common issues
        """
        self.auto_correct = auto_correct

    def validate_file(self, file_path: str) -> Tuple[ValidationResult, Optional[pd.DataFrame]]:
        """Validate CSV file comprehensively

        Args:
            file_path: Path to CSV file

        Returns:
            Tuple of (ValidationResult, DataFrame or None if invalid)
        """
        result = ValidationResult()
        file_path = Path(file_path)

        # Check file existence
        if not file_path.exists():
            result.add_error(f"File not found: {file_path}")
            return result, None

        if not file_path.suffix.lower() == '.csv':
            result.add_warning(f"File extension is not .csv: {file_path.suffix}")

        # Detect encoding
        encoding = self._detect_encoding(file_path)
        result.encoding = encoding
        result.add_info(f"Detected encoding: {encoding}")

        # Detect delimiter
        delimiter = self._detect_delimiter(file_path, encoding)
        result.delimiter = delimiter
        result.add_info(f"Detected delimiter: {repr(delimiter)}")

        # Load data
        try:
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                delimiter=delimiter,
                on_bad_lines='warn',
                engine='python'
            )
        except Exception as e:
            result.add_error(f"Failed to read CSV: {str(e)}")
            return result, None

        # Basic statistics
        result.statistics['total_rows'] = len(df)
        result.statistics['total_columns'] = len(df.columns)
        result.add_info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

        # Clean column names
        if self.auto_correct:
            df = self._clean_column_names(df, result)

        # Validate columns
        column_mapping = self._validate_columns(df, result)
        result.column_mapping = column_mapping

        if not result.is_valid:
            return result, None

        # Map columns to standard names
        df_mapped = self._map_columns(df, column_mapping)

        # Validate data types
        self._validate_data_types(df_mapped, result)

        # Check for missing data
        self._check_missing_data(df_mapped, result)

        # Validate dates and times
        self._validate_datetime(df_mapped, result)

        # Check for duplicates
        self._check_duplicates(df_mapped, result)

        # Analyze data quality
        self._analyze_data_quality(df_mapped, result)

        # Auto-correct if enabled
        if self.auto_correct and result.warnings:
            df_mapped = self._auto_correct_issues(df_mapped, result)

        return result, df_mapped if result.is_valid else df

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet

        Args:
            file_path: Path to file

        Returns:
            str: Detected encoding
        """
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')

            # Handle common encoding aliases
            encoding_map = {
                'ascii': 'utf-8',
                'ISO-8859-1': 'latin-1',
                'Windows-1252': 'cp1252',
            }

            return encoding_map.get(encoding, encoding)

    def _detect_delimiter(self, file_path: Path, encoding: str) -> str:
        """Detect CSV delimiter

        Args:
            file_path: Path to file
            encoding: File encoding

        Returns:
            str: Detected delimiter
        """
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            sample = f.read(2048)

            # Try csv.Sniffer
            try:
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                return delimiter
            except:
                pass

            # Fallback: count potential delimiters
            delimiters = [',', '\t', '|', ';']
            delimiter_counts = {d: sample.count(d) for d in delimiters}

            # Return most frequent delimiter
            return max(delimiter_counts, key=delimiter_counts.get)

    def _clean_column_names(self, df: pd.DataFrame, result: ValidationResult) -> pd.DataFrame:
        """Clean column names (remove whitespace, standardize case)

        Args:
            df: DataFrame
            result: Validation result

        Returns:
            pd.DataFrame: DataFrame with cleaned columns
        """
        original_columns = df.columns.tolist()
        cleaned_columns = []

        for col in original_columns:
            # Remove leading/trailing whitespace
            cleaned = col.strip()

            # Remove BOM characters
            cleaned = cleaned.replace('\ufeff', '')

            # Remove non-printable characters
            cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)

            cleaned_columns.append(cleaned)

        # Check for changes
        if original_columns != cleaned_columns:
            df.columns = cleaned_columns
            result.add_info("Column names cleaned (removed whitespace/special characters)")

        return df

    def _validate_columns(self, df: pd.DataFrame, result: ValidationResult) -> Dict[str, str]:
        """Validate required columns exist

        Args:
            df: DataFrame
            result: Validation result

        Returns:
            Dict: Mapping of standard names to actual column names
        """
        column_mapping = {}
        df_columns = df.columns.tolist()

        # Check required columns
        for standard_name, possible_names in self.REQUIRED_COLUMNS.items():
            found = False
            for possible in possible_names:
                if possible in df_columns:
                    column_mapping[standard_name] = possible
                    found = True
                    break

            if not found:
                result.add_error(f"Required column '{standard_name}' not found. "
                               f"Expected one of: {', '.join(possible_names)}")

        # Check optional columns
        for standard_name, possible_names in self.OPTIONAL_COLUMNS.items():
            for possible in possible_names:
                if possible in df_columns:
                    column_mapping[standard_name] = possible
                    break

        # Report unmapped columns
        mapped_columns = set(column_mapping.values())
        unmapped = [col for col in df_columns if col not in mapped_columns]
        if unmapped:
            result.add_info(f"Unmapped columns will be preserved: {', '.join(unmapped)}")

        return column_mapping

    def _map_columns(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Map columns to standard names

        Args:
            df: DataFrame
            column_mapping: Mapping dictionary

        Returns:
            pd.DataFrame: DataFrame with standardized column names
        """
        df_mapped = df.copy()

        # Rename mapped columns
        rename_dict = {v: k for k, v in column_mapping.items()}
        df_mapped = df_mapped.rename(columns=rename_dict)

        return df_mapped

    def _validate_data_types(self, df: pd.DataFrame, result: ValidationResult):
        """Validate data types of columns

        Args:
            df: DataFrame
            result: Validation result
        """
        # Check text column
        if 'text' in df.columns:
            non_string = df['text'].apply(lambda x: not isinstance(x, str) and pd.notna(x))
            if non_string.any():
                result.add_warning(f"{non_string.sum()} non-string values in 'text' column")

        # Check date column
        if 'date' in df.columns:
            # Try to parse dates
            sample = df['date'].dropna().head(100)
            date_format = self._detect_date_format(sample)
            if date_format:
                result.statistics['date_format'] = date_format
            else:
                result.add_warning("Could not detect consistent date format")

        # Check time column
        if 'time' in df.columns:
            sample = df['time'].dropna().head(100)
            time_format = self._detect_time_format(sample)
            if time_format:
                result.statistics['time_format'] = time_format
            else:
                result.add_warning("Could not detect consistent time format")

    def _detect_date_format(self, sample: pd.Series) -> Optional[str]:
        """Detect date format from sample

        Args:
            sample: Sample of date values

        Returns:
            Optional[str]: Detected format or None
        """
        for date_format in self.DATE_FORMATS:
            try:
                # Try to parse all samples with this format
                parsed = sample.apply(lambda x: datetime.strptime(str(x), date_format))
                return date_format
            except:
                continue
        return None

    def _detect_time_format(self, sample: pd.Series) -> Optional[str]:
        """Detect time format from sample

        Args:
            sample: Sample of time values

        Returns:
            Optional[str]: Detected format or None
        """
        for time_format in self.TIME_FORMATS:
            try:
                # Try to parse all samples with this format
                parsed = sample.apply(lambda x: datetime.strptime(str(x), time_format))
                return time_format
            except:
                continue
        return None

    def _validate_datetime(self, df: pd.DataFrame, result: ValidationResult):
        """Validate date and time values

        Args:
            df: DataFrame
            result: Validation result
        """
        if 'date' in df.columns and 'time' in df.columns:
            # Try to combine into datetime
            try:
                df['datetime_combined'] = pd.to_datetime(
                    df['date'].astype(str) + ' ' + df['time'].astype(str),
                    errors='coerce'
                )

                invalid_datetime = df['datetime_combined'].isna() & df['date'].notna()
                if invalid_datetime.any():
                    result.add_warning(f"{invalid_datetime.sum()} rows with invalid date/time")

                # Check for future dates
                future_dates = df['datetime_combined'] > datetime.now()
                if future_dates.any():
                    result.add_warning(f"{future_dates.sum()} messages with future timestamps")

                # Get date range
                if not df['datetime_combined'].isna().all():
                    min_date = df['datetime_combined'].min()
                    max_date = df['datetime_combined'].max()
                    result.statistics['date_range'] = f"{min_date:%Y-%m-%d} to {max_date:%Y-%m-%d}"
                    result.statistics['duration_days'] = (max_date - min_date).days

            except Exception as e:
                result.add_warning(f"Could not validate datetime: {str(e)}")

    def _check_missing_data(self, df: pd.DataFrame, result: ValidationResult):
        """Check for missing data

        Args:
            df: DataFrame
            result: Validation result
        """
        missing_stats = {}

        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(df)) * 100
                missing_stats[col] = f"{missing_count} ({missing_pct:.1f}%)"

                if col in ['text', 'sender', 'date']:
                    if missing_pct > 5:
                        result.add_warning(f"High missing data in '{col}': {missing_stats[col]}")
                    elif missing_pct > 0:
                        result.add_info(f"Missing data in '{col}': {missing_stats[col]}")

        result.statistics['missing_data'] = missing_stats

        # Check for completely empty messages
        if 'text' in df.columns:
            empty_messages = df['text'].isna() | (df['text'].str.strip() == '')
            if empty_messages.any():
                result.add_info(f"{empty_messages.sum()} empty messages")

    def _check_duplicates(self, df: pd.DataFrame, result: ValidationResult):
        """Check for duplicate messages

        Args:
            df: DataFrame
            result: Validation result
        """
        # Check for exact duplicates (all columns)
        exact_duplicates = df.duplicated().sum()
        if exact_duplicates > 0:
            result.add_warning(f"{exact_duplicates} exact duplicate rows")

        # Check for duplicate messages (same text, sender, date, time)
        if all(col in df.columns for col in ['text', 'sender', 'date', 'time']):
            message_duplicates = df.duplicated(subset=['text', 'sender', 'date', 'time']).sum()
            if message_duplicates > 0:
                result.add_info(f"{message_duplicates} duplicate messages (same text/sender/datetime)")

    def _analyze_data_quality(self, df: pd.DataFrame, result: ValidationResult):
        """Analyze overall data quality

        Args:
            df: DataFrame
            result: Validation result
        """
        # Speaker statistics
        if 'sender' in df.columns:
            speaker_counts = df['sender'].value_counts()
            result.statistics['unique_speakers'] = len(speaker_counts)
            result.statistics['messages_per_speaker'] = {
                'mean': speaker_counts.mean(),
                'median': speaker_counts.median(),
                'min': speaker_counts.min(),
                'max': speaker_counts.max()
            }

            # Check for speaker imbalance
            if len(speaker_counts) > 1:
                imbalance_ratio = speaker_counts.max() / speaker_counts.min()
                if imbalance_ratio > 10:
                    result.add_info(f"High speaker imbalance (ratio: {imbalance_ratio:.1f})")

        # Message length statistics
        if 'text' in df.columns:
            df['text_length'] = df['text'].fillna('').str.len()
            result.statistics['message_length'] = {
                'mean': df['text_length'].mean(),
                'median': df['text_length'].median(),
                'min': df['text_length'].min(),
                'max': df['text_length'].max()
            }

            # Check for suspiciously long messages
            very_long = df['text_length'] > 5000
            if very_long.any():
                result.add_warning(f"{very_long.sum()} messages longer than 5000 characters")

        # Check for encoding issues
        if 'text' in df.columns:
            encoding_issues = df['text'].str.contains(r'[�\\x]', na=False, regex=True)
            if encoding_issues.any():
                result.add_warning(f"{encoding_issues.sum()} messages with potential encoding issues")

    def _auto_correct_issues(self, df: pd.DataFrame, result: ValidationResult) -> pd.DataFrame:
        """Auto-correct common issues

        Args:
            df: DataFrame
            result: Validation result

        Returns:
            pd.DataFrame: Corrected DataFrame
        """
        df_corrected = df.copy()

        # Fix encoding issues in text
        if 'text' in df_corrected.columns:
            # Replace common encoding artifacts
            replacements = {
                '\\xe2\\x80\\x99': "'",  # Right single quotation mark
                '\\xe2\\x80\\x9c': '"',  # Left double quotation mark
                '\\xe2\\x80\\x9d': '"',  # Right double quotation mark
                '\\xe2\\x80\\x93': '-',  # En dash
                '\\xe2\\x80\\x94': '--', # Em dash
                '\\xc2\\xa0': ' ',       # Non-breaking space
                '�': '',                 # Replacement character
            }

            for old, new in replacements.items():
                df_corrected['text'] = df_corrected['text'].str.replace(old, new, regex=False)

            result.add_info("Applied encoding fixes to text column")

        # Standardize date format
        if 'date' in df_corrected.columns:
            date_format = result.statistics.get('date_format')
            if date_format:
                try:
                    df_corrected['date_parsed'] = pd.to_datetime(
                        df_corrected['date'],
                        format=date_format,
                        errors='coerce'
                    )
                    df_corrected['date'] = df_corrected['date_parsed'].dt.strftime('%Y-%m-%d')
                    df_corrected = df_corrected.drop(columns=['date_parsed'])
                    result.add_info("Standardized date format to YYYY-MM-DD")
                except:
                    pass

        # Remove exact duplicates
        before_len = len(df_corrected)
        df_corrected = df_corrected.drop_duplicates()
        if len(df_corrected) < before_len:
            result.add_info(f"Removed {before_len - len(df_corrected)} duplicate rows")

        return df_corrected


def validate_csv_file(file_path: str, auto_correct: bool = True) -> Tuple[bool, Optional[pd.DataFrame]]:
    """Convenience function to validate CSV file

    Args:
        file_path: Path to CSV file
        auto_correct: Whether to auto-correct issues

    Returns:
        Tuple of (is_valid, DataFrame or None)
    """
    validator = CSVValidator(auto_correct=auto_correct)
    result, df = validator.validate_file(file_path)

    # Print summary
    print(result.get_summary())

    return result.is_valid, df