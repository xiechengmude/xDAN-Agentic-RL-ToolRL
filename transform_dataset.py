import os
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq


def process_parquet_file(file_path: str, default_type: str = "") -> pa.Table:
    """
    Process a single parquet file to add the 'type' field to extra_info column.

    Args:
        file_path: Path to the input parquet file
        default_type: Default value for the 'type' field

    Returns:
        Processed PyArrow Table with updated extra_info column
    """
    # Step 1. Read the existing Parquet file using PyArrow.
    old_table = pq.read_table(file_path)

    # Step 2. Retrieve the "extra_info" column (a StructArray).
    # Convert it to a list of dictionaries for easy modification.
    extra_info_list = old_table.column("extra_info").to_pylist()

    # Step 3. Modify each row in extra_info_list to include the new "type" field.
    new_extra_info_list = []
    for entry in extra_info_list:
        # entry is expected to be a dict like {"index": some_int, "split": some_str}
        # Only add "type" field if it doesn't already exist
        if "type" not in entry:
            entry["type"] = default_type  # Set the default or desired value here.
        else:
            return old_table
        new_extra_info_list.append(entry)

    # print(new_extra_info_list)
    # Step 4. Create a new extra_info column as a PyArrow array with the desired schema.
    new_extra_info_type = pa.struct([pa.field("index", pa.int64()), pa.field("split", pa.string()), pa.field("type", pa.string())])
    new_extra_info_array = pa.array(new_extra_info_list, type=new_extra_info_type)

    # Step 5. Produce a new PyArrow Table with the updated extra_info column.
    # Replace the old "extra_info" column.
    extra_info_index = old_table.schema.get_field_index("extra_info")
    new_table = old_table.set_column(extra_info_index, "extra_info", new_extra_info_array)

    return new_table


def merge_parquet_files(input_files: List[str], output_file: str, file_type: str = None):
    """
    Merge multiple parquet files together, adding type information to each.

    Args:
        input_files: List of input parquet file paths
        output_file: Path to the output merged parquet file
        file_types: Optional list of type values for each input file (defaults to filename without extension)
    """
    if file_type is None:
        # Use filename without extension as default type
        file_type = [os.path.splitext(os.path.basename(f))[0] for f in input_files][0]

    processed_tables = []

    # Process each input file
    for file_path in input_files:
        print(f"Processing {file_path} with type '{file_type}'...")
        processed_table = process_parquet_file(file_path, file_type)
        processed_tables.append(processed_table)
        print(f"  - Processed {len(processed_table)} rows")

    # Merge all tables using batch writing to avoid concat issues with nested data
    print("Merging tables...")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if len(processed_tables) == 1:
        # Single table case - write directly
        print(f"Writing table with {len(processed_tables[0])} total rows to {output_file}")
        pq.write_table(processed_tables[0], output_file)
    else:
        # Multiple tables - use ParquetWriter to write incrementally
        print("Writing tables incrementally to avoid memory issues...")

        # Get schema from first table
        schema = processed_tables[0].schema

        # Create ParquetWriter
        with pq.ParquetWriter(output_file, schema) as writer:
            total_rows = 0
            for i, table in enumerate(processed_tables):
                print(f"  - Writing table {i + 1}/{len(processed_tables)} ({len(table)} rows)")
                writer.write_table(table)
                total_rows += len(table)

        print(f"Successfully wrote {total_rows} total rows to {output_file}")


# Example usage:
if __name__ == "__main__":
    # Define input files to merge
    input_files = [
        "dataset/rlla_4k/test.parquet",
        "dataset/Eurus-2-RL-Data/train.parquet",  # Add more files as needed
    ]

    # Optional: Define specific type labels for each file
    file_types = "default"

    # Merge files
    merge_parquet_files(input_files, "dataset/rlla_4k/train.parquet", file_types)
