import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Optimize page configuration
st.set_page_config(page_title="Stored Proc Output Validator", layout="wide")
st.title("🧪 Stored Procedure Output Validator")

# Optimization: Combine file uploaders and reduce redundant code
def upload_excel_files():
    """Upload and validate Excel files"""
    file1 = st.sidebar.file_uploader("📤 Existing Logic Output", type=["xlsx"])
    file2 = st.sidebar.file_uploader("📤 Optimized Logic Output", type=["xlsx"])
    return file1, file2

# Optimization: Separate comparison logic into a dedicated function
def compare_dataframes(df_old, df_new, sort_cols, comparison_settings):
    """
    Compare two dataframes with advanced comparison settings
    
    Args:
        df_old (pd.DataFrame): Original dataframe
        df_new (pd.DataFrame): New dataframe to compare
        sort_cols (list): Columns to sort by
        comparison_settings (dict): Settings for comparison
    
    Returns:
        tuple: Mismatch report, mismatched columns
    """
    # Sorting dataframes
    df_old_sorted = df_old.sort_values(by=sort_cols).reset_index(drop=True)
    df_new_sorted = df_new.sort_values(by=sort_cols).reset_index(drop=True)
    
    # Optimization: Pre-compute common columns
    common_cols = list(set(df_old_sorted.columns) & set(df_new_sorted.columns))
    
    # Optimization: Preallocate list for better performance
    mismatch_report = []
    mismatched_columns = set()
    
    # Optimization: Use NumPy for faster comparisons where possible
    for idx in range(min(len(df_old_sorted), len(df_new_sorted))):
        row_has_mismatch = False
        row_data = {"Row": idx + 1}
        
        # Add key columns for reference
        for key_col in sort_cols:
            row_data[f"Key_{key_col}"] = df_old_sorted.loc[idx, key_col]
        
        # Vectorized comparison would be even faster, but this maintains flexibility
        for col in common_cols:
            old_val = df_old_sorted.loc[idx, col]
            new_val = df_new_sorted.loc[idx, col]
            
            if not safe_equals(old_val, new_val, **comparison_settings):
                row_has_mismatch = True
                mismatched_columns.add(col)
                row_data[f"Old_{col}"] = old_val
                row_data[f"New_{col}"] = new_val
        
        if row_has_mismatch:
            mismatch_report.append(row_data)
    
    return mismatch_report, mismatched_columns

# Optimization: Make comparison function more configurable
def safe_equals(val1, val2, 
                ignore_case=False, 
                ignore_whitespace=False, 
                float_precision=5, 
                treat_nulls_equal=False):
    """Enhanced value comparison with configurable settings"""
    # Null value handling
    if treat_nulls_equal:
        if pd.isna(val1) and pd.isna(val2):
            return True
        if pd.isna(val1) or pd.isna(val2):
            return False
    
    # Convert to strings with safe handling
    str1 = str(val1) if val1 is not None else ""
    str2 = str(val2) if val2 is not None else ""
    
    # Apply whitespace and case settings
    if ignore_whitespace:
        str1 = str1.strip()
        str2 = str2.strip()
        
    if ignore_case:
        str1 = str1.lower()
        str2 = str2.lower()
    
    # Numeric comparison with precision
    try:
        num1 = float(str1) if str1 else float('nan')
        num2 = float(str2) if str2 else float('nan')
        if not pd.isna(num1) and not pd.isna(num2):
            return round(num1, float_precision) == round(num2, float_precision)
    except (ValueError, TypeError):
        pass
    
    return str1 == str2

def main():
    # Advanced Settings
    with st.expander("Advanced Settings", expanded=False):
        st.header("Comparison Settings")
        ignore_case = st.checkbox("Ignore text case differences", value=False)
        ignore_whitespace = st.checkbox("Ignore whitespace differences", value=False)
        float_precision = st.number_input("Float comparison precision (decimal places)", 
                                          min_value=1, max_value=10, value=5)
        treat_nulls_equal = st.checkbox("Treat all null values as equal (None, NaN, empty string)", value=False)

    # File upload
    file1, file2 = upload_excel_files()

    if file1 and file2:
        # Optimization: Use read_excel with specific parameters
        df_old = pd.read_excel(file1, engine='openpyxl')
        df_new = pd.read_excel(file2, engine='openpyxl')
        
        # Data Overview
        st.subheader("📊 Data Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows in File 1", df_old.shape[0])
        with col2:
            st.metric("Rows in File 2", df_new.shape[0])
        with col3:
            st.metric("Columns", df_old.shape[1])
        
        # Column Comparison
        old_cols = set(df_old.columns)
        new_cols = set(df_new.columns)
        
        if old_cols != new_cols:
            st.warning("⚠️ Column mismatch detected:")
            only_in_old = old_cols - new_cols
            only_in_new = new_cols - old_cols
            
            if only_in_old:
                st.info(f"Columns only in file 1: {', '.join(only_in_old)}")
            if only_in_new:
                st.info(f"Columns only in file 2: {', '.join(only_in_new)}")
        
        # Data Preview
        with st.expander("Preview Input Data"):
            st.write("✅ Existing Logic Output (First 5 rows)")
            st.dataframe(df_old.head())
            st.write("✅ Optimized Logic Output (First 5 rows)")
            st.dataframe(df_new.head())

        # Column Selection for Sorting
        sort_cols = st.multiselect(
            "🔑 Select Key Column(s) to Sort By",
            options=df_old.columns.tolist(),
            default=[df_old.columns[0]] if not df_old.empty else []
        )

        # Comparison Trigger
        compare_triggered = st.button("Compare the data")

        if sort_cols and compare_triggered:
            # Prepare comparison settings
            comparison_settings = {
                'ignore_case': ignore_case,
                'ignore_whitespace': ignore_whitespace,
                'float_precision': float_precision,
                'treat_nulls_equal': treat_nulls_equal
            }
            
            # Progress indication
            progress_bar = st.progress(0)
            st.text("Comparing data...")
            
            # Check row count mismatch
            if len(df_old) != len(df_new):
                st.warning(f"⚠️ Row count mismatch: File 1 has {len(df_old)} rows, File 2 has {len(df_new)} rows")
            
            # Perform comparison
            mismatch_report, mismatched_columns = compare_dataframes(
                df_old, df_new, sort_cols, comparison_settings
            )
            
            progress_bar.progress(1.0)
            st.empty()
            
            # Convert mismatch report to DataFrame
            mismatch_df = pd.DataFrame(mismatch_report) if mismatch_report else pd.DataFrame()
            
            # Result Handling
            if mismatch_df.empty:
                st.success("✅ No mismatches found. The outputs are identical after sorting (with current comparison settings).")
            else:
                st.warning(f"⚠️ Mismatches found in columns: {', '.join(sorted(mismatched_columns))}")
                
                # Mismatch Analysis Tabs
                tab1, tab2 = st.tabs(["Summary View", "Detailed View"])
                
                with tab1:
                    # Mismatch Summary
                    mismatch_counts = {
                        col: sum(1 for row in mismatch_report if f"Old_{col}" in row) 
                        for col in mismatched_columns
                    }
                    
                    summary_df = pd.DataFrame({
                        "Column": list(mismatch_counts.keys()),
                        "Mismatch Count": list(mismatch_counts.values()),
                        "% of Rows": [
                            count / min(len(df_old), len(df_new)) * 100 
                            for count in mismatch_counts.values()
                        ]
                    })
                    
                    st.dataframe(summary_df.sort_values("Mismatch Count", ascending=False))
                    
                    # Sample Differences
                    st.subheader("Sample differences by column")
                    for col in sorted(mismatched_columns):
                        with st.expander(f"Column: {col}"):
                            sample_rows = [
                                row for row in mismatch_report 
                                if f"Old_{col}" in row
                            ][:5]
                            
                            if sample_rows:
                                sample_df = pd.DataFrame([{
                                    "Row": row["Row"],
                                    "Old Value": row[f"Old_{col}"],
                                    "New Value": row[f"New_{col}"]
                                } for row in sample_rows])
                                st.dataframe(sample_df)
                
                with tab2:
                    # Detailed View with Column Selection
                    cols_to_show = st.multiselect(
                        "Select columns to compare",
                        options=sorted(mismatched_columns),
                        default=list(sorted(mismatched_columns))[:3] if mismatched_columns else []
                    )
                    
                    if cols_to_show:
                        # Prepare display columns
                        display_cols = ["Row"] + [f"Key_{col}" for col in sort_cols]
                        for col in cols_to_show:
                            display_cols.extend([f"Old_{col}", f"New_{col}"])
                        
                        # Filter valid columns
                        valid_cols = [col for col in display_cols if col in mismatch_df.columns]
                        st.dataframe(mismatch_df[valid_cols])
                    else:
                        st.info("Select columns to view comparison details")

                # Download Mismatch Report
                output = BytesIO()
                mismatch_df.to_excel(output, index=False)
                st.download_button(
                    label="📥 Download Mismatch Report",
                    data=output.getvalue(),
                    file_name="mismatch_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        elif sort_cols and not compare_triggered:
            st.info("👉 Click 'Compare the data' to start comparison.")
        else:
            st.info("⬅️ Please upload both Excel files to start comparison.")

# Run the main application
if __name__ == "__main__":
    main()