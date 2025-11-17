import os
import requests
import pandas as pd
import tabula
import fitz  # PyMuPDF
from PyPDF2 import PdfReader, PdfWriter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import logging
import time
import gc
import warnings
import platform
import subprocess
import sys
import tempfile
import shutil
from contextlib import contextmanager

warnings.filterwarnings('ignore')

# ------------------------------
# 1. CONFIGURATION
# ------------------------------
PDF_DIR = "data/pdf"
CHUNKS_DIR = "data/chunks"
OUTPUT_DIR = "data/output"
TEMP_DIR = "data/temp"
FINAL_CSV = "final_groundwater_data.csv"
CHUNK_SIZE = 50   # Number of pages per PDF chunk
MAX_RETRIES = 3   # Retry failed downloads

# Create directories
for dir_path in [PDF_DIR, CHUNKS_DIR, OUTPUT_DIR, TEMP_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('groundwater_extraction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure console encoding for Windows if necessary
if platform.system() == "Windows":
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except TypeError:
            pass

PDF_LINKS = [
    "https://cgwb.gov.in/sites/default/files/inline-files/pre-monsoon_1994-2003.pdf",
    # "https://cgwb.gov.in/sites/default/files/inline-files/pre-monsoon_2004-2013.pdf", 
    # "https://cgwb.gov.in/sites/default/files/inline-files/pre-monsoon_2014-2024.pdf",
    # "https://cgwb.gov.in/sites/default/files/inline-files/january_wl_1994-2024-compressed.pdf",
    # "https://cgwb.gov.in/sites/default/files/inline-files/august_wl_1994-2023_compressed.pdf",
    # "https://cgwb.gov.in/sites/default/files/inline-files/post-monsoon_wl_1994-2023_compressed.pdf"
]

# ------------------------------
# 2. MEMORY AND FILE MANAGEMENT
# ------------------------------
@contextmanager
def safe_pdf_context():
    """Context manager for safe PDF operations, ensuring garbage collection."""
    try:
        yield
    finally:
        gc.collect()
        time.sleep(0.1)

def force_cleanup():
    """Force garbage collection and add a brief pause for file handle release on Windows."""
    gc.collect()
    if platform.system() == "Windows":
        time.sleep(0.2)

# ------------------------------
# 3. PDF EXTRACTION LIBRARIES
# ------------------------------
def extract_with_pymupdf(pdf_path):
    """Extract tables using PyMuPDF (fitz)."""
    tables = []
    try:
        doc = fitz.open(pdf_path)
        logger.info(f"PyMuPDF: Processing {len(doc)} pages in {os.path.basename(pdf_path)}")
        for page_num, page in enumerate(doc):
            try:
                page_tables = page.find_tables()
                if page_tables:
                    for table_idx, table in enumerate(page_tables):
                        df = table.to_pandas()
                        if not df.empty and df.shape[0] > 1:
                            df['source_file'] = os.path.basename(pdf_path)
                            df['extraction_method'] = 'pymupdf'
                            df['page_number'] = page_num + 1
                            df['table_index'] = table_idx
                            tables.append(df)
            except Exception as page_error:
                logger.debug(f"PyMuPDF page {page_num} error: {str(page_error)}")
        doc.close()
        logger.info(f"PyMuPDF extracted {len(tables)} tables from {os.path.basename(pdf_path)}")
        return tables
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed for {os.path.basename(pdf_path)}: {str(e)}")
        return []

def extract_with_tabula_safe(pdf_path):
    """Extract tables using tabula with memory and area constraints."""
    try:
        tables = tabula.read_pdf(
            pdf_path,
            pages="all",
            multiple_tables=True,
            pandas_options={'header': None},
            silent=True,
            java_options="-Xmx512m"
        )
        valid_tables = []
        for i, df in enumerate(tables):
            if isinstance(df, pd.DataFrame) and not df.empty and df.shape[0] > 1:
                df['source_file'] = os.path.basename(pdf_path)
                df['extraction_method'] = 'tabula'
                df['table_index'] = i
                valid_tables.append(df)
        return valid_tables
    except Exception as e:
        logger.warning(f"Tabula extraction failed for {os.path.basename(pdf_path)}: {str(e)}")
        return []

def extract_with_pypdf2(pdf_path):
    """Fallback: Extract text using PyPDF2 and parse manually."""
    tables = []
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text:
                        # Attempt to parse structured text from the page
                        tables_from_page = parse_groundwater_text(text, pdf_path, page_num + 1)
                        tables.extend(tables_from_page)
                except Exception as page_error:
                    logger.debug(f"PyPDF2 page {page_num} error: {str(page_error)}")
        logger.info(f"PyPDF2 extracted {len(tables)} potential tables from {os.path.basename(pdf_path)}")
        return tables
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed for {os.path.basename(pdf_path)}: {str(e)}")
        return []

def parse_groundwater_text(text, source_file, page_num):
    """Parse groundwater-specific data from extracted raw text."""
    data_rows = []
    lines = text.split('\n')
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        
        numeric_parts = []
        text_parts = []
        for part in parts:
            try:
                val = float(part)
                # Check for plausible coordinate or depth values
                if (8 <= val <= 37) or (68 <= val <= 97) or (0 <= val <= 200):
                    numeric_parts.append(val)
                else:
                    text_parts.append(part)
            except ValueError:
                text_parts.append(part)
        
        # A valid row should have at least two numeric values (lat/lon) and some text (location)
        if len(numeric_parts) >= 2 and len(text_parts) >= 1:
            row_data = {
                'location_text': ' '.join(text_parts),
                'col_1': numeric_parts[0],
                'col_2': numeric_parts[1],
                'col_3': numeric_parts[2] if len(numeric_parts) > 2 else None,
            }
            data_rows.append(row_data)

    if data_rows:
        df = pd.DataFrame(data_rows)
        df['source_file'] = os.path.basename(source_file)
        df['page_number'] = page_num
        df['extraction_method'] = 'pypdf2_parsed'
        return [df]
    return []

# ------------------------------
# 4. PDF PROCESSING WORKFLOW
# ------------------------------
def process_pdf_with_fallbacks(pdf_path):
    """Process a single PDF using multiple libraries as fallbacks."""
    logger.info(f"Processing {os.path.basename(pdf_path)} with fallback methods...")
    extraction_methods = [
        ("PyMuPDF", extract_with_pymupdf),
        ("Tabula", extract_with_tabula_safe),
        ("PyPDF2", extract_with_pypdf2)
    ]
    
    for method_name, extraction_func in extraction_methods:
        try:
            logger.info(f"   Trying {method_name}...")
            with safe_pdf_context():
                tables = extraction_func(pdf_path)
                if tables:
                    logger.info(f"   {method_name} found {len(tables)} tables. Success!")
                    return tables
                else:
                    logger.info(f"   {method_name} found no tables.")
        except Exception as method_error:
            logger.warning(f"   {method_name} failed: {str(method_error)}")
        finally:
            force_cleanup()
    
    logger.warning(f"All extraction methods failed for {os.path.basename(pdf_path)}")
    return []

def safe_pdf_splitting(pdf_path, chunk_size=CHUNK_SIZE):
    """Splits a large PDF into smaller, more manageable chunks."""
    logger.info(f"Splitting {os.path.basename(pdf_path)}...")
    chunk_files = []
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            total_pages = len(reader.pages)
            if total_pages <= chunk_size:
                logger.info(f"PDF has {total_pages} pages, no chunking needed.")
                return [pdf_path]
            
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            for i in range(0, total_pages, chunk_size):
                writer = PdfWriter()
                end_page = min(i + chunk_size, total_pages)
                chunk_name = f"{pdf_name}_chunk_{i+1:04d}-{end_page:04d}.pdf"
                chunk_path = os.path.join(CHUNKS_DIR, chunk_name)
                
                for j in range(i, end_page):
                    writer.add_page(reader.pages[j])
                
                with open(chunk_path, "wb") as chunk_file:
                    writer.write(chunk_file)
                chunk_files.append(chunk_path)
                logger.debug(f"Created chunk: {chunk_name}")
        logger.info(f"Successfully split into {len(chunk_files)} chunks.")
        return chunk_files
    except Exception as e:
        logger.error(f"Error splitting PDF {os.path.basename(pdf_path)}: {str(e)}")
        return [pdf_path] # Return original file as a fallback

# ------------------------------
# 5. DATA INGESTION (DOWNLOAD)
# ------------------------------
def download_pdf_with_retry(url, max_retries=MAX_RETRIES):
    """Downloads a PDF from a URL with retry logic."""
    fname = os.path.join(PDF_DIR, url.split("/")[-1])
    if os.path.exists(fname) and os.path.getsize(fname) > 10000:
        logger.info(f"File already exists: {os.path.basename(fname)}")
        return True
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {os.path.basename(fname)} (attempt {attempt + 1}/{max_retries})...")
            headers = {'User-Agent': 'Mozilla/5.0'}
            with requests.get(url, stream=True, timeout=60, headers=headers) as r:
                r.raise_for_status()
                with open(fname, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.info(f"Successfully downloaded: {os.path.basename(fname)}")
            return True
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
    
    logger.error(f"Failed to download {os.path.basename(fname)} after {max_retries} attempts.")
    return False

def download_all_pdfs():
    """Downloads all PDFs listed in the PDF_LINKS configuration."""
    logger.info("Starting PDF downloads...")
    success_count = sum(1 for url in PDF_LINKS if download_pdf_with_retry(url))
    logger.info(f"Successfully downloaded {success_count}/{len(PDF_LINKS)} PDFs.")
    return success_count > 0

# ------------------------------
# 6. SEQUENTIAL PROCESSING PIPELINE
# ------------------------------
def process_single_pdf_safe(pdf_path):
    """Processes a single PDF file, including splitting and extraction."""
    logger.info(f"Processing: {os.path.basename(pdf_path)}")
    try:
        with safe_pdf_context():
            chunk_files = safe_pdf_splitting(pdf_path)
        
        all_tables = []
        for chunk_file in chunk_files:
            try:
                with safe_pdf_context():
                    chunk_tables = process_pdf_with_fallbacks(chunk_file)
                    all_tables.extend(chunk_tables)
            except Exception as chunk_error:
                logger.error(f"Error processing chunk {os.path.basename(chunk_file)}: {str(chunk_error)}")
        
        logger.info(f"Extracted {len(all_tables)} tables from {os.path.basename(pdf_path)}")
        return all_tables
    except Exception as e:
        logger.error(f"Critical error processing {os.path.basename(pdf_path)}: {str(e)}")
        return []

def safe_sequential_extraction():
    """Processes all downloaded PDFs sequentially."""
    logger.info("Starting safe sequential PDF processing...")
    pdf_files = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        logger.error("No PDF files found in the data directory.")
        return []

    all_tables = []
    for pdf_path in pdf_files:
        pdf_tables = process_single_pdf_safe(pdf_path)
        all_tables.extend(pdf_tables)
        force_cleanup()

    logger.info(f"Processing complete! Extracted {len(all_tables)} total tables.")
    return all_tables

# ------------------------------
# 7. DATA CLEANING AND STANDARDIZATION
# ------------------------------
def identify_and_standardize_columns(df):
    """Identifies and renames columns based on content patterns."""
    df_clean = df.copy()
    for col in df_clean.columns:
        col_data = df_clean[col].astype(str).str.strip()
        # Regex to match latitude patterns (e.g., 8.0 to 37.0)
        # CORRECTED REGEX
        if col_data.str.match(r'^(?:[8-9]|[1-2]\d|3[0-7])(?:\.\d+)?$').any():
            df_clean = df_clean.rename(columns={col: 'latitude'})
            continue
        # Regex to match longitude patterns (e.g., 68.0 to 97.0)
        # CORRECTED REGEX
        if col_data.str.match(r'^(?:6[8-9]|[7-8]\d|9[0-7])(?:\.\d+)?$').any():
            df_clean = df_clean.rename(columns={col: 'longitude'})
            continue
    return df_clean

def enhanced_data_cleaning(dfs):
    """Cleans, merges, and standardizes all extracted dataframes."""
    if not dfs:
        logger.warning("No dataframes to clean.")
        return None
    
    logger.info(f"Cleaning {len(dfs)} raw dataframes...")
    try:
        # Concatenate all dataframes, handling potential errors
        merged_df = pd.concat(dfs, ignore_index=True, sort=False)
        logger.info(f"Initial merged shape: {merged_df.shape}")

        # Basic cleaning
        merged_df = merged_df.dropna(how='all').reset_index(drop=True)
        merged_df = merged_df.rename(columns=lambda x: str(x).strip())

        # Standardize column names
        merged_df = standardize_groundwater_columns(merged_df)
        
        # Clean data values
        merged_df = clean_groundwater_values(merged_df)
        
        # Final validation and filtering
        merged_df = validate_groundwater_data(merged_df)
        
        # Generate comprehensive summary (This part was previously unreachable)
        logger.info("--- CLEANED DATA SUMMARY ---")
        logger.info(f"   Final shape: {merged_df.shape}")
        logger.info(f"   Columns: {list(merged_df.columns)}")
        
        for col in ['state', 'district']:
            if col in merged_df.columns:
                logger.info(f"   Unique {col}s: {merged_df[col].nunique()}")
        
        if 'depth_to_water_level' in merged_df.columns:
            depth_stats = merged_df['depth_to_water_level'].describe()
            logger.info(f"   Depth stats: min={depth_stats['min']:.1f}m, max={depth_stats['max']:.1f}m, mean={depth_stats['mean']:.1f}m")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error during data cleaning: {str(e)}")
        return None

def standardize_groundwater_columns(df):
    """Standardizes column names using a mapping of common patterns."""
    column_mapping = {}
    for col in df.columns:
        col_str = str(col).lower().strip()
        if any(p in col_str for p in ['state', 'pradesh']): column_mapping[col] = 'state'
        elif any(p in col_str for p in ['district', 'dist']): column_mapping[col] = 'district'
        elif any(p in col_str for p in ['village', 'location']): column_mapping[col] = 'village'
        elif any(p in col_str for p in ['lat', 'north']): column_mapping[col] = 'latitude'
        elif any(p in col_str for p in ['lon', 'east']): column_mapping[col] = 'longitude'
        elif any(p in col_str for p in ['depth', 'dtwl', 'water_level']): column_mapping[col] = 'depth_to_water_level'
        elif any(p in col_str for p in ['date', 'year']): column_mapping[col] = 'date'
        elif any(p in col_str for p in ['well', 'station']): column_mapping[col] = 'well_id'
        
    df_mapped = df.rename(columns=column_mapping)
    
    # Infer coordinate columns if they weren't explicitly named
    if 'latitude' not in df_mapped.columns or 'longitude' not in df_mapped.columns:
        df_mapped = infer_coordinate_columns(df_mapped)
        
    logger.info(f"Standardized columns: {list(column_mapping.values())}")
    return df_mapped

def infer_coordinate_columns(df):
    """Infers coordinate columns by analyzing the numeric ranges of the data."""
    for col in df.columns:
        try:
            numeric_vals = pd.to_numeric(df[col], errors='coerce').dropna()
            if not numeric_vals.empty:
                # Check for latitude range (India: 8-37)
                if numeric_vals.between(8, 37).all() and 'latitude' not in df.columns:
                    df = df.rename(columns={col: 'latitude'})
                    logger.info(f"Inferred 'latitude' from column '{col}'.")
                # Check for longitude range (India: 68-97)
                elif numeric_vals.between(68, 97).all() and 'longitude' not in df.columns:
                    df = df.rename(columns={col: 'longitude'})
                    logger.info(f"Inferred 'longitude' from column '{col}'.")
        except Exception:
            continue
    return df

def clean_groundwater_values(df):
    """Cleans the data within key numeric and text columns."""
    numeric_cols = ['latitude', 'longitude', 'depth_to_water_level']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    text_cols = ['state', 'district', 'village']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace('nan', None)
            
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
    return df

def validate_groundwater_data(df):
    """Applies final validation rules to filter out invalid records."""
    initial_count = len(df)
    
    # Remove rows with invalid coordinates for India
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df = df[df['latitude'].between(8, 37) & df['longitude'].between(68, 97)]
    
    # Remove rows with unrealistic depth values
    if 'depth_to_water_level' in df.columns:
        df = df[df['depth_to_water_level'].between(0, 200)]
    
    # Drop rows that are still mostly empty
    df = df.dropna(how='all', subset=[c for c in df.columns if c not in ['source_file', 'extraction_method']])
    
    # Remove duplicates
    key_cols = [col for col in ['latitude', 'longitude', 'date'] if col in df.columns]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols, keep='first')
    
    final_count = len(df)
    logger.info(f"Validation complete. Retained {final_count}/{initial_count} records.")
    return df.reset_index(drop=True)

def validate_and_fix_csv_structure(df):
    """Ensures the dataframe is safe to save as a CSV."""
    logger.info("Validating final dataframe structure before saving...")
    if df is None or df.empty:
        return None
    
    # Remove columns that contain complex objects like lists or dicts
    problematic_cols = [col for col in df.columns if any(isinstance(val, (list, dict)) for val in df[col].dropna())]
    if problematic_cols:
        logger.warning(f"Removing problematic columns with complex objects: {problematic_cols}")
        df = df.drop(columns=problematic_cols)
        
    # Ensure all data is converted to a string representation to avoid CSV errors
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', ';', regex=False)
    
    return df

# ------------------------------
# 8. MAIN PIPELINE
# ------------------------------
def main():
    """Main pipeline to run the entire data extraction and processing workflow."""
    start_time = time.time()
    logger.info("--- Starting Groundwater Data Extraction Pipeline ---")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    
    # Step 1: Download PDFs
    if not download_all_pdfs():
        logger.error("No PDFs could be downloaded. Exiting pipeline.")
        return

    # Step 2: Extract data
    all_tables = safe_sequential_extraction()
    if not all_tables:
        logger.error("No tables could be extracted from any PDF. Exiting pipeline.")
        return

    # Step 3: Clean and standardize data
    final_df = enhanced_data_cleaning(all_tables)
    if final_df is None or final_df.empty:
        logger.error("No data remained after cleaning. Exiting pipeline.")
        return
        
    # Step 4: Validate structure for CSV output
    final_df = validate_and_fix_csv_structure(final_df)
    if final_df is None or final_df.empty:
        logger.error("No data after final validation. Exiting pipeline.")
        return

    # Step 5: Save final dataset
    final_csv_path = os.path.join(OUTPUT_DIR, FINAL_CSV)
    final_df.to_csv(final_csv_path, index=False, encoding='utf-8')
    logger.info(f"--- Pipeline Complete! ---")
    logger.info(f"Final dataset saved to: {final_csv_path}")
    
    end_time = time.time()
    logger.info(f"Total processing time: {(end_time - start_time) / 60:.2f} minutes")
    logger.info(f"Total records extracted: {len(final_df)}")

if __name__ == "__main__":
    main()