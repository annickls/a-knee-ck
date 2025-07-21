#!/usr/bin/env python3
"""
Script to add a column filled with zeros after trailing commas in a CSV file.
Just change the input_file name in the main() function and click run in VSCode!
"""
import os
import sys

def add_zero_after_trailing_commas(input_file, output_file=None):
    """
    Add a column filled with zeros after trailing commas in a CSV file.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str, optional): Path to the output file. If None, overwrites input file.
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    # If no output file specified, create a temporary file and then replace original
    if output_file is None:
        output_file = input_file + '.tmp'
        overwrite_original = True
    else:
        overwrite_original = False
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            with open(output_file, 'w', encoding='utf-8') as outfile:
                lines_processed = 0
                for line in infile:
                    # Add zero after trailing comma
                    if line.endswith(',\n'):
                        modified_line = line[:-1] + '0,\n'  # Replace comma+newline with comma+zero+newline
                    elif line.endswith(','):
                        modified_line = line + '0,'  # Add zero after comma at end of file
                    else:
                        modified_line = line  # Keep line as is
                    
                    outfile.write(modified_line)
                    lines_processed += 1
        
        # If we're overwriting the original file, replace it with the temp file
        if overwrite_original:
            os.replace(output_file, input_file)
            print(f"Successfully processed {lines_processed} lines in '{input_file}'")
        else:
            print(f"Successfully processed {lines_processed} lines from '{input_file}' to '{output_file}'")
        
        return True
    
    except Exception as e:
        print(f"Error processing file: {e}")
        # Clean up temp file if it exists
        if overwrite_original and os.path.exists(output_file):
            os.remove(output_file)
        return False

def main():
    """Main function - just click run in VSCode!"""
    # Change this to your CSV file path
    input_file = "data_20250711_150014.csv"  # Replace with your actual file name
    
    # Create output file name by adding "_with_zeros" before the extension
    base_name = os.path.splitext(input_file)[0]
    extension = os.path.splitext(input_file)[1]
    output_file = f"{base_name}_with_zeros{extension}"
    
    print(f"Processing file: {input_file}")
    print(f"Output will be saved to: {output_file}")
    
    success = add_zero_after_trailing_commas(input_file, output_file)
    
    if success:
        print("Process completed successfully!")
        print(f"Modified file saved as: {output_file}")
    else:
        print("Process failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()