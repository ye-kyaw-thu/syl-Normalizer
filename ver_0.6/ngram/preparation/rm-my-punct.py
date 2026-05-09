import argparse
import sys

def remove_punctuation(input_path, output_path):
    # Defining the specific Burmese punctuation marks
    # ၊ (U+104A) and ။ (U+104B)
    targets = ['၊', '။']
    
    try:
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                # Replace each target with an empty string
                clean_line = line
                for char in targets:
                    clean_line = clean_line.replace(char, '')
                
                # Write the cleaned line to the output file
                fout.write(clean_line)
                
        print(f"Success: Cleaned file saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: The file at {input_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove Burmese punctuation (၊ and ။) from a file.")
    
    # Adding the requested flags
    parser.add_argument("-i", "--input", required=True, help="Path to the input file")
    parser.add_argument("-o", "--output", required=True, help="Path to the output file")
    
    args = parser.parse_args()
    
    remove_punctuation(args.input, args.output)

