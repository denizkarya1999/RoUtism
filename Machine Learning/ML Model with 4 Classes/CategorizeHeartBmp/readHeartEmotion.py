import csv

def get_emotion(bpm_value):
    """
    A simple function to categorize emotion based on heart rate (bpm).
    Customize this however you like!
    """
    bpm = int(bpm_value)
    if bpm < 90:
        return "Calm"
    elif bpm < 100:
        return "Normal"
    elif bpm < 110:
        return "Elevated"
    else:
        return "High"
        
def process_heart_rate_logs(input_filename, output_filename):
    with open(input_filename, 'r', encoding='utf-8') as infile, \
         open(output_filename, 'w', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        
        # Skip the original header row
        header = next(reader, None)
        
        line_number = 1
        
        for row in reader:
            # Make sure the row has enough columns before accessing BPM
            if not row or len(row) < 4:
                continue
            
            bpm_value = row[3]  # 'bpm' is the 4th column in the CSV
            emotion = get_emotion(bpm_value)
            
            # Format how each line should look in the output .txt file
            # Example format: "1) 2025-05-15T22:01:30-04:00,Thursday, May 15,10:01:30 PM,94,...  [Emotion: Calm]"
            line_str = f"{line_number}) {','.join(row)}  [Emotion: {emotion}]"
            
            # Write this line to the .txt file, plus a newline
            outfile.write(line_str + "\n")
            
            line_number += 1

# Example usage
if __name__ == "__main__":
    # Read from HeartRate.txt and write the results to HeartRate_output.txt
    process_heart_rate_logs("HeartRate.txt", "HeartRate_output.txt")
    print("Finished processing! Check 'HeartRate_output.txt' for results.")
