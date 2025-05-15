#!/usr/bin/env python3
import sys
from pathlib import Path
from datetime import datetime, timedelta

def categorize_heart_rate(hr: int) -> str:
    """
    Map a heart rate value (in bpm) to an emotional state.
    Adjust the ranges and emotions as needed.
    """
    if hr < 60:
        return "Very Relaxed"
    elif hr < 75:
        return "Calm"
    elif hr < 90:
        return "Neutral"
    elif hr < 105:
        return "Focused"
    elif hr < 120:
        return "Anxious"
    elif hr < 140:
        return "Excited"
    elif hr < 160:
        return "Angry"
    else:
        return "Panic"

def parse_log_file(path: Path):
    """
    Read lines like "2025-05-14 22:54:34 – 155 bpm" from `path`,
    yield (datetime, int) pairs.
    """
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("–")  # split on en-dash
            if len(parts) != 2:
                continue
            
            ts_str, hr_part = parts
            ts = datetime.fromisoformat(ts_str.strip())
            hr = int(hr_part.strip().split()[0])
            
            entries.append((ts, hr))
    
    # Sort by timestamp, just in case
    entries.sort(key=lambda x: x[0])
    return entries

def generate_time_series(start_ts: datetime, end_ts: datetime):
    """
    Generate timestamps at 1-second increments (HH:MM:SS.000),
    plus a +5 ms offset (HH:MM:SS.005), starting right after 'start_ts'
    and stopping strictly before 'end_ts'.

    Example:
      If start_ts = 22:54:34.005 and end_ts = 22:54:39.000,
      this yields:
        22:54:35.000
        22:54:35.005
        22:54:36.000
        22:54:36.005
        ...
        22:54:38.000
        22:54:38.005
      (It won't include 22:54:39.000 because that's 'end_ts'.)
    """
    # 1) Find the next whole second after start_ts
    next_sec = start_ts.replace(microsecond=0)
    if start_ts.microsecond > 0:
        next_sec += timedelta(seconds=1)
    
    current = next_sec
    while current < end_ts:
        # HH:MM:SS.000
        yield current
        # HH:MM:SS.005
        candidate_5ms = current + timedelta(microseconds=5000)
        if candidate_5ms < end_ts:
            yield candidate_5ms
        
        current += timedelta(seconds=1)

def format_time_for_output(ts: datetime) -> str:
    """
    Return a string in the format HH.MM.SS.mmm
    Example: "22.54.35.005"
    """
    return ts.strftime("%H.%M.%S.%f")[:-3]

def main():
    script_dir = Path(__file__).resolve().parent
    txt_files = list(script_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt file found in {script_dir}")
        sys.exit(1)

    log_path = txt_files[0]
    output_path = script_dir / f"{log_path.stem}_categorized{log_path.suffix}"

    print(f"Reading heart-rate log from: {log_path.name}")
    print(f"Writing categorized output to: {output_path.name}\n")

    entries = parse_log_file(log_path)
    if not entries:
        print("No valid entries found in the log file.")
        sys.exit(0)

    with output_path.open("w", encoding="utf-8") as out_f:
        line_number = 1  # Start line count
        
        # 1) Handle the very first reading
        first_ts, first_hr = entries[0]
        first_state = categorize_heart_rate(first_hr)
        
        # Output the first reading's time
        ts_str = format_time_for_output(first_ts)
        out_f.write(f"Line ({line_number}) {ts_str} — {first_hr} bpm — {first_state}\n")
        line_number += 1
        
        ts_str_5ms = format_time_for_output(first_ts + timedelta(microseconds=5000))
        out_f.write(f"Line ({line_number}) {ts_str_5ms} — {first_hr} bpm — {first_state}\n")
        line_number += 1
        
        # 2) Loop through each consecutive pair of entries
        for i in range(len(entries) - 1):
            current_ts, current_hr = entries[i]
            next_ts, next_hr = entries[i + 1]
            current_state = categorize_heart_rate(current_hr)
            next_state = categorize_heart_rate(next_hr)

            # Fill-in timestamps AFTER current_ts, up to (but not including) next_ts
            for t in generate_time_series(current_ts, next_ts):
                fill_state = categorize_heart_rate(current_hr)
                t_str = format_time_for_output(t)
                out_f.write(f"Line ({line_number}) {t_str} — {current_hr} bpm — {fill_state}\n")
                line_number += 1
            
            # Write the "new" reading at next_ts (twice: .000 and .005)
            new_ts_str = format_time_for_output(next_ts)
            out_f.write(f"Line ({line_number}) {new_ts_str} — {next_hr} bpm — {next_state}\n")
            line_number += 1

            new_ts_str_5ms = format_time_for_output(next_ts + timedelta(microseconds=5000))
            out_f.write(f"Line ({line_number}) {new_ts_str_5ms} — {next_hr} bpm — {next_state}\n")
            line_number += 1

    print("Done.")

if __name__ == "__main__":
    main()
