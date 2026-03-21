import glob
import re
import csv
import os

def main():
    file_pattern = os.path.join("data", "SG*.SCX.txt")
    files = glob.glob(file_pattern)
    
    if not files:
        print("No files found matching data/SG*.SCX.txt")
        return

    files.sort()

    line_regex = re.compile(r"^\[name\](.*?)\[line\](.*)$")

    output_rows = []

    blocks = []
    current_speaker = None
    current_text = []

    for fw in files:
        with open(fw, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = line_regex.search(line)
            
            if match:
                name = match.group(1).strip()
                raw_text = match.group(2).strip()
            else:
                # If there's no [name] prefix, it's a thought/narration.
                # Skip these so they aren't included in the dialogue blocks.
                continue

            # Remove any trailing tags like [%p], [%e], or inline ones like [color...]
            text = re.sub(r"\[.*?\]", "", raw_text).strip()
            
            if not text:
                continue

            # Strip opening/closing quotes to keep the dialogue clean
            if text.startswith('“') and text.endswith('”'):
                text = text[1:-1]
            elif text.startswith('"') and text.endswith('"'):
                text = text[1:-1]

            # Accumulate multiple phrases from the same speaker
            if name == current_speaker:
                current_text.append(text)
            else:
                if current_speaker is not None:
                    # Join previous accumulated text into one block
                    blocks.append((current_speaker, " ".join(current_text)))
                current_speaker = name
                current_text = [text]

    # Don't forget the last block
    if current_speaker is not None:
        blocks.append((current_speaker, " ".join(current_text)))

    # Find where Kurisu speaks, and pair it with the preceding dialogue block
    for i in range(len(blocks)):
        speaker, text = blocks[i]
        if speaker.lower() == "kurisu":
            if i > 0:
                prev_speaker, prev_text = blocks[i - 1]
                # Skip if either speaker just says "..."
                if not prev_text.strip(' .') or not text.strip(' .'):
                    continue
                output_rows.append([prev_speaker, prev_text, text])
            else:
                # Skip if Kurisu just says "..."
                if not text.strip(' .'):
                    continue
                output_rows.append(["", "", text])

    csv_file = "data.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["character", "user", "makise"])
        writer.writerows(output_rows)

    print(f"Extraction complete! Saved {len(output_rows)} rows to {csv_file}.")

if __name__ == "__main__":
    main()
