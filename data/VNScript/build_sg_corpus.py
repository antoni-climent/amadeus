from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
TXT_FILES_DIR = Path(__file__).resolve().parent / "txt_files"
OUTPUT_FILE = Path(__file__).resolve().parent / "SG_corpus.txt"

LINE_PATTERN = re.compile(r"^\[name\](.*?)\[line\](.*)$", re.DOTALL)
COLOR_PATTERN = re.compile(r'\[color index="[^"]*"\]')
TAG_PATTERN = re.compile(r"\[[^\]]+\]")


def clean_text(text):
    text = COLOR_PATTERN.sub("", text)
    text = TAG_PATTERN.sub("", text)
    text = text.replace("\r\n", "\n").strip()
    text = " ".join(text.split())

    if len(text) >= 2 and text[0] in {'"', "“"} and text[-1] in {'"', "”"}:
        text = text[1:-1].strip()

    return text


def normalize_speaker(name):
    name = name.strip()
    if not name or name == "???":
        return "Unknown"
    return name


def parse_sg_file(path):
    events = []
    raw_parts = path.read_text(encoding="utf-8").split("[%p]")

    for raw_part in raw_parts:
        part = raw_part.strip()
        if not part:
            continue

        match = LINE_PATTERN.match(part)
        if match:
            speaker = normalize_speaker(match.group(1))
            content = clean_text(match.group(2))
        else:
            speaker = "Narrator"
            content = clean_text(part)

        if not content:
            continue

        if events and events[-1][0] == speaker:
            events[-1] = (speaker, f"{events[-1][1]} {content}".strip())
            continue

        events.append((speaker, content))

    return [f"{speaker}: {content}" for speaker, content in events]


def build_corpus(output_file=OUTPUT_FILE):
    sg_files = sorted(TXT_FILES_DIR.glob("SG*.txt"))
    if not sg_files:
        raise FileNotFoundError(f"No SG text files found in {TXT_FILES_DIR}")

    scenes = []
    for path in sg_files:
        events = parse_sg_file(path)
        if not events:
            continue
        scene_name = path.stem.replace(".SCX", "")
        scene_text = " ".join([f"Scene: {scene_name}", *events]).strip()
        scenes.append(scene_text)

    corpus = "\n".join(scenes).strip() + "\n"
    output_file.write_text(corpus, encoding="utf-8")
    return output_file, len(sg_files), len(scenes)


def main():
    output_file, file_count, scene_count = build_corpus()
    print(f"Wrote {scene_count} scenes from {file_count} files to {output_file}")


if __name__ == "__main__":
    main()
