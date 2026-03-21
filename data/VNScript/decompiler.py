import os
import struct
import sys
import json

"""
This program is used to decompile .mpk or .npa files from the Steins;Gate VN.
Usage: python decompiler.py <file.mpk or file.npa>
"""

class MPKDecompiler:
    @staticmethod
    def decompile(filepath):
        print(f"Trying to read from file {filepath}")
        if not os.path.exists(filepath):
            print(f"Error: File {filepath} not found.")
            return

        with open(filepath, 'rb') as f:
            # Read magic and header
            magic = f.read(4).decode('ascii')
            if magic != "MPK\0":
                print(f"Magic mismatch: expected 'MPK\\0', got {magic}")
                # return # Some files might have different magic but we'll try to proceed if it matches format

            f.read(2) # unused
            header_version = struct.unpack('<H', f.read(2))[0]
            file_count = struct.unpack('<I', f.read(4))[0]

            print(f"Your file is a MPK format / STEAM EDITION (Header Version: {header_version}, File Count: {file_count})")

            # Header buffer
            header_size = 0x38 + (file_count * 0x100)
            header_data = f.read(header_size)

            directory = f"dir_{os.path.basename(filepath)}"
            os.makedirs(directory, exist_ok=True)

            log_path = f"{filepath}.log"
            meta_json_path = os.path.join(directory, f"{os.path.basename(filepath)}.json")

            meta = []
            buffer_offset = 0x38
            first_file = True

            with open(log_path, 'w', encoding='utf-8') as log_file:
                for i in range(file_count):
                    file_num = struct.unpack('<I', header_data[buffer_offset:buffer_offset+4])[0]
                    buffer_offset += 4
                    
                    if file_num == 0 and not first_file:
                        break
                    first_file = False

                    offset = struct.unpack('<Q', header_data[buffer_offset:buffer_offset+8])[0]
                    buffer_offset += 8
                    length1 = struct.unpack('<Q', header_data[buffer_offset:buffer_offset+8])[0]
                    buffer_offset += 8
                    length2 = struct.unpack('<Q', header_data[buffer_offset:buffer_offset+8])[0]
                    buffer_offset += 8

                    filename_bytes = header_data[buffer_offset:buffer_offset+0xE4]
                    filename = filename_bytes.split(b'\0')[0].decode('utf-8', errors='ignore')
                    buffer_offset += 0xE4

                    print(f"[+]{filename}\tOffset[{offset:X}]\tSize[{length1:X}]")
                    log_file.write(f"[+]{filename}\tOffset[{offset:X}]\tSize[{length1:X}]\n")

                    meta.append({"index": file_num, "name": filename})

                    # Extract data
                    out_path = os.path.join(directory, filename.replace('\\', os.sep))
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)

                    current_pos = f.tell()
                    f.seek(offset)
                    data = f.read(length1)
                    with open(out_path, 'wb') as out_file:
                        out_file.write(data)
                    f.seek(current_pos)

            with open(meta_json_path, 'w', encoding='utf-8') as mj:
                json.dump(meta, mj, indent=4)

            print(f"\nExtraction finished. Files are in {directory}")

class NPADecompiler:
    KEY = bytearray(b"BUCKTICK")
    KEY_LEN = 8

    @classmethod
    def scramble_key(cls):
        for i in range(cls.KEY_LEN):
            cls.KEY[i] = (~cls.KEY[i]) & 0xFF

    @classmethod
    def decrypt_buffer(cls, buffer):
        for i in range(len(buffer)):
            buffer[i] ^= cls.KEY[i % cls.KEY_LEN]

    @classmethod
    def decompile(cls, filepath):
        print(f"Trying to read from file {filepath}")
        if not os.path.exists(filepath):
            print(f"Error: File {filepath} not found.")
            return

        cls.scramble_key()

        with open(filepath, 'rb') as f:
            header_len_data = f.read(4)
            if not header_len_data: return
            header_len = struct.unpack('<I', header_len_data)[0]

            print("Your file is a NPA format / JAST USA EDITION")
            print("I'm decrypting the header...")

            header_bytes = bytearray(f.read(header_len))
            cls.decrypt_buffer(header_bytes)

            file_count = struct.unpack('<I', header_bytes[0:4])[0]
            buffer_offset = 4

            log_path = f"{filepath}.log"
            with open(log_path, 'w', encoding='utf-8') as log_file:
                for i in range(file_count):
                    name_len = struct.unpack('<I', header_bytes[buffer_offset:buffer_offset+4])[0]
                    name_bytes = header_bytes[buffer_offset+4 : buffer_offset+4+name_len]
                    name = name_bytes.decode('utf-16-le')

                    size = struct.unpack('<I', header_bytes[buffer_offset + name_len + 4 : buffer_offset + name_len + 8])[0]
                    offset = struct.unpack('<I', header_bytes[buffer_offset + name_len + 8 : buffer_offset + name_len + 12])[0]
                    # unk = struct.unpack('<I', header_bytes[buffer_offset + name_len + 12 : buffer_offset + name_len + 16])[0]

                    print(f"[+]{name}\tOffset[{offset:X}]\tSize[{size:X}]")
                    log_file.write(f"[+]{name}\tOffset[{offset:X}]\tSize[{size:X}]\n")

                    buffer_offset += name_len + 16

                    # Extract data
                    os.makedirs(os.path.dirname(name) or '.', exist_ok=True)
                    current_pos = f.tell()
                    f.seek(offset)
                    data = bytearray(f.read(size))
                    cls.decrypt_buffer(data)
                    with open(name, 'wb') as out_file:
                        out_file.write(data)
                    f.seek(current_pos)

            print(f"\nExtraction finished. Files extracted.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python decompiler.py <file.mpk or file.npa>")
        sys.exit(1)

    target_file = sys.argv[1]
    ext = os.path.splitext(target_file)[1].lower()

    if ext == ".mpk":
        MPKDecompiler.decompile(target_file)
    elif ext == ".npa":
        NPADecompiler.decompile(target_file)
    else:
        print(f"Unknown file format: {ext}")
