
import os

file_path = 'requirements.txt'

# Try to detect encoding or just try common ones
encodings = ['utf-16', 'utf-8', 'cp1252']
content = ""
used_encoding = 'utf-8'

for enc in encodings:
    try:
        with open(file_path, 'r', encoding=enc) as f:
            content = f.read()
        used_encoding = enc
        break
    except Exception:
        continue

if 'chromadb' not in content.lower():
    print(f"Adding chromadb to {file_path} (encoding: {used_encoding})")
    with open(file_path, 'a', encoding=used_encoding) as f:
        # Ensure newline at start if needed
        if content and not content.endswith('\n'):
            f.write('\n')
        f.write('chromadb\n')
else:
    print("chromadb already in requirements.txt")
