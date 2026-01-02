
import os

file_path = 'd:\\Internship projects\\backend\\requirements.txt'

# Try to detect encoding or just try common ones
encodings = ['utf-16', 'utf-8', 'cp1252', 'utf-16-le', 'utf-16-be']
content = ""
used_encoding = 'utf-8'

for enc in encodings:
    try:
        with open(file_path, 'r', encoding=enc) as f:
            content = f.read()
        used_encoding = enc
        print(f"Successfully read requirements.txt with encoding: {enc}")
        break
    except Exception as e:
        continue

if not content:
    print("Could not read requirements.txt")
    exit(1)

required_packages = [
    'chromadb',
    'google-generativeai',
    'django-recaptcha',
    'gunicorn',
    'uvicorn',
    'python-docx',
    'scikit-learn',
    'numpy',
    'requests',
    'python-dotenv'
]

with open(file_path, 'a', encoding=used_encoding) as f:
    added = False
    for package in required_packages:
        # Simple check: search for package name (ignoring version specifiers for now)
        # This isn't perfect but handles most cases where package is completely missing
        if package.lower() not in content.lower():
            print(f"Adding {package} to {file_path}")
            if content and not content.endswith('\n') and not added:
                f.write('\n')
                added = True
            elif not added and not content.endswith('\n'):
                 f.write('\n')
                 added = True
            
            f.write(f'{package}\n')
            
print("Requirements update complete.")
