"""
Helper script to package the project into a clean ZIP archive for Google Colab.
Excludes virtual environments, git cache, checkpoints, and temporary logs.
"""
import os
import zipfile

EXCLUDE_DIRS = {".git", ".gemini", "venv", ".venv", "env", "__pycache__", ".pytest_cache", ".idea", ".vscode"}
EXCLUDE_EXTS = {".pyc", ".log", ".tmp"}

def create_colab_zip(output_filename: str = "LLM_XRAY_PROJECT_COLAB.zip"):
    project_root = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(project_root, output_filename)
    
    print(f"📦 Packaging project for Google Colab into '{output_filename}'...")
    count = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(project_root):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]

            for file in files:
                if file == output_filename:
                    continue
                _, ext = os.path.splitext(file)
                if ext in EXCLUDE_EXTS:
                    continue

                abs_file_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_file_path, project_root)
                zipf.write(abs_file_path, rel_path)
                count += 1

    file_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"✅ Created '{output_filename}' ({file_size_mb:.2f} MB, {count} files).")
    print(f"👉 You can now upload this file to Google Colab!")

if __name__ == "__main__":
    create_colab_zip()

