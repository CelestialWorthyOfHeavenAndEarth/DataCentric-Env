from huggingface_hub import HfApi, create_repo
import os

api = HfApi()
REPO_ID = "Aswini-Kumar/datacentric-env"

# Create the Space (Docker SDK, public)
print(f"Creating Space: {REPO_ID}")
try:
    create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )
    print("Space created (or already exists).")
except Exception as e:
    print(f"Space creation note: {e}")

# Upload all files
ROOT = os.path.dirname(os.path.abspath(__file__))

EXCLUDE = {".git", "__pycache__", ".pyc", "results.png", "datacentric-grpo", "datacentric-grpo-final"}

def should_skip(path):
    for ex in EXCLUDE:
        if ex in path:
            return True
    return False

uploaded = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    # skip excluded dirs in-place
    dirnames[:] = [d for d in dirnames if not should_skip(os.path.join(dirpath, d))]
    for filename in filenames:
        local_path = os.path.join(dirpath, filename)
        if should_skip(local_path):
            continue
        if filename == os.path.basename(__file__):
            continue  # don't upload this script itself
        rel_path = os.path.relpath(local_path, ROOT).replace("\\", "/")
        print(f"  Uploading: {rel_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=rel_path,
            repo_id=REPO_ID,
            repo_type="space",
        )
        uploaded.append(rel_path)

print(f"\nDone. {len(uploaded)} files uploaded to https://huggingface.co/spaces/{REPO_ID}")
