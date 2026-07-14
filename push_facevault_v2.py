"""
FaceVault GitHub Push Script v2 — Correct folder structure
Run from ANYWHERE: python push_facevault_v2.py YOUR_TOKEN PATH_TO_FACEVAULT_FOLDER

Example:
  python push_facevault_v2.py ghp_xxx D:\Downloads\Code\.Repos\My\FaceVault\facevault
  
If no path given, looks for facevault/ next to this script.
"""
import sys, os
from pathlib import Path

TOKEN        = sys.argv[1] if len(sys.argv) > 1 else input("Token: ").strip()
PROJECT_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else None

try:
    from github import Github, Auth
except ImportError:
    os.system("pip install PyGithub -q")
    from github import Github, Auth

g    = Github(auth=Auth.Token(TOKEN))
user = g.get_user()
print(f"Logged in as: {user.login}")

# Find project root — must contain demo.py or README.md + Utils/
if PROJECT_PATH is None:
    script_dir = Path(__file__).parent
    for candidate in [script_dir / "facevault", script_dir]:
        if (candidate / "demo.py").exists() or (candidate / "Utils").exists():
            PROJECT_PATH = candidate
            break

if PROJECT_PATH is None or not PROJECT_PATH.exists():
    print("ERROR: Can't find the facevault project folder.")
    print("Usage: python push_facevault_v2.py TOKEN C:\\path\\to\\facevault")
    sys.exit(1)

print(f"Project root: {PROJECT_PATH}")

# Get repo
repo = user.get_repo("facevault")
print(f"Repo: {repo.html_url}")

# Collect all existing files in repo (to know what to update vs create)
def get_all_repo_files(repo):
    """Recursively get all file paths in repo."""
    existing = {}
    def _walk(contents):
        for item in contents:
            if item.type == "dir":
                _walk(repo.get_contents(item.path))
            else:
                existing[item.path] = item.sha
    try:
        _walk(repo.get_contents(""))
    except Exception as e:
        print(f"  Note: {e}")
    return existing

print("Scanning existing repo files...")
existing = get_all_repo_files(repo)
print(f"Found {len(existing)} existing files in repo")

# Collect all local files
SKIP = {"__pycache__", ".git", ".pytest_cache", "push_facevault", "push_facevault_v2"}
all_files = []
for fpath in sorted(PROJECT_PATH.rglob("*")):
    if not fpath.is_file():
        continue
    rel = str(fpath.relative_to(PROJECT_PATH)).replace("\\", "/")
    if any(s in rel for s in SKIP):
        continue
    all_files.append((rel, fpath))

print(f"Found {len(all_files)} local files to push\n")

pushed = updated = errors = 0

for rel, fpath in all_files:
    try:
        content = fpath.read_bytes()
    except Exception as e:
        print(f"  ✗ {rel}: read error {e}")
        errors += 1
        continue

    if rel in existing:
        # Update existing file
        try:
            repo.update_file(
                path    = rel,
                message = f"refactor: reorganize {rel} into correct folder structure",
                content = content,
                sha     = existing[rel],
                branch  = "main"
            )
            print(f"  ↑ {rel}")
            updated += 1
        except Exception as e:
            print(f"  ✗ {rel} (update): {e}")
            errors += 1
    else:
        # Create new file
        try:
            repo.create_file(
                path    = rel,
                message = f"feat: add {rel}",
                content = content,
                branch  = "main"
            )
            print(f"  ✓ {rel}")
            pushed += 1
        except Exception as e:
            print(f"  ✗ {rel} (create): {e}")
            errors += 1

# Delete old flat files that should now be in subfolders
flat_to_remove = [
    "pipeline.py", "test_pipeline.py", "push_facevault.py",
    "utils_cam.py", "utils_file.py", "utils_filters.py",
    "utils_nn.py", "utils_quant.py", "demo_tsne.png",
    "demo_heatmap.png", "facevault_demo_report.json",
]
print("\nCleaning up old flat files...")
for fname in flat_to_remove:
    if fname in existing:
        try:
            f = repo.get_contents(fname)
            repo.delete_file(
                path    = fname,
                message = f"cleanup: remove flat {fname} (now in correct subfolder)",
                sha     = f.sha,
                branch  = "main"
            )
            print(f"  🗑  deleted: {fname}")
        except Exception as e:
            print(f"  - {fname}: {e}")

print(f"\n{'='*50}")
print(f"  Created:  {pushed}")
print(f"  Updated:  {updated}")
print(f"  Errors:   {errors}")
print(f"  View at:  {repo.html_url}")
print(f"{'='*50}")
