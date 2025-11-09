#!/usr/bin/env bash
set -euo pipefail

# CONFIGURATION — edit as needed
REMOTE="origin"
BRANCHES="--all"
LFS_SIZE_THRESHOLD_MB=4096       # files larger than this size (in MB) will be targeted
KEEP_LOCAL_BACKUP_DIR="../repo‐backup‐$(date +%Y%m%d_%H%M%S)"

echo "=== BACKUP existing clone (just in case) ==="
git clone --mirror . "${KEEP_LOCAL_BACKUP_DIR}"
echo "Backup created at ${KEEP_LOCAL_BACKUP_DIR}"

echo "=== Fetch all LFS pointers and ensure local state ==="
git lfs fetch --all
git lfs ls-files --all > lfs_all.txt
echo "List of all LFS‐tracked files written to lfs_all.txt"

echo "=== Identify large LFS objects above threshold (${LFS_SIZE_THRESHOLD_MB} MB) ==="
awk -v thresh=${LFS_SIZE_THRESHOLD_MB} '
  { size=$1; unit=substr($2,length($2),1);
    if(unit=="K") size_mb=size/1024;
    else if(unit=="M") size_mb=size;
    else if(unit=="G") size_mb=size*1024;
    else size_mb=size/ (1024*1024);
    if(size_mb>=thresh) print $0;
  }' lfs_all.txt > lfs_large.txt

if [[ ! -s lfs_large.txt ]]; then
  echo "No LFS‐tracked files exceed ${LFS_SIZE_THRESHOLD_MB} MB. Exiting."
  exit 0
fi

echo "Large LFS objects identified:"
cat lfs_large.txt

echo "=== Generating list of file-paths to remove from LFS and history ==="
# Extract the file paths (assuming last column is path)
awk '{ print $NF }' lfs_large.txt > paths_to_purge.txt

echo "Paths to purge:"
cat paths_to_purge.txt

echo "=== Untrack these files from LFS and remove from working tree ==="
while IFS= read -r path; do
  echo "Processing: $path"
  git lfs untrack "$path" || true
  git rm --cached "$path" || true
done < paths_to_purge.txt

git commit -m "Remove large LFS objects and convert to normal git removal"

echo "=== Rewrite history to purge these paths from all branches & tags ==="
# Note: using git-filter-repo (must be installed) is recommended. :contentReference[oaicite:2]{index=2}
FILTER_CMD="git filter-repo --invert-paths"
for p in $(cat paths_to_purge.txt); do
  FILTER_CMD+=" --path $p"
done
FILTER_CMD+=" --refs $BRANCHES --force --replace-refs delete-no-add"

echo "Running: ${FILTER_CMD}"
eval "$FILTER_CMD"

echo "=== Clean up dangling objects locally ==="
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "=== Push rewritten history to remote ==="
git push ${REMOTE} --force --all
git push ${REMOTE} --force --tags

echo "=== Verify LFS pointers are gone locally ==="
git lfs ls-files
echo "If none listed (or only ones you expect), cleanup done locally."

echo "=== IMPORTANT NEXT STEPS ==="
echo "1. Tell all collaborators: **re-clone** the repository, because commits have been rewritten."
echo "2. On the hosting service (e.g. GitHub), check whether LFS storage usage has dropped. If not, you may need to open a support ticket to purge LFS objects."

echo "Done."
