# Extract the current directory from the script path
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Main directory is a parent directory for the current directory
MAIN_DIR="$(dirname "$CURRENT_DIR")"

# Run Qdrant from the snapshot by changing the CMD to the snapshot path
docker run -p 6333:6333 -p 6334:6334 \
  -v "$MAIN_DIR/shared_data/qdrant_snapshots:/snapshots" \
  qdrant/qdrant:latest \
  ./entrypoint.sh --snapshot "/snapshots/wands-products-2494690272794688-2024-05-28-15-02-03.snapshot:wands-products"