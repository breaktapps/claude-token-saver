#!/usr/bin/env bash
# bump-version.sh — Synchronize version across all manifests
# Usage: ./scripts/bump-version.sh [major|minor|patch]
# Example: ./scripts/bump-version.sh patch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(dirname "$SCRIPT_DIR")"
MARKETPLACE_DIR="$(dirname "$PLUGIN_DIR")/marketplace"

PLUGIN_JSON="$PLUGIN_DIR/.claude-plugin/plugin.json"
PYPROJECT_TOML="$PLUGIN_DIR/pyproject.toml"
MARKETPLACE_JSON="$MARKETPLACE_DIR/plugin-metadata.json"

BUMP_TYPE="${1:-patch}"

# Read current version from plugin.json
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 is required" >&2
    exit 1
fi

CURRENT_VERSION=$(python3 -c "import json; print(json.load(open('$PLUGIN_JSON'))['version'])")

if [[ -z "$CURRENT_VERSION" ]]; then
    echo "Error: Could not read current version from $PLUGIN_JSON" >&2
    exit 1
fi

IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

case "$BUMP_TYPE" in
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    patch)
        PATCH=$((PATCH + 1))
        ;;
    *)
        echo "Error: bump type must be major, minor, or patch" >&2
        echo "Usage: $0 [major|minor|patch]" >&2
        exit 1
        ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"

echo "Bumping version: $CURRENT_VERSION -> $NEW_VERSION ($BUMP_TYPE)"

# Update plugin.json
python3 - <<EOF
import json
with open('$PLUGIN_JSON', 'r') as f:
    data = json.load(f)
data['version'] = '$NEW_VERSION'
with open('$PLUGIN_JSON', 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
print("Updated: $PLUGIN_JSON")
EOF

# Update pyproject.toml
if [[ -f "$PYPROJECT_TOML" ]]; then
    python3 - <<EOF
import re
with open('$PYPROJECT_TOML', 'r') as f:
    content = f.read()
content = re.sub(r'^version\s*=\s*"[^"]+"', 'version = "$NEW_VERSION"', content, flags=re.MULTILINE)
with open('$PYPROJECT_TOML', 'w') as f:
    f.write(content)
print("Updated: $PYPROJECT_TOML")
EOF
fi

# Update marketplace metadata
if [[ -f "$MARKETPLACE_JSON" ]]; then
    python3 - <<EOF
import json
with open('$MARKETPLACE_JSON', 'r') as f:
    data = json.load(f)
data['version'] = '$NEW_VERSION'
with open('$MARKETPLACE_JSON', 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
print("Updated: $MARKETPLACE_JSON")
EOF
fi

echo "Version bumped to $NEW_VERSION successfully"
