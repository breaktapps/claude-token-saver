#!/usr/bin/env bash
# bump-version.sh — Bump VERSION file and sync to all manifests
# Usage: ./scripts/bump-version.sh [major|minor|patch]
#
# Source of truth: ../../VERSION (project root)
# Syncs to: plugin.json, pyproject.toml, marketplace.json, plugin-metadata.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(cd "$PLUGIN_DIR/../.." && pwd)"
MARKETPLACE_DIR="$PROJECT_ROOT/products/marketplace"

VERSION_FILE="$PROJECT_ROOT/VERSION"
PLUGIN_JSON="$PLUGIN_DIR/.claude-plugin/plugin.json"
PYPROJECT_TOML="$PLUGIN_DIR/pyproject.toml"
MARKETPLACE_JSON="$MARKETPLACE_DIR/plugin-metadata.json"
MARKETPLACE_PLUGIN_JSON="$MARKETPLACE_DIR/.claude-plugin/marketplace.json"

BUMP_TYPE="${1:-patch}"

# Read current version
CURRENT_VERSION=$(cat "$VERSION_FILE" | tr -d '[:space:]')
if [[ -z "$CURRENT_VERSION" ]]; then
    echo "Error: VERSION file empty or missing at $VERSION_FILE" >&2
    exit 1
fi

# Compute new version
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"
case "$BUMP_TYPE" in
    major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
    minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
    patch) PATCH=$((PATCH + 1)) ;;
    *)
        echo "Error: bump type must be major, minor, or patch" >&2
        exit 1
        ;;
esac
NEW_VERSION="$MAJOR.$MINOR.$PATCH"

echo "Bumping: $CURRENT_VERSION → $NEW_VERSION ($BUMP_TYPE)"

# 1. Update VERSION file (source of truth)
echo "$NEW_VERSION" > "$VERSION_FILE"
echo "  ✓ VERSION"

# 2. Sync all manifests
python3 -c "
import json, re

new = '$NEW_VERSION'

# plugin.json
with open('$PLUGIN_JSON') as f: d = json.load(f)
d['version'] = new
with open('$PLUGIN_JSON', 'w') as f: json.dump(d, f, indent=2); f.write('\n')
print('  ✓ plugin.json')

# pyproject.toml
with open('$PYPROJECT_TOML') as f: c = f.read()
c = re.sub(r'^version\s*=\s*\"[^\"]+\"', f'version = \"{new}\"', c, flags=re.MULTILINE)
with open('$PYPROJECT_TOML', 'w') as f: f.write(c)
print('  ✓ pyproject.toml')

# marketplace plugin-metadata.json
with open('$MARKETPLACE_JSON') as f: d = json.load(f)
d['version'] = new
with open('$MARKETPLACE_JSON', 'w') as f: json.dump(d, f, indent=2); f.write('\n')
print('  ✓ plugin-metadata.json')

# marketplace marketplace.json
with open('$MARKETPLACE_PLUGIN_JSON') as f: d = json.load(f)
for p in d.get('plugins', []):
    if p.get('name') == 'claude-token-saver':
        p['version'] = new
with open('$MARKETPLACE_PLUGIN_JSON', 'w') as f: json.dump(d, f, indent=2); f.write('\n')
print('  ✓ marketplace.json')
"

echo ""
echo "Done: v$NEW_VERSION"
