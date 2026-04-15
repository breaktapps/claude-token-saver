#!/bin/bash
# Uninstall claude-token-saver from ALL scopes
# Workaround for Claude Code uninstall scope detection bug

set -euo pipefail

PLUGIN_ID="claude-token-saver@breaktapps-plugins"
REMOVED=false

echo "==> Removing claude-token-saver from all scopes..."

# Try all scopes
for scope in user local project; do
  if claude plugin uninstall "$PLUGIN_ID" --scope "$scope" 2>/dev/null; then
    echo "    Removed from scope: $scope"
    REMOVED=true
  fi
done

# Clean settings.local.json in current project
LOCAL_SETTINGS=".claude/settings.local.json"
if [ -f "$LOCAL_SETTINGS" ] && grep -q "claude-token-saver" "$LOCAL_SETTINGS"; then
  python3 -c "
import json, sys
with open('$LOCAL_SETTINGS') as f:
    data = json.load(f)
plugins = data.get('enabledPlugins', {})
if '$PLUGIN_ID' in plugins:
    del plugins['$PLUGIN_ID']
if not plugins:
    data.pop('enabledPlugins', None)
with open('$LOCAL_SETTINGS', 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
"
  echo "    Cleaned $LOCAL_SETTINGS"
  REMOVED=true
fi

# Clean installed_plugins.json
INSTALLED="$HOME/.claude/plugins/installed_plugins.json"
if [ -f "$INSTALLED" ] && grep -q "claude-token-saver" "$INSTALLED"; then
  python3 -c "
import json
with open('$INSTALLED') as f:
    data = json.load(f)
plugins = data.get('plugins', {})
if '$PLUGIN_ID' in plugins:
    del plugins['$PLUGIN_ID']
with open('$INSTALLED', 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
"
  echo "    Cleaned installed_plugins.json"
  REMOVED=true
fi

# Remove cache
CACHE="$HOME/.claude/plugins/cache/breaktapps-plugins/claude-token-saver"
if [ -d "$CACHE" ]; then
  rm -rf "$CACHE"
  echo "    Removed cache: $CACHE"
  REMOVED=true
fi

if [ "$REMOVED" = true ]; then
  echo ""
  echo "Done. Run /reload-plugins to apply."
else
  echo "Plugin not found in any scope."
fi
