#!/usr/bin/env bash
# Hit POST /api/rag/run-interpret for issue groups 101–105 with an optional pause between calls.
#
# Which table/columns supply context (e.g. tickets vs issue_group_comments) is configured on the
# worker only — {PREFIX}_ISSUE_GROUP_SOURCE_TABLE, _TEXT_COLUMNS, etc. This script does not pass those.
#
# Usage:
#   export WORKER_BASE_URL="https://your-service.up.railway.app"  # no trailing slash
#   export WORKER_API_KEY="..."   # if Railway has WORKER_API_KEY set (or tenant key)
#   ./scripts/run_interpret_issue_groups.sh
#
# Optional:
#   PROJECT_ID=greatrx DELAY_SECONDS=120 FIRST_ID=101 LAST_ID=105 ./scripts/run_interpret_issue_groups.sh

set -euo pipefail

BASE_URL="${WORKER_BASE_URL:-}"
TOKEN="${WORKER_API_KEY:-}"
PROJECT="${PROJECT_ID:-greatrx}"
DELAY="${DELAY_SECONDS:-120}"
FIRST="${FIRST_ID:-101}"
LAST="${LAST_ID:-105}"

if [[ -z "$BASE_URL" ]]; then
  echo "Set WORKER_BASE_URL to your deployed worker URL (e.g. https://xxx.up.railway.app)" >&2
  exit 1
fi

BASE_URL="${BASE_URL%/}"

_pretty_json() {
  if command -v jq >/dev/null 2>&1; then
    jq .
  else
    python3 -c 'import sys,json; print(json.dumps(json.load(sys.stdin), indent=2))'
  fi
}

for ((id = FIRST; id <= LAST; id++)); do
  echo "=== issue_group_id=${id} ==="
  hdr=(-H "Content-Type: application/json")
  if [[ -n "$TOKEN" ]]; then
    hdr+=(-H "Authorization: Bearer ${TOKEN}")
  fi
  code=$(curl -sS -o /tmp/run_interpret_resp.json -w "%{http_code}" "${hdr[@]}" \
    -X POST "${BASE_URL}/api/rag/run-interpret" \
    -d "{\"project_id\":\"${PROJECT}\",\"issue_group_id\":${id}}")
  echo "HTTP ${code}"
  _pretty_json < /tmp/run_interpret_resp.json || cat /tmp/run_interpret_resp.json
  echo
  if (( id < LAST )); then
    echo "Sleeping ${DELAY}s before next group..."
    sleep "${DELAY}"
  fi
done

echo "Done."
