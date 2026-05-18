#!/usr/bin/env bash
set -u

REPO="${NEMO_AUTOPUSH_REPO:-/home/cgandycl/Nemotron}"
APP_REL="use-case-examples/RAG Agent with Nemotron RAG Models"
REMOTE="${NEMO_AUTOPUSH_REMOTE:-autosave}"
PUSH_BRANCH="${NEMO_AUTOPUSH_PUSH_BRANCH:-mnemosyne-auto}"
LOG_FILE="${NEMO_AUTOPUSH_LOG:-$HOME/.local/state/nemo-auto-git-push.log}"
LOCK_FILE="${NEMO_AUTOPUSH_LOCK:-$HOME/.cache/nemo-auto-git-push.lock}"

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$LOCK_FILE")"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >> "$LOG_FILE"
}

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "another auto-push run is active; skipping"
  exit 0
fi

cd "$REPO" || {
  log "repo not found: $REPO"
  exit 1
}

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  log "not a git repository: $REPO"
  exit 1
fi

BRANCH="${NEMO_AUTOPUSH_BRANCH:-$(git branch --show-current)}"
if [ -z "$BRANCH" ]; then
  log "detached HEAD; skipping"
  exit 1
fi

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  log "remote not found: $REMOTE"
  exit 1
fi

git_dir="$(git rev-parse --git-dir)"
if [ -d "$git_dir/rebase-merge" ] || [ -d "$git_dir/rebase-apply" ] || [ -f "$git_dir/MERGE_HEAD" ]; then
  log "merge or rebase in progress; skipping"
  exit 1
fi

export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-Mnemosyne Auto Push}"
export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-mnemosyne-auto-push@localhost}"
export GIT_COMMITTER_NAME="${GIT_COMMITTER_NAME:-$GIT_AUTHOR_NAME}"
export GIT_COMMITTER_EMAIL="${GIT_COMMITTER_EMAIL:-$GIT_AUTHOR_EMAIL}"

git update-index -q --refresh >/dev/null 2>&1 || true

# Stage only source/UI files from the local RAG app plus this automation script.

safe_paths=(
  "scripts/nemo-auto-git-push.sh"
  "$APP_REL/rag_api.py"
  "$APP_REL/_nv_ingest_helper.py"
  "$APP_REL/migrate_collection_to_nvidia.py"
  "$APP_REL/claude_sync.py"
  "$APP_REL/clifflabs_chat_sync.py"
  "$APP_REL/gmail_sync.py"
  "$APP_REL/multi_chat_sync.py"
  "$APP_REL/gmail_memory_first_run.sh"
  "$APP_REL/run_nvidia_rag.sh"
  "$APP_REL/frontend"
)

for path in "${safe_paths[@]}"; do
  if [ -e "$path" ]; then
    git add -- "$path"
  fi
done

shopt -s nullglob
noise_paths=(
  "$APP_REL"/*.log
  "$APP_REL"/*.pid
  "$APP_REL"/*.json
  "$APP_REL"/*.bak
  "$APP_REL"/*.bak-*
  "$APP_REL"/*.bak.*
)

if [ "${#noise_paths[@]}" -gt 0 ]; then
  git reset -q -- "${noise_paths[@]}" >/dev/null 2>&1 || true
fi

committed=0
if ! git diff --cached --quiet; then
  message="chore: auto-push $(date -u '+%Y-%m-%d %H:%M UTC')"
  if git commit -m "$message" >> "$LOG_FILE" 2>&1; then
    committed=1
    log "committed changes: $message"
  else
    log "commit failed"
    exit 1
  fi
fi

remote_sha="$(git ls-remote "$REMOTE" "refs/heads/$PUSH_BRANCH" 2>>"$LOG_FILE" | awk '{print $1}' || true)"
local_sha="$(git rev-parse HEAD)"

if [ "$committed" -eq 0 ] && [ -n "$remote_sha" ] && [ "$remote_sha" = "$local_sha" ]; then
  log "no committed changes to push"
  exit 0
fi

if git push "$REMOTE" "HEAD:refs/heads/$PUSH_BRANCH" >> "$LOG_FILE" 2>&1; then
  log "pushed HEAD to $REMOTE/$PUSH_BRANCH"
else
  log "push failed"
  exit 1
fi
