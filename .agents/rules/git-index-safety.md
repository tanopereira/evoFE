# Git Index Safety Rules

## NEVER use `git update-index --assume-unchanged` or `--skip-worktree` on source files

This is an absolute prohibition. No exceptions.

### Why

assume-unchanged causes git to silently omit files from every git add, commit, and push.
git status falsely shows a clean tree. Work accumulates on disk but never reaches the remote.
Other machines receive stale versions. Work is irrecoverably lost when files are overwritten.

This caused severe data loss in evoFE with:
R/transformers.R, R/evolve.R, R/individual.R, R/topology.R, NAMESPACE, inst/viewer/index.html

### At the start of EVERY session on any evoFE machine, run:

  git ls-files -v | grep "^[a-z]" || echo "CLEAN: no assume-unchanged flags"

If any files appear, clear ALL of them immediately:

  git ls-files -v | grep "^[a-z]" | awk '{print $2}' | xargs git update-index --no-assume-unchanged

### Before editing any file that was previously assume-unchanged

1. READ the full file content before making any edits
2. Never overwrite a locally-modified file without first diffing it against HEAD

### Correct alternatives

Local config overrides    -> .gitignore
Experimental work         -> feature branch + git stash
Temporarily clean status  -> git stash push -m "description"
Build-generated files     -> .gitignore
