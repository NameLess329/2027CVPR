---
name: git-local-publish
description: Build, initialize, commit, tag, and publish a local project to a GitHub remote. Use when the user asks to create or prepare a local Git repository, configure Git identity, create commits with user-specified commit messages, create tags with user-specified tag names/messages, add or verify GitHub remotes, and push branches/tags to the cloud. Default target: publish E:\Image_Decomposition\code to https://github.com/NameLess329/2027CVPR.git unless the user specifies another path or remote.
---

# Git Local Publish

Use this skill to turn a local project folder into a clean Git repository and publish it to GitHub. The default Git identity is:

- `user.name`: `NameLess329`
- `user.email`: `2280825040@qq.com`

## Default Project Target

When the user says to upload, publish, push, or sync the `code` folder without giving another destination, use:

- `repo_path`: `E:\Image_Decomposition\code`
- `remote_url`: `https://github.com/NameLess329/2027CVPR.git`
- `repo_name`: `2027CVPR`
- `owner`: `NameLess329`
- default branch: current branch if it exists, otherwise `main`

Still ask for or confirm the commit message and tag because the user wants to choose them at runtime.

Do not invent the commit message or tag unless the user explicitly asks you to choose one. If the user has not provided them, ask for:

- commit message
- tag name, if they want a tag
- tag message, if they want an annotated tag
- GitHub remote URL or target repository name, unless the default `NameLess329/2027CVPR.git` target clearly applies

## Safety Rules

- Never run destructive Git commands such as `git reset --hard`, `git clean -fd`, `git checkout -- <file>`, or force push unless the user explicitly asks for that exact operation.
- Before committing, inspect `git status --short` and avoid including unrelated user changes when working inside an existing repository.
- Before pushing, show or summarize the branch, remote, commit, and tag that will be pushed.
- Prefer annotated tags for releases: `git tag -a <tag> -m "<tag message>"`.
- Use the current branch unless the user asks to create or switch branches.
- If authentication fails, explain whether the user needs GitHub CLI login, HTTPS token authentication, or SSH key setup.

## Required Inputs

For a standard publish workflow, collect or infer:

- `repo_path`: local project directory. Default to `E:\Image_Decomposition\code` for this user's `code` folder.
- `remote_url`: GitHub remote URL. Default to `https://github.com/NameLess329/2027CVPR.git` for this user's `code` folder.
- `commit_message`: user-provided commit message.
- `tag_name`: optional user-provided tag, such as `v1.0.0`.
- `tag_message`: optional user-provided annotated tag message.
- `branch`: optional branch name, default to the current branch or `main` for a new repo.

## Workflow

### 1. Enter the repository

```powershell
Set-Location -LiteralPath "<repo_path>"
```

Verify the directory:

```powershell
Get-Location
Get-ChildItem -Force | Select-Object -First 20
```

### 2. Initialize Git if needed

Check whether the directory is already a Git repository:

```powershell
git rev-parse --is-inside-work-tree
```

If it is not a repository:

```powershell
git init
git branch -M main
```

### 3. Configure Git identity locally

Set identity in the repository, not globally, unless the user asks for global config:

```powershell
git config user.name "NameLess329"
git config user.email "2280825040@qq.com"
git config --get user.name
git config --get user.email
```

Use global config only on explicit request:

```powershell
git config --global user.name "NameLess329"
git config --global user.email "2280825040@qq.com"
```

### 4. Check branch, status, and ignored files

```powershell
git branch --show-current
git status --short
git status --ignored --short
```

If the repository has generated artifacts, model weights, datasets, caches, virtual environments, or secrets, ensure `.gitignore` excludes them before staging. Common entries:

```gitignore
.venv/
venv/
__pycache__/
*.pyc
.env
.env.*
*.log
outputs/
checkpoints/
models/
datasets/
data/
*.pt
*.pth
*.ckpt
*.safetensors
```

Only add broad ignore rules when they match the project. Do not hide required source files.

### 5. Add or verify the GitHub remote

Inspect existing remotes:

```powershell
git remote -v
```

If no `origin` exists:

```powershell
git remote add origin "<remote_url>"
```

For the default `code` upload target:

```powershell
git remote add origin "https://github.com/NameLess329/2027CVPR.git"
```

If `origin` exists but points to the wrong place, ask before changing it. With confirmation:

```powershell
git remote set-url origin "<remote_url>"
```

For the default target, the confirmed replacement command is:

```powershell
git remote set-url origin "https://github.com/NameLess329/2027CVPR.git"
```

### 6. Stage changes

For a new repository or a full intended publish:

```powershell
git add .
```

For an existing repository with mixed changes, stage only the requested files:

```powershell
git add -- "<file_or_directory>"
```

Review staged changes:

```powershell
git diff --cached --stat
git status --short
```

For large or suspicious additions, inspect before committing:

```powershell
git diff --cached --name-only
```

### 7. Commit with the user-specified message

If there are staged changes:

```powershell
git commit -m "<commit_message>"
```

If there are no staged changes, do not create an empty commit unless the user explicitly asks:

```powershell
git diff --cached --quiet
```

### 8. Create an optional tag

If the user provides `tag_name`, check whether it already exists:

```powershell
git tag --list "<tag_name>"
```

Prefer an annotated tag:

```powershell
git tag -a "<tag_name>" -m "<tag_message>"
```

If the user wants a lightweight tag:

```powershell
git tag "<tag_name>"
```

If the tag already exists, do not overwrite it unless the user explicitly asks. Tag replacement requires deleting and recreating the local tag, then force-updating the remote tag, which is risky.

### 9. Push branch and tags

Push the current branch to `origin` and set upstream:

```powershell
git push -u origin HEAD
```

If the user provided a tag, push only that tag:

```powershell
git push origin "<tag_name>"
```

If the user asks to push all tags:

```powershell
git push origin --tags
```

Avoid `--force` and `--force-with-lease` unless the user explicitly asks and understands the risk.

### 10. Verify the publish

```powershell
git status --short
git branch -vv
git remote -v
git log --oneline --decorate -n 5
git tag --list --sort=-creatordate | Select-Object -First 10
```

Summarize:

- repository path
- branch pushed
- remote URL
- commit hash and commit message
- tag pushed, if any
- any files intentionally ignored or skipped

## GitHub Repository Creation Options

If the remote repository does not exist, choose one of these routes.

### GitHub CLI

If `gh` is installed and logged in:

```powershell
gh auth status
gh repo create "NameLess329/<repo_name>" --private --source . --remote origin --push
```

For the default `2027CVPR` repository:

```powershell
gh repo create "NameLess329/2027CVPR" --private --source . --remote origin --push
```

For a public repository:

```powershell
gh repo create "NameLess329/<repo_name>" --public --source . --remote origin --push
```

For a public default `2027CVPR` repository:

```powershell
gh repo create "NameLess329/2027CVPR" --public --source . --remote origin --push
```

If the user already committed locally and wants tags pushed too:

```powershell
git push -u origin HEAD
git push origin "<tag_name>"
```

### Manual GitHub Creation

If `gh` is unavailable:

1. Ask the user to create an empty GitHub repository in the browser.
2. Use the repository URL as `remote_url`.
3. Add `origin`, then push with `git push -u origin HEAD`.

## Authentication Guidance

For SSH remotes:

```powershell
ssh -T git@github.com
```

If SSH authentication fails, the user needs to add their public key to GitHub.

For HTTPS remotes, GitHub requires a personal access token or Git credential manager login instead of an account password.

For GitHub CLI:

```powershell
gh auth login
gh auth status
```

## Common Failure Handling

- `fatal: remote origin already exists`: inspect `git remote -v`; use `git remote set-url origin "<remote_url>"` only after confirming the intended remote.
- `src refspec main does not match any`: no commit exists yet or the branch name differs; create a commit or push `HEAD`.
- `Updates were rejected`: remote has commits not present locally; inspect with `git fetch origin` and `git log --oneline --graph --decorate --all -n 20`, then choose merge/rebase only with user confirmation.
- `Authentication failed`: use `gh auth login`, SSH key setup, or HTTPS token authentication.
- Large file rejection: remove the large file from the index with `git rm --cached "<file>"`, add it to `.gitignore`, recommit or amend only with user confirmation. Consider Git LFS if the user needs to version large binaries.

## Preferred Final Response

After a successful run, report the result concisely:

```text
已完成 GitHub 发布：
- 路径: <repo_path>
- 分支: <branch>
- remote: <remote_url>
- commit: <short_hash> <commit_message>
- tag: <tag_name or none>
```

Mention any skipped files, authentication blockers, or commands the user must run manually.
