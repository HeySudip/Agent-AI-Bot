import base64
import logging
from langchain.tools import tool
from config import load_config

logger = logging.getLogger(__name__)


def get_gh_client():
    """Return authenticated Github client or None."""
    cfg = load_config()
    token = cfg.get("github_token", "")
    if not token:
        return None
    from github import Github
    return Github(token)


def no_token() -> str:
    return "GitHub token not connected. Just paste your token (ghp_...) in the chat and I'll connect right away."


def fmt_repo(repo) -> str:
    desc = repo.description or "no description"
    vis = "🔒" if repo.private else "🌍"
    return f"{vis} [{repo.full_name}]({repo.html_url}) ⭐{repo.stargazers_count} — {desc}"


def build_github_tools() -> list:

    # ─────────────────────────────────────────────────
    # Repositories
    # ─────────────────────────────────────────────────

    @tool
    def github_list_my_repos(sort_by: str = "updated") -> str:
        """List all your GitHub repositories. sort_by: 'updated', 'created', 'pushed', 'full_name'"""
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            user = gh.get_user()
            repos = list(user.get_repos(sort=sort_by, direction="desc"))
            if not repos:
                return "You don't have any repositories yet."
            lines = [f"Found {len(repos)} repositories:\n"]
            for r in repos[:25]:
                lines.append(fmt_repo(r))
            if len(repos) > 25:
                lines.append(f"\n…and {len(repos) - 25} more.")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"github_list_my_repos error: {e}")
            return f"Error listing repos: {str(e)}"

    @tool
    def github_create_repo(
        name: str,
        description: str = "",
        private: bool = False,
        has_readme: bool = True,
        gitignore_template: str = "",
        license_template: str = "",
    ) -> str:
        """
        Create a new GitHub repository.
        name: repo name (no spaces, use hyphens)
        description: short description
        private: True for private repo
        has_readme: auto-initialize with README
        gitignore_template: e.g. 'Python', 'Node', 'Go'
        license_template: e.g. 'mit', 'apache-2.0', 'gpl-3.0'
        """
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            kwargs = {
                "name": name,
                "description": description,
                "private": private,
                "auto_init": has_readme,
            }
            if gitignore_template:
                kwargs["gitignore_template"] = gitignore_template
            if license_template:
                kwargs["license_template"] = license_template
            repo = gh.get_user().create_repo(**kwargs)
            visibility = "🔒 private" if private else "🌍 public"
            return (
                f"✅ Repo created ({visibility})\n"
                f"📌 **{repo.full_name}**\n"
                f"🔗 {repo.html_url}\n"
                f"📋 Clone: `git clone {repo.clone_url}`"
            )
        except Exception as e:
            return f"Error creating repo: {str(e)}"

    @tool
    def github_delete_repo(repo_full_name: str, confirmed: bool = False) -> str:
        """
        Delete a GitHub repository permanently.
        repo_full_name: 'owner/repo-name'
        confirmed: must be True to actually delete (prevents accidents)
        """
        gh = get_gh_client()
        if not gh:
            return no_token()
        if not confirmed:
            return (
                f"⚠️ Are you sure you want to delete **{repo_full_name}**? "
                f"This is permanent and cannot be undone. "
                f"Say 'yes delete {repo_full_name}' to confirm."
            )
        try:
            gh.get_repo(repo_full_name).delete()
            return f"✅ Repository **{repo_full_name}** has been permanently deleted."
        except Exception as e:
            return f"Error deleting repo: {str(e)}"

    @tool
    def github_get_repo_info(repo_full_name: str) -> str:
        """Get detailed information about a GitHub repository."""
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            r = gh.get_repo(repo_full_name)
            topics = ", ".join(r.get_topics()) or "none"
            open_issues = r.open_issues_count
            return (
                f"**{r.full_name}**\n"
                f"📝 {r.description or 'No description'}\n"
                f"🌍 Visibility: {'Private' if r.private else 'Public'}\n"
                f"⭐ Stars: {r.stargazers_count} | 🍴 Forks: {r.forks_count} | 👁 Watchers: {r.watchers_count}\n"
                f"🐛 Open issues: {open_issues}\n"
                f"💻 Language: {r.language or 'Unknown'}\n"
                f"🌿 Default branch: {r.default_branch}\n"
                f"🏷 Topics: {topics}\n"
                f"📅 Created: {r.created_at.strftime('%Y-%m-%d')}\n"
                f"🔄 Last updated: {r.updated_at.strftime('%Y-%m-%d %H:%M')}\n"
                f"🔗 {r.html_url}\n"
                f"📋 Clone: `{r.clone_url}`"
            )
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def github_fork_repo(repo_full_name: str) -> str:
        """Fork a public GitHub repository to your account."""
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            fork = gh.get_user().create_fork(repo)
            return (
                f"✅ Forked **{repo_full_name}** to your account!\n"
                f"🔗 Your fork: {fork.html_url}\n"
                f"📋 Clone: `git clone {fork.clone_url}`"
            )
        except Exception as e:
            return f"Error forking repo: {str(e)}"

    @tool
    def github_search_repos(query: str, language: str = "", sort: str = "stars", limit: int = 6) -> str:
        """
        Search GitHub for public repositories.
        query: search keywords
        language: filter by language, e.g. 'python', 'javascript'
        sort: 'stars', 'forks', 'updated'
        """
        try:
            gh = get_gh_client()
            if not gh:
                from github import Github
                gh = Github()
            q = query
            if language:
                q += f" language:{language}"
            repos = gh.search_repositories(query=q, sort=sort)
            results = []
            count = min(limit, 8)
            for repo in list(repos[:count]):
                results.append(
                    f"⭐ {repo.stargazers_count:,} | **[{repo.full_name}]({repo.html_url})**\n"
                    f"   {repo.description or 'No description'}\n"
                    f"   💻 {repo.language or 'Unknown'} | 🍴 {repo.forks_count} forks"
                )
            return "\n\n".join(results) if results else "No repositories found."
        except Exception as e:
            return f"Error searching repos: {str(e)}"

    @tool
    def github_update_repo(
        repo_full_name: str,
        description: str = None,
        private: bool = None,
        has_issues: bool = None,
        has_wiki: bool = None,
        default_branch: str = None,
    ) -> str:
        """Update settings of a GitHub repository you own."""
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            kwargs = {}
            if description is not None:
                kwargs["description"] = description
            if private is not None:
                kwargs["private"] = private
            if has_issues is not None:
                kwargs["has_issues"] = has_issues
            if has_wiki is not None:
                kwargs["has_wiki"] = has_wiki
            if default_branch is not None:
                kwargs["default_branch"] = default_branch
            if not kwargs:
                return "No changes specified."
            repo.edit(**kwargs)
            return f"✅ Updated settings for **{repo_full_name}**"
        except Exception as e:
            return f"Error updating repo: {str(e)}"

    # ─────────────────────────────────────────────────
    # Files
    # ─────────────────────────────────────────────────

    @tool
    def github_list_files(repo_full_name: str, path: str = "", branch: str = "") -> str:
        """
        List files and folders in a GitHub repository directory.
        path: subdirectory path (leave empty for root)
        branch: branch name (leave empty for default branch)
        """
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            ref = branch or repo.default_branch
            contents = repo.get_contents(path, ref=ref)
            if not isinstance(contents, list):
                contents = [contents]
            dirs = [c for c in contents if c.type == "dir"]
            files = [c for c in contents if c.type == "file"]
            lines = [f"📁 `{repo_full_name}/{path or ''}` (branch: {ref})\n"]
            for d in sorted(dirs, key=lambda x: x.name):
                lines.append(f"📁 {d.name}/")
            for f in sorted(files, key=lambda x: x.name):
                size = f"{f.size:,} bytes" if f.size < 1024 else f"{f.size // 1024} KB"
                lines.append(f"📄 {f.name} ({size})")
            return "\n".join(lines)
        except Exception as e:
            return f"Error listing files: {str(e)}"

    @tool
    def github_read_file(repo_full_name: str, file_path: str, branch: str = "") -> str:
        """Read the contents of a file from a GitHub repository."""
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            ref = branch or repo.default_branch
            f = repo.get_contents(file_path, ref=ref)
            if isinstance(f, list):
                return f"'{file_path}' is a directory. Use github_list_files instead."
            content = f.decoded_content.decode("utf-8", errors="replace")
            size_kb = f.size / 1024
            return (
                f"📄 **{file_path}** ({size_kb:.1f} KB, branch: {ref})\n\n"
                f"```\n{content[:6000]}\n```"
                + (f"\n\n_(file truncated — {len(content)} total chars)_" if len(content) > 6000 else "")
            )
        except Exception as e:
            return f"Error reading file: {str(e)}"

    @tool
    def github_create_or_update_file(
        repo_full_name: str,
        file_path: str,
        content: str,
        commit_message: str = "",
        branch: str = "",
    ) -> str:
        """
        Create a new file or update an existing file in a GitHub repository.
        repo_full_name: 'owner/repo'
        file_path: path like 'src/main.py' or 'README.md'
        content: full file content (text)
        commit_message: git commit message
        branch: branch name (default branch if empty)
        """
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            ref = branch or repo.default_branch
            msg = commit_message or f"Update {file_path}"
            try:
                existing = repo.get_contents(file_path, ref=ref)
                repo.update_file(
                    path=file_path,
                    message=msg,
                    content=content,
                    sha=existing.sha,
                    branch=ref,
                )
                action = "Updated"
            except Exception:
                repo.create_file(
                    path=file_path,
                    message=msg,
                    content=content,
                    branch=ref,
                )
                action = "Created"
            return (
                f"✅ {action} `{file_path}` in **{repo_full_name}**\n"
                f"🌿 Branch: {ref}\n"
                f"💬 Commit: {msg}\n"
                f"🔗 {repo.html_url}/blob/{ref}/{file_path}"
            )
        except Exception as e:
            return f"Error saving file: {str(e)}"

    @tool
    def github_delete_file(
        repo_full_name: str,
        file_path: str,
        commit_message: str = "",
        branch: str = "",
    ) -> str:
        """Delete a file from a GitHub repository."""
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            ref = branch or repo.default_branch
            f = repo.get_contents(file_path, ref=ref)
            msg = commit_message or f"Delete {file_path}"
            repo.delete_file(file_path, msg, f.sha, branch=ref)
            return f"✅ Deleted `{file_path}` from **{repo_full_name}** (branch: {ref})"
        except Exception as e:
            return f"Error deleting file: {str(e)}"

    @tool
    def github_rename_file(
        repo_full_name: str,
        old_path: str,
        new_path: str,
        branch: str = "",
    ) -> str:
        """Rename or move a file in a GitHub repository (copy to new path + delete old)."""
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            ref = branch or repo.default_branch
            old_file = repo.get_contents(old_path, ref=ref)
            content = old_file.decoded_content.decode("utf-8", errors="replace")
            # Create at new path
            try:
                existing_new = repo.get_contents(new_path, ref=ref)
                repo.update_file(new_path, f"Move {old_path} to {new_path}", content, existing_new.sha, branch=ref)
            except Exception:
                repo.create_file(new_path, f"Move {old_path} to {new_path}", content, branch=ref)
            # Delete old path
            repo.delete_file(old_path, f"Remove {old_path} (moved to {new_path})", old_file.sha, branch=ref)
            return f"✅ Renamed `{old_path}` → `{new_path}` in **{repo_full_name}**"
        except Exception as e:
            return f"Error renaming file: {str(e)}"

    # ─────────────────────────────────────────────────
    # Branches
    # ─────────────────────────────────────────────────

    @tool
    def github_list_branches(repo_full_name: str) -> str:
        """List all branches in a GitHub repository."""
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            branches = list(repo.get_branches())
            default = repo.default_branch
            lines = [f"🌿 Branches in **{repo_full_name}** ({len(branches)} total):\n"]
            for b in branches:
                marker = " ← default" if b.name == default else ""
                lines.append(f"  • `{b.name}`{marker}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def github_create_branch(
        repo_full_name: str,
        branch_name: str,
        from_branch: str = "",
    ) -> str:
        """Create a new branch in a GitHub repository."""
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            source = from_branch or repo.default_branch
            source_branch = repo.get_branch(source)
            repo.create_git_ref(
                ref=f"refs/heads/{branch_name}",
                sha=source_branch.commit.sha,
            )
            return (
                f"✅ Branch `{branch_name}` created from `{source}` in **{repo_full_name}**"
            )
        except Exception as e:
            return f"Error creating branch: {str(e)}"

    @tool
    def github_delete_branch(repo_full_name: str, branch_name: str) -> str:
        """Delete a branch from a GitHub repository."""
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            ref = repo.get_git_ref(f"heads/{branch_name}")
            ref.delete()
            return f"✅ Deleted branch `{branch_name}` from **{repo_full_name}**"
        except Exception as e:
            return f"Error deleting branch: {str(e)}"

    # ─────────────────────────────────────────────────
    # Issues
    # ─────────────────────────────────────────────────

    @tool
    def github_list_issues(repo_full_name: str, state: str = "open", limit: int = 10) -> str:
        """
        List issues in a GitHub repository.
        state: 'open', 'closed', or 'all'
        """
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            issues = list(repo.get_issues(state=state))[:limit]
            if not issues:
                return f"No {state} issues found in **{repo_full_name}**."
            lines = [f"🐛 **{state.capitalize()} issues in {repo_full_name}:**\n"]
            for i in issues:
                label_str = ", ".join(l.name for l in i.labels) if i.labels else ""
                label_str = f" [{label_str}]" if label_str else ""
                lines.append(
                    f"#{i.number} — {i.title}{label_str}\n"
                    f"   Opened by @{i.user.login} · {i.created_at.strftime('%Y-%m-%d')}\n"
                    f"   {i.html_url}"
                )
            return "\n\n".join(lines)
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def github_create_issue(
        repo_full_name: str,
        title: str,
        body: str = "",
        labels: str = "",
        assignee: str = "",
    ) -> str:
        """
        Create an issue on a GitHub repository.
        labels: comma-separated label names, e.g. 'bug,help wanted'
        assignee: GitHub username to assign
        """
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            kwargs = {"title": title, "body": body}
            if labels:
                kwargs["labels"] = [l.strip() for l in labels.split(",")]
            if assignee:
                kwargs["assignee"] = assignee
            issue = repo.create_issue(**kwargs)
            return (
                f"✅ Issue created in **{repo_full_name}**\n"
                f"#{issue.number} — {issue.title}\n"
                f"🔗 {issue.html_url}"
            )
        except Exception as e:
            return f"Error creating issue: {str(e)}"

    @tool
    def github_close_issue(repo_full_name: str, issue_number: int, comment: str = "") -> str:
        """Close an issue in a GitHub repository, optionally adding a closing comment."""
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            issue = repo.get_issue(issue_number)
            if comment:
                issue.create_comment(comment)
            issue.edit(state="closed")
            return f"✅ Closed issue #{issue_number} in **{repo_full_name}**"
        except Exception as e:
            return f"Error closing issue: {str(e)}"

    @tool
    def github_comment_on_issue(repo_full_name: str, issue_number: int, comment: str) -> str:
        """Add a comment to an issue or pull request."""
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            issue = repo.get_issue(issue_number)
            c = issue.create_comment(comment)
            return f"✅ Comment added to #{issue_number}\n🔗 {c.html_url}"
        except Exception as e:
            return f"Error adding comment: {str(e)}"

    # ─────────────────────────────────────────────────
    # Pull Requests
    # ─────────────────────────────────────────────────

    @tool
    def github_list_pull_requests(repo_full_name: str, state: str = "open") -> str:
        """List pull requests in a GitHub repository. state: 'open', 'closed', 'all'"""
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            prs = list(repo.get_pulls(state=state))[:10]
            if not prs:
                return f"No {state} pull requests in **{repo_full_name}**."
            lines = [f"🔀 **{state.capitalize()} PRs in {repo_full_name}:**\n"]
            for pr in prs:
                lines.append(
                    f"#{pr.number} — {pr.title}\n"
                    f"   `{pr.head.ref}` → `{pr.base.ref}` by @{pr.user.login}\n"
                    f"   {pr.html_url}"
                )
            return "\n\n".join(lines)
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def github_create_pull_request(
        repo_full_name: str,
        title: str,
        head_branch: str,
        base_branch: str = "",
        body: str = "",
        draft: bool = False,
    ) -> str:
        """
        Create a pull request in a GitHub repository.
        head_branch: the branch with your changes
        base_branch: the branch to merge into (default branch if empty)
        """
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            base = base_branch or repo.default_branch
            pr = repo.create_pull(
                title=title,
                body=body,
                head=head_branch,
                base=base,
                draft=draft,
            )
            return (
                f"✅ Pull request created!\n"
                f"#{pr.number} — {pr.title}\n"
                f"`{pr.head.ref}` → `{pr.base.ref}`\n"
                f"🔗 {pr.html_url}"
            )
        except Exception as e:
            return f"Error creating PR: {str(e)}"

    @tool
    def github_merge_pull_request(
        repo_full_name: str,
        pr_number: int,
        merge_method: str = "merge",
        commit_message: str = "",
    ) -> str:
        """
        Merge a pull request.
        merge_method: 'merge', 'squash', or 'rebase'
        """
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            pr = repo.get_pull(pr_number)
            if not pr.mergeable:
                return f"PR #{pr_number} cannot be merged right now (conflicts or CI failing)."
            kwargs = {"merge_method": merge_method}
            if commit_message:
                kwargs["commit_message"] = commit_message
            result = pr.merge(**kwargs)
            return f"✅ PR #{pr_number} merged! {result.message}"
        except Exception as e:
            return f"Error merging PR: {str(e)}"

    # ─────────────────────────────────────────────────
    # Commits & History
    # ─────────────────────────────────────────────────

    @tool
    def github_get_commits(repo_full_name: str, branch: str = "", limit: int = 10) -> str:
        """Get recent commit history for a GitHub repository."""
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            ref = branch or repo.default_branch
            commits = list(repo.get_commits(sha=ref))[:limit]
            if not commits:
                return "No commits found."
            lines = [f"📜 Recent commits on `{ref}` in **{repo_full_name}**:\n"]
            for c in commits:
                sha_short = c.sha[:7]
                msg = c.commit.message.splitlines()[0][:72]
                author = c.commit.author.name
                date = c.commit.author.date.strftime("%Y-%m-%d")
                lines.append(f"`{sha_short}` {msg}\n         👤 {author} · {date}")
            return "\n\n".join(lines)
        except Exception as e:
            return f"Error fetching commits: {str(e)}"

    # ─────────────────────────────────────────────────
    # Gists
    # ─────────────────────────────────────────────────

    @tool
    def github_create_gist(
        filename: str,
        content: str,
        description: str = "",
        public: bool = True,
    ) -> str:
        """
        Create a GitHub Gist (quick code share).
        filename: e.g. 'script.py', 'example.js'
        content: the code or text content
        public: False for secret gist
        """
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            from github import InputFileContent
            gist = gh.get_user().create_gist(
                public=public,
                files={filename: InputFileContent(content)},
                description=description,
            )
            return (
                f"✅ Gist created!\n"
                f"📄 {filename}\n"
                f"🔗 {gist.html_url}"
            )
        except Exception as e:
            return f"Error creating gist: {str(e)}"

    @tool
    def github_list_gists(limit: int = 10) -> str:
        """List your GitHub Gists."""
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            gists = list(gh.get_user().get_gists())[:limit]
            if not gists:
                return "You have no gists yet."
            lines = [f"📎 Your Gists ({len(gists)} shown):\n"]
            for g in gists:
                filenames = ", ".join(g.files.keys())
                vis = "🌍 public" if g.public else "🔒 secret"
                lines.append(f"{vis} — {filenames}\n  {g.description or 'no description'}\n  {g.html_url}")
            return "\n\n".join(lines)
        except Exception as e:
            return f"Error: {str(e)}"

    # ─────────────────────────────────────────────────
    # User / Org
    # ─────────────────────────────────────────────────

    @tool
    def github_get_user_profile(username: str = "") -> str:
        """
        Get a GitHub user profile.
        Leave username empty to get your own profile.
        """
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            user = gh.get_user(username) if username else gh.get_user()
            return (
                f"👤 **{user.name or user.login}** (@{user.login})\n"
                f"📝 {user.bio or 'No bio'}\n"
                f"📍 {user.location or 'Unknown location'}\n"
                f"🏢 {user.company or 'No company'}\n"
                f"📦 Public repos: {user.public_repos}\n"
                f"👥 Followers: {user.followers} | Following: {user.following}\n"
                f"📅 Joined: {user.created_at.strftime('%Y-%m-%d')}\n"
                f"🔗 {user.html_url}"
            )
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def github_add_collaborator(repo_full_name: str, username: str, permission: str = "push") -> str:
        """
        Add a collaborator to a GitHub repository.
        permission: 'pull', 'push', 'admin', 'maintain', 'triage'
        """
        gh = get_gh_client()
        if not gh:
            return no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            repo.add_to_collaborators(username, permission=permission)
            return f"✅ Added @{username} as collaborator to **{repo_full_name}** with `{permission}` permission."
        except Exception as e:
            return f"Error: {str(e)}"

    return [
        github_list_my_repos,
        github_create_repo,
        github_delete_repo,
        github_get_repo_info,
        github_fork_repo,
        github_search_repos,
        github_update_repo,
        github_list_files,
        github_read_file,
        github_create_or_update_file,
        github_delete_file,
        github_rename_file,
        github_list_branches,
        github_create_branch,
        github_delete_branch,
        github_list_issues,
        github_create_issue,
        github_close_issue,
        github_comment_on_issue,
        github_list_pull_requests,
        github_create_pull_request,
        github_merge_pull_request,
        github_get_commits,
        github_create_gist,
        github_list_gists,
        github_get_user_profile,
        github_add_collaborator,
    ]
