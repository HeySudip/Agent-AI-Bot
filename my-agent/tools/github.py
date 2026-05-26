"""GitHub tools for repository management via PyGithub.

Provides LangChain tools for: repos, files, branches, issues, PRs, commits,
gists, and user profiles. All operations require a GitHub token stored in
the bot's config (auto-detected from chat when user pastes a ghp_... token).
"""

from __future__ import annotations

import logging
from typing import Optional

from github import Github, InputFileContent
from github.GithubException import GithubException
from langchain.tools import tool

from config import load_config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_gh_client() -> Optional[Github]:
    """Return an authenticated PyGithub client, or None if no token is set."""
    token = load_config().get("github_token", "")
    return Github(token) if token else None


def _no_token() -> str:
    return (
        "GitHub token not connected. Just paste your token (ghp_...) "
        "in the chat and I'll connect right away."
    )


def _fmt_repo(repo) -> str:
    """Format a single repository for display."""
    desc = repo.description or "no description"
    vis = "\U0001f512" if repo.private else "\U0001f30d"
    return f"{vis} [{repo.full_name}]({repo.html_url}) \u2b50{repo.stargazers_count} \u2014 {desc}"


def _gh_error(action: str, exc: Exception) -> str:
    """Produce a user-friendly error message and log the exception."""
    logger.error("GitHub %s failed: %s", action, exc)
    if isinstance(exc, GithubException):
        msg = exc.data.get("message", str(exc)) if isinstance(exc.data, dict) else str(exc)
        return f"GitHub error ({action}): {msg}"
    return f"Error ({action}): {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool builder
# ─────────────────────────────────────────────────────────────────────────────


def build_github_tools() -> list:
    """Build and return all GitHub LangChain tools."""

    # ─── Repositories ────────────────────────────────────────────────────────

    @tool
    def github_list_my_repos(sort_by: str = "updated") -> str:
        """List your GitHub repositories.

        Args:
            sort_by: Sort order — 'updated', 'created', 'pushed', or 'full_name'.
        """
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            repos = list(gh.get_user().get_repos(sort=sort_by, direction="desc"))
            if not repos:
                return "You don't have any repositories yet."
            lines = [f"Found {len(repos)} repositories:\n"]
            for r in repos[:25]:
                lines.append(_fmt_repo(r))
            if len(repos) > 25:
                lines.append(f"\n\u2026and {len(repos) - 25} more.")
            return "\n".join(lines)
        except Exception as e:
            return _gh_error("list repos", e)

    @tool
    def github_create_repo(
        name: str,
        description: str = "",
        private: bool = False,
        has_readme: bool = True,
        gitignore_template: str = "",
        license_template: str = "",
    ) -> str:
        """Create a new GitHub repository.

        Args:
            name: Repository name (no spaces, use hyphens).
            description: Short description.
            private: True for a private repository.
            has_readme: Auto-initialize with a README.
            gitignore_template: e.g. 'Python', 'Node', 'Go'.
            license_template: e.g. 'mit', 'apache-2.0', 'gpl-3.0'.
        """
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            kwargs: dict = {
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
            visibility = "\U0001f512 private" if private else "\U0001f30d public"
            return (
                f"\u2705 Repo created ({visibility})\n"
                f"\U0001f4cc **{repo.full_name}**\n"
                f"\U0001f517 {repo.html_url}\n"
                f"\U0001f4cb Clone: `git clone {repo.clone_url}`"
            )
        except Exception as e:
            return _gh_error("create repo", e)

    @tool
    def github_delete_repo(repo_full_name: str, confirmed: bool = False) -> str:
        """Delete a GitHub repository permanently.

        Args:
            repo_full_name: 'owner/repo-name'.
            confirmed: Must be True to actually delete (safety gate).
        """
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        if not confirmed:
            return (
                f"\u26a0\ufe0f Are you sure you want to delete **{repo_full_name}**? "
                f"This is permanent and cannot be undone. "
                f"Say 'yes delete {repo_full_name}' to confirm."
            )
        try:
            gh.get_repo(repo_full_name).delete()
            return f"\u2705 Repository **{repo_full_name}** has been permanently deleted."
        except Exception as e:
            return _gh_error("delete repo", e)

    @tool
    def github_get_repo_info(repo_full_name: str) -> str:
        """Get detailed information about a GitHub repository."""
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            r = gh.get_repo(repo_full_name)
            topics = ", ".join(r.get_topics()) or "none"
            return (
                f"**{r.full_name}**\n"
                f"\U0001f4dd {r.description or 'No description'}\n"
                f"\U0001f30d Visibility: {'Private' if r.private else 'Public'}\n"
                f"\u2b50 Stars: {r.stargazers_count} | \U0001f374 Forks: {r.forks_count} | \U0001f441 Watchers: {r.watchers_count}\n"
                f"\U0001f41b Open issues: {r.open_issues_count}\n"
                f"\U0001f4bb Language: {r.language or 'Unknown'}\n"
                f"\U0001f33f Default branch: {r.default_branch}\n"
                f"\U0001f3f7 Topics: {topics}\n"
                f"\U0001f4c5 Created: {r.created_at.strftime('%Y-%m-%d')}\n"
                f"\U0001f504 Last updated: {r.updated_at.strftime('%Y-%m-%d %H:%M')}\n"
                f"\U0001f517 {r.html_url}\n"
                f"\U0001f4cb Clone: `{r.clone_url}`"
            )
        except Exception as e:
            return _gh_error("get repo info", e)

    @tool
    def github_fork_repo(repo_full_name: str) -> str:
        """Fork a public GitHub repository to your account."""
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            fork = gh.get_user().create_fork(repo)
            return (
                f"\u2705 Forked **{repo_full_name}** to your account!\n"
                f"\U0001f517 Your fork: {fork.html_url}\n"
                f"\U0001f4cb Clone: `git clone {fork.clone_url}`"
            )
        except Exception as e:
            return _gh_error("fork repo", e)

    @tool
    def github_search_repos(
        query: str, language: str = "", sort: str = "stars", limit: int = 6
    ) -> str:
        """Search GitHub for public repositories.

        Args:
            query: Search keywords.
            language: Filter by language, e.g. 'python', 'javascript'.
            sort: Sort by 'stars', 'forks', or 'updated'.
            limit: Max results (capped at 8).
        """
        try:
            gh = _get_gh_client() or Github()
            q = f"{query} language:{language}" if language else query
            repos = gh.search_repositories(query=q, sort=sort)
            count = min(limit, 8)
            results = []
            for repo in list(repos[:count]):
                results.append(
                    f"\u2b50 {repo.stargazers_count:,} | **[{repo.full_name}]({repo.html_url})**\n"
                    f"   {repo.description or 'No description'}\n"
                    f"   \U0001f4bb {repo.language or 'Unknown'} | \U0001f374 {repo.forks_count} forks"
                )
            return "\n\n".join(results) if results else "No repositories found."
        except Exception as e:
            return _gh_error("search repos", e)

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
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            kwargs: dict = {}
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
            return f"\u2705 Updated settings for **{repo_full_name}**"
        except Exception as e:
            return _gh_error("update repo", e)


    # ─── Files ───────────────────────────────────────────────────────────────

    @tool
    def github_list_files(repo_full_name: str, path: str = "", branch: str = "") -> str:
        """List files and folders in a GitHub repository directory.

        Args:
            repo_full_name: 'owner/repo'.
            path: Subdirectory path (empty for root).
            branch: Branch name (empty for default branch).
        """
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            ref = branch or repo.default_branch
            contents = repo.get_contents(path, ref=ref)
            if not isinstance(contents, list):
                contents = [contents]
            dirs = sorted([c for c in contents if c.type == "dir"], key=lambda x: x.name)
            files = sorted([c for c in contents if c.type == "file"], key=lambda x: x.name)
            lines = [f"\U0001f4c1 `{repo_full_name}/{path or ''}` (branch: {ref})\n"]
            for d in dirs:
                lines.append(f"\U0001f4c1 {d.name}/")
            for f in files:
                size = f"{f.size:,} bytes" if f.size < 1024 else f"{f.size // 1024} KB"
                lines.append(f"\U0001f4c4 {f.name} ({size})")
            return "\n".join(lines)
        except Exception as e:
            return _gh_error("list files", e)

    @tool
    def github_read_file(repo_full_name: str, file_path: str, branch: str = "") -> str:
        """Read the contents of a file from a GitHub repository.

        Args:
            repo_full_name: 'owner/repo'.
            file_path: Path to the file within the repo.
            branch: Branch name (empty for default branch).
        """
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            ref = branch or repo.default_branch
            f = repo.get_contents(file_path, ref=ref)
            if isinstance(f, list):
                return f"'{file_path}' is a directory. Use github_list_files instead."
            content = f.decoded_content.decode("utf-8", errors="replace")
            size_kb = f.size / 1024
            truncated = (
                f"\n\n_(file truncated \u2014 {len(content)} total chars)_"
                if len(content) > 6000
                else ""
            )
            return (
                f"\U0001f4c4 **{file_path}** ({size_kb:.1f} KB, branch: {ref})\n\n"
                f"```\n{content[:6000]}\n```{truncated}"
            )
        except Exception as e:
            return _gh_error("read file", e)

    @tool
    def github_create_or_update_file(
        repo_full_name: str,
        file_path: str,
        content: str,
        commit_message: str = "",
        branch: str = "",
    ) -> str:
        """Create or update a file in a GitHub repository.

        Args:
            repo_full_name: 'owner/repo'.
            file_path: Path like 'src/main.py' or 'README.md'.
            content: Full file content (text).
            commit_message: Git commit message.
            branch: Branch name (empty for default branch).
        """
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            ref = branch or repo.default_branch
            msg = commit_message or f"Update {file_path}"
            try:
                existing = repo.get_contents(file_path, ref=ref)
                repo.update_file(
                    path=file_path, message=msg, content=content,
                    sha=existing.sha, branch=ref,
                )
                action = "Updated"
            except GithubException:
                repo.create_file(path=file_path, message=msg, content=content, branch=ref)
                action = "Created"
            return (
                f"\u2705 {action} `{file_path}` in **{repo_full_name}**\n"
                f"\U0001f33f Branch: {ref}\n"
                f"\U0001f4ac Commit: {msg}\n"
                f"\U0001f517 {repo.html_url}/blob/{ref}/{file_path}"
            )
        except Exception as e:
            return _gh_error("create/update file", e)

    @tool
    def github_delete_file(
        repo_full_name: str, file_path: str, commit_message: str = "", branch: str = ""
    ) -> str:
        """Delete a file from a GitHub repository.

        Args:
            repo_full_name: 'owner/repo'.
            file_path: Path to the file to delete.
            commit_message: Git commit message.
            branch: Branch name (empty for default branch).
        """
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            ref = branch or repo.default_branch
            f = repo.get_contents(file_path, ref=ref)
            msg = commit_message or f"Delete {file_path}"
            repo.delete_file(file_path, msg, f.sha, branch=ref)
            return f"\u2705 Deleted `{file_path}` from **{repo_full_name}** (branch: {ref})"
        except Exception as e:
            return _gh_error("delete file", e)

    @tool
    def github_rename_file(
        repo_full_name: str, old_path: str, new_path: str, branch: str = ""
    ) -> str:
        """Rename or move a file in a GitHub repository (copy + delete).

        Args:
            repo_full_name: 'owner/repo'.
            old_path: Current file path.
            new_path: Desired new file path.
            branch: Branch name (empty for default branch).
        """
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            ref = branch or repo.default_branch
            old_file = repo.get_contents(old_path, ref=ref)
            content = old_file.decoded_content.decode("utf-8", errors="replace")
            # Create at new path
            try:
                existing_new = repo.get_contents(new_path, ref=ref)
                repo.update_file(
                    new_path, f"Move {old_path} to {new_path}",
                    content, existing_new.sha, branch=ref,
                )
            except GithubException:
                repo.create_file(new_path, f"Move {old_path} to {new_path}", content, branch=ref)
            # Delete old path
            repo.delete_file(
                old_path, f"Remove {old_path} (moved to {new_path})",
                old_file.sha, branch=ref,
            )
            return f"\u2705 Renamed `{old_path}` \u2192 `{new_path}` in **{repo_full_name}**"
        except Exception as e:
            return _gh_error("rename file", e)


    # ─── Branches ────────────────────────────────────────────────────────────

    @tool
    def github_list_branches(repo_full_name: str) -> str:
        """List all branches in a GitHub repository."""
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            branches = list(repo.get_branches())
            default = repo.default_branch
            lines = [f"\U0001f33f Branches in **{repo_full_name}** ({len(branches)} total):\n"]
            for b in branches:
                marker = " \u2190 default" if b.name == default else ""
                lines.append(f"  \u2022 `{b.name}`{marker}")
            return "\n".join(lines)
        except Exception as e:
            return _gh_error("list branches", e)

    @tool
    def github_create_branch(
        repo_full_name: str, branch_name: str, from_branch: str = ""
    ) -> str:
        """Create a new branch in a GitHub repository.

        Args:
            repo_full_name: 'owner/repo'.
            branch_name: Name for the new branch.
            from_branch: Source branch (empty for default branch).
        """
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            source = from_branch or repo.default_branch
            sha = repo.get_branch(source).commit.sha
            repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=sha)
            return f"\u2705 Branch `{branch_name}` created from `{source}` in **{repo_full_name}**"
        except Exception as e:
            return _gh_error("create branch", e)

    @tool
    def github_delete_branch(repo_full_name: str, branch_name: str) -> str:
        """Delete a branch from a GitHub repository."""
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            repo.get_git_ref(f"heads/{branch_name}").delete()
            return f"\u2705 Deleted branch `{branch_name}` from **{repo_full_name}**"
        except Exception as e:
            return _gh_error("delete branch", e)

    # ─── Issues ──────────────────────────────────────────────────────────────

    @tool
    def github_list_issues(repo_full_name: str, state: str = "open", limit: int = 10) -> str:
        """List issues in a GitHub repository.

        Args:
            state: 'open', 'closed', or 'all'.
            limit: Maximum number of issues to return.
        """
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            issues = list(repo.get_issues(state=state))[:limit]
            if not issues:
                return f"No {state} issues found in **{repo_full_name}**."
            lines = [f"\U0001f41b **{state.capitalize()} issues in {repo_full_name}:**\n"]
            for i in issues:
                labels = ", ".join(l.name for l in i.labels)
                label_str = f" [{labels}]" if labels else ""
                lines.append(
                    f"#{i.number} \u2014 {i.title}{label_str}\n"
                    f"   Opened by @{i.user.login} \u00b7 {i.created_at.strftime('%Y-%m-%d')}\n"
                    f"   {i.html_url}"
                )
            return "\n\n".join(lines)
        except Exception as e:
            return _gh_error("list issues", e)

    @tool
    def github_create_issue(
        repo_full_name: str,
        title: str,
        body: str = "",
        labels: str = "",
        assignee: str = "",
    ) -> str:
        """Create an issue on a GitHub repository.

        Args:
            repo_full_name: 'owner/repo'.
            title: Issue title.
            body: Issue body (markdown).
            labels: Comma-separated label names, e.g. 'bug,help wanted'.
            assignee: GitHub username to assign.
        """
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            kwargs: dict = {"title": title, "body": body}
            if labels:
                kwargs["labels"] = [l.strip() for l in labels.split(",")]
            if assignee:
                kwargs["assignee"] = assignee
            issue = repo.create_issue(**kwargs)
            return (
                f"\u2705 Issue created in **{repo_full_name}**\n"
                f"#{issue.number} \u2014 {issue.title}\n"
                f"\U0001f517 {issue.html_url}"
            )
        except Exception as e:
            return _gh_error("create issue", e)

    @tool
    def github_close_issue(repo_full_name: str, issue_number: int, comment: str = "") -> str:
        """Close an issue, optionally adding a closing comment."""
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            issue = repo.get_issue(issue_number)
            if comment:
                issue.create_comment(comment)
            issue.edit(state="closed")
            return f"\u2705 Closed issue #{issue_number} in **{repo_full_name}**"
        except Exception as e:
            return _gh_error("close issue", e)

    @tool
    def github_comment_on_issue(repo_full_name: str, issue_number: int, comment: str) -> str:
        """Add a comment to an issue or pull request."""
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            c = repo.get_issue(issue_number).create_comment(comment)
            return f"\u2705 Comment added to #{issue_number}\n\U0001f517 {c.html_url}"
        except Exception as e:
            return _gh_error("comment on issue", e)


    # ─── Pull Requests ───────────────────────────────────────────────────────

    @tool
    def github_list_pull_requests(repo_full_name: str, state: str = "open") -> str:
        """List pull requests in a GitHub repository.

        Args:
            state: 'open', 'closed', or 'all'.
        """
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            prs = list(repo.get_pulls(state=state))[:10]
            if not prs:
                return f"No {state} pull requests in **{repo_full_name}**."
            lines = [f"\U0001f500 **{state.capitalize()} PRs in {repo_full_name}:**\n"]
            for pr in prs:
                lines.append(
                    f"#{pr.number} \u2014 {pr.title}\n"
                    f"   `{pr.head.ref}` \u2192 `{pr.base.ref}` by @{pr.user.login}\n"
                    f"   {pr.html_url}"
                )
            return "\n\n".join(lines)
        except Exception as e:
            return _gh_error("list PRs", e)

    @tool
    def github_create_pull_request(
        repo_full_name: str,
        title: str,
        head_branch: str,
        base_branch: str = "",
        body: str = "",
        draft: bool = False,
    ) -> str:
        """Create a pull request.

        Args:
            repo_full_name: 'owner/repo'.
            title: PR title.
            head_branch: Branch with your changes.
            base_branch: Branch to merge into (empty for default).
            body: PR description (markdown).
            draft: True to create as draft PR.
        """
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            base = base_branch or repo.default_branch
            pr = repo.create_pull(title=title, body=body, head=head_branch, base=base, draft=draft)
            return (
                f"\u2705 Pull request created!\n"
                f"#{pr.number} \u2014 {pr.title}\n"
                f"`{pr.head.ref}` \u2192 `{pr.base.ref}`\n"
                f"\U0001f517 {pr.html_url}"
            )
        except Exception as e:
            return _gh_error("create PR", e)

    @tool
    def github_merge_pull_request(
        repo_full_name: str,
        pr_number: int,
        merge_method: str = "merge",
        commit_message: str = "",
    ) -> str:
        """Merge a pull request.

        Args:
            pr_number: PR number.
            merge_method: 'merge', 'squash', or 'rebase'.
            commit_message: Custom merge commit message.
        """
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            pr = repo.get_pull(pr_number)
            if not pr.mergeable:
                return f"PR #{pr_number} cannot be merged right now (conflicts or CI failing)."
            kwargs: dict = {"merge_method": merge_method}
            if commit_message:
                kwargs["commit_message"] = commit_message
            result = pr.merge(**kwargs)
            return f"\u2705 PR #{pr_number} merged! {result.message}"
        except Exception as e:
            return _gh_error("merge PR", e)

    # ─── Commits & History ───────────────────────────────────────────────────

    @tool
    def github_get_commits(repo_full_name: str, branch: str = "", limit: int = 10) -> str:
        """Get recent commit history for a repository.

        Args:
            branch: Branch name (empty for default).
            limit: Number of commits to return.
        """
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            ref = branch or repo.default_branch
            commits = list(repo.get_commits(sha=ref))[:limit]
            if not commits:
                return "No commits found."
            lines = [f"\U0001f4dc Recent commits on `{ref}` in **{repo_full_name}**:\n"]
            for c in commits:
                sha_short = c.sha[:7]
                msg = c.commit.message.splitlines()[0][:72]
                author = c.commit.author.name
                date = c.commit.author.date.strftime("%Y-%m-%d")
                lines.append(f"`{sha_short}` {msg}\n         \U0001f464 {author} \u00b7 {date}")
            return "\n\n".join(lines)
        except Exception as e:
            return _gh_error("get commits", e)

    # ─── Gists ───────────────────────────────────────────────────────────────

    @tool
    def github_create_gist(
        filename: str, content: str, description: str = "", public: bool = True
    ) -> str:
        """Create a GitHub Gist (quick code share).

        Args:
            filename: e.g. 'script.py', 'example.js'.
            content: The code or text content.
            description: Gist description.
            public: False for a secret gist.
        """
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            gist = gh.get_user().create_gist(
                public=public,
                files={filename: InputFileContent(content)},
                description=description,
            )
            return (
                f"\u2705 Gist created!\n"
                f"\U0001f4c4 {filename}\n"
                f"\U0001f517 {gist.html_url}"
            )
        except Exception as e:
            return _gh_error("create gist", e)

    @tool
    def github_list_gists(limit: int = 10) -> str:
        """List your GitHub Gists."""
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            gists = list(gh.get_user().get_gists())[:limit]
            if not gists:
                return "You have no gists yet."
            lines = [f"\U0001f4ce Your Gists ({len(gists)} shown):\n"]
            for g in gists:
                filenames = ", ".join(g.files.keys())
                vis = "\U0001f30d public" if g.public else "\U0001f512 secret"
                lines.append(
                    f"{vis} \u2014 {filenames}\n"
                    f"  {g.description or 'no description'}\n"
                    f"  {g.html_url}"
                )
            return "\n\n".join(lines)
        except Exception as e:
            return _gh_error("list gists", e)

    # ─── User / Collaborators ────────────────────────────────────────────────

    @tool
    def github_get_user_profile(username: str = "") -> str:
        """Get a GitHub user profile. Leave username empty for your own profile."""
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            user = gh.get_user(username) if username else gh.get_user()
            return (
                f"\U0001f464 **{user.name or user.login}** (@{user.login})\n"
                f"\U0001f4dd {user.bio or 'No bio'}\n"
                f"\U0001f4cd {user.location or 'Unknown location'}\n"
                f"\U0001f3e2 {user.company or 'No company'}\n"
                f"\U0001f4e6 Public repos: {user.public_repos}\n"
                f"\U0001f465 Followers: {user.followers} | Following: {user.following}\n"
                f"\U0001f4c5 Joined: {user.created_at.strftime('%Y-%m-%d')}\n"
                f"\U0001f517 {user.html_url}"
            )
        except Exception as e:
            return _gh_error("get user profile", e)

    @tool
    def github_add_collaborator(
        repo_full_name: str, username: str, permission: str = "push"
    ) -> str:
        """Add a collaborator to a GitHub repository.

        Args:
            repo_full_name: 'owner/repo'.
            username: GitHub username to add.
            permission: 'pull', 'push', 'admin', 'maintain', or 'triage'.
        """
        gh = _get_gh_client()
        if not gh:
            return _no_token()
        try:
            repo = gh.get_repo(repo_full_name)
            repo.add_to_collaborators(username, permission=permission)
            return (
                f"\u2705 Added @{username} as collaborator to **{repo_full_name}** "
                f"with `{permission}` permission."
            )
        except Exception as e:
            return _gh_error("add collaborator", e)

    # ─── Return all tools ────────────────────────────────────────────────────

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
