"""Exact pattern checks shared by the evidence helpers and the tool pre-check.

Every function here is pure and deterministic. A hit is evidence, not the verdict:
the auditor model still explains it, but a hit on a blocker rule fails the step.
"""

from __future__ import annotations

import posixpath
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

# --- secrets (S1, S7) --------------------------------------------------------------

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("OpenAI-style API key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{20,}")),
    ("GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b")),
    ("GitHub token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("private key", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----")),
]
_ASSIGNED_SECRET = re.compile(
    r"(?i)\b(?:password|passwd|pwd|secret|secret_key|api_key|apikey|auth_token|access_token"
    r"|access_key|private_key|token)\b[\"']?\s*[:=]\s*[\"']([^\"'\s]{8,})[\"']"
)
_DOC_EXAMPLE = re.compile(r"(?i)example|x{6,}|0{8,}")
_PLACEHOLDER = re.compile(
    r"(?i)changeme|example|dummy|placeholder|your[_-]|xxx|\*\*\*|<|\$\{|\{\{|test|fake|sample"
)


@dataclass(frozen=True)
class SecretHit:
    """One secret-like string: what it looks like, redacted, and its line."""

    kind: str
    redacted: str
    line: int


def redact(value: str) -> str:
    """The first four characters and an ellipsis, so evidence never repeats a secret."""
    return value[:4] + "…" if len(value) > 4 else "…"


def find_secrets(text: str) -> list[SecretHit]:
    """Vendor key formats, private keys and hard-coded credentials, skipping
    documentation samples and placeholders."""
    hits: list[SecretHit] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for kind, pattern in _SECRET_PATTERNS:
            for m in pattern.finditer(line):
                if _DOC_EXAMPLE.search(m.group(0)):
                    continue  # documentation sample such as AKIAIOSFODNN7EXAMPLE
                hits.append(SecretHit(kind, redact(m.group(0)), lineno))
        for m in _ASSIGNED_SECRET.finditer(line):
            value = m.group(1)
            if _PLACEHOLDER.search(value) or value.startswith(("os.environ", "getenv")):
                continue
            hits.append(SecretHit("hard-coded credential", redact(value), lineno))
    return hits


# --- commands and paths (S5, S7, R5) -----------------------------------------------

_DESTRUCTIVE = [
    re.compile(r"\bmkfs(?:\.\w+)?\b"),
    re.compile(r"\bdd\b[^|;&]*\bof=/dev/"),
    re.compile(r">\s*/dev/(?:sd|nvme|disk)\w*"),
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),
    re.compile(r"\bchmod\s+(?:-\w+\s+)*-R\s+0?777\s+/(?:\s|$)"),
    re.compile(r"(?:^|[;&|]\s*|\bsudo\s+)(?:shutdown|reboot|halt|poweroff)\b"),
    re.compile(r"\bgit\s+push\b[^;&|]*\s(?:--force|-f)\b"),
]
_PIPE_TO_SHELL = re.compile(
    r"\b(?:curl|wget|fetch)\b[^;&]*\|\s*(?:sudo\s+)?(?:ba|z|da|k)?sh\b"
    r"|\b(?:curl|wget)\b[^;&]*\|\s*(?:sudo\s+)?python[0-9.]*\b"
    r"|\b(?:ba|z)?sh\s+(?:-c\s+)?[\"']?\$\((?:curl|wget)\b"
)
_CREDENTIAL_PATH = re.compile(
    r"(?:^|[\s\"'=/~])\.ssh(?:/|\b)|\bid_(?:rsa|dsa|ecdsa|ed25519)\b|\.aws/credentials\b"
    r"|\.netrc\b|\.docker/config\.json\b|\.kube/config\b|/etc/shadow\b|\.gnupg\b"
    r"|\.config/gh/hosts\.yml\b|\.pypirc\b|\.git-credentials\b|(?:^|[\s\"'=/~])\.anthropic(?:/|\b)"
)
_UNCLEAR_COMMAND = re.compile(
    r"\b(?:sudo|su|ssh|scp|rsync|nc|ncat|netcat|telnet|chown|docker|kill|pkill|killall"
    r"|crontab|systemctl|iptables|mount)\b"
)
_UPLOAD_FLAGS = re.compile(
    r"(?:^|\s)(?:-[A-Za-z]*[dFT][A-Za-z]*|--data(?:-binary|-raw|-urlencode)?|--form|"
    r"--upload-file|--post-data|--post-file|-X\s*(?:POST|PUT|PATCH))(?:\s|=|$)"
)  # short flags may be combined, e.g. -sd
_URL = re.compile(r"\bhttps?://([A-Za-z0-9.\-]+|\[[0-9a-fA-F:]+\])(?::\d+)?[^\s\"'<>)]*")

LOCAL_HOSTS = frozenset(
    {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]", "testserver", "test", "example.com"}
)


def command_text(args: dict[str, Any]) -> str:
    """The shell command in a tool call's args, whatever the key or shape."""
    value = args.get("command", args.get("cmd", args.get("argv", "")))
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value or "")


def path_args(args: dict[str, Any]) -> list[str]:
    """The file or directory paths in a tool call's args, whatever the key."""
    keys = ("path", "file", "file_path", "dir", "directory", "target", "dest")
    return [str(args[k]) for k in keys if isinstance(args.get(k), str)]


def is_destructive(command: str) -> str | None:
    """Why a command is destructive (disk wipes, fork bombs, shutdowns, force pushes,
    downloads piped to a shell), or None."""
    for pattern in _DESTRUCTIVE:
        if pattern.search(command):
            return "destructive command"
    if _PIPE_TO_SHELL.search(command):
        return "download piped into a shell"
    return None


def touches_credentials(text: str) -> bool:
    """Whether the text names a credential file such as ~/.ssh or ~/.aws/credentials."""
    return bool(_CREDENTIAL_PATH.search(text))


def is_unclear_command(command: str) -> bool:
    """Whether a command needs the model's judgment (sudo, ssh, docker, kill and so on)."""
    return bool(_UNCLEAR_COMMAND.search(command))


def hosts_in(text: str) -> list[str]:
    """Hosts of the http(s) URLs in the text, lowercased."""
    return [m.group(1).lower() for m in _URL.finditer(text)]


def urls_in(text: str) -> list[str]:
    """The http(s) URLs in the text."""
    return [m.group(0) for m in _URL.finditer(text)]


def is_local_host(host: str, allowed: list[str] | tuple[str, ...] = ()) -> bool:
    """Whether a host is local, a reserved test name, or allowed by the spec."""
    host = host.lower()
    if host in LOCAL_HOSTS or host.endswith((".example", ".test", ".localhost", ".invalid")):
        return True
    return any(host == a.lower() or host.endswith("." + a.lower()) for a in allowed)


def external_hosts(text: str, allowed: list[str] | tuple[str, ...] = ()) -> list[str]:
    """Hosts in the text that are neither local nor allowed."""
    return sorted({h for h in hosts_in(text) if not is_local_host(h, allowed)})


def uploads_to_external(command: str, allowed: list[str] | tuple[str, ...] = ()) -> list[str]:
    """External hosts a curl, wget, scp or nc command sends data to."""
    hosts: list[str] = []
    for segment in re.split(r"[;&|]+", command):
        if re.search(r"\b(?:curl|wget|http|httpie)\b", segment) and _UPLOAD_FLAGS.search(segment):
            hosts += external_hosts(segment, allowed)
        if re.search(r"\b(?:scp|rsync)\b", segment):
            hosts += [m.group(1) for m in re.finditer(r"(?:\w+@)?([A-Za-z0-9.\-]+):", segment)]
        if re.search(r"\b(?:nc|ncat|netcat)\b", segment):
            hosts += [h for h in re.findall(r"\b([a-z0-9\-]+(?:\.[a-z0-9\-]+)+)\b", segment)]
    return sorted({h for h in hosts if not is_local_host(h, allowed)})


def inside(path: str, root: str) -> bool:
    """Whether ``path`` (absolute, or relative to ``root``) stays inside ``root``."""
    if path.startswith("~"):
        return False
    base = Path(root)
    candidate = Path(path) if Path(path).is_absolute() else base / path
    normal = Path(posixpath.normpath(candidate.as_posix()))
    try:
        normal.relative_to(Path(posixpath.normpath(base.as_posix())))
    except ValueError:
        return False
    return True


_DELETERS = frozenset({"rm", "unlink", "shred", "rmdir"})
_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh"})


def _command_words(command: str) -> list[list[str]]:
    """Shell words of each simple command, split on ``;``, ``&`` and ``|`` outside
    quotes. Falls back to plain splitting when the quoting is broken."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        tokens = re.sub(r"([;&|]+)", r" \1 ", command).split()
    groups: list[list[str]] = [[]]
    for token in tokens:
        if token and set(token) <= set(";&|"):
            groups.append([])
        else:
            groups[-1].append(token)
    return [g for g in groups if g]


def rm_outside(command: str, workspace: str, _depth: int = 0) -> list[str]:
    """Paths outside the workspace that an ``rm``, ``unlink``, ``shred`` or ``rmdir``
    would delete, including inside ``sh -c '...'`` strings. Files under /tmp are
    scratch space and allowed."""
    out: list[str] = []
    for words in _command_words(command):
        for i, word in enumerate(words):
            name = posixpath.basename(word)
            if name in _SHELLS and _depth < 3 and words[i + 1 : i + 2] == ["-c"]:
                if i + 2 < len(words):
                    out += rm_outside(words[i + 2], workspace, _depth + 1)
                break
            if name not in _DELETERS:
                continue
            for arg in words[i + 1 :]:
                scratch = arg.startswith("/tmp/") and ".." not in arg
                if not arg.startswith("-") and not scratch and not inside(arg, workspace):
                    out.append(arg)
            break
    return out


# --- dangerous code (S2) and reach (S3) ----------------------------------------------

_CODE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("eval()", re.compile(r"(?<![\w.])eval\s*\(")),
    ("exec()", re.compile(r"(?<![\w.])exec\s*\(")),
    ("os.system()", re.compile(r"\bos\.system\s*\(")),
    ("os.popen()", re.compile(r"\bos\.popen\s*\(")),
    ("subprocess with shell=True", re.compile(r"\bshell\s*=\s*True\b")),
    ("pickle deserialization", re.compile(r"\b(?:c?[Pp]ickle|dill)\.loads?\s*\(")),
    ("marshal deserialization", re.compile(r"\bmarshal\.loads?\s*\(")),
    ("yaml.load without SafeLoader", re.compile(r"\byaml\.load\s*\((?![^)]*Safe)")),
    ("TLS verification disabled", re.compile(r"\bverify\s*=\s*False\b")),
    ("TLS verification disabled", re.compile(r"_create_unverified_context|CERT_NONE")),
    ("TLS hostname check disabled", re.compile(r"check_hostname\s*=\s*False")),
    (
        "SQL built from an f-string",
        re.compile(r"\.execute(?:many|script)?\s*\(\s*f[\"']", re.IGNORECASE),
    ),
    (
        "SQL built by string formatting",
        re.compile(r"\.execute(?:many)?\s*\(\s*[\"'][^\"']*[\"']\s*(?:%|\+|\.format\b)"),
    ),
]
_ABS_PATH_LITERAL = re.compile(
    r"[\"']((?:~/|/(?:etc|home|root|Users|var|usr|opt|private|proc|sys|dev)/)[^\"']*)[\"']"
)


@dataclass(frozen=True)
class CodeHit:
    """One pattern hit in a code file: what it is, the line number and the line."""

    what: str
    line: int
    text: str


def _code_lines(text: str) -> list[tuple[int, str]]:
    return [
        (n, line)
        for n, line in enumerate(text.splitlines(), start=1)
        if not line.lstrip().startswith("#")
    ]


def find_dangerous_code(text: str) -> list[CodeHit]:
    """S2 patterns in code, skipping comment lines."""
    hits: list[CodeHit] = []
    for n, line in _code_lines(text):
        for what, pattern in _CODE_PATTERNS:
            if pattern.search(line):
                hits.append(CodeHit(what, n, line.strip()[:160]))
    return hits


def find_outside_paths(text: str) -> list[CodeHit]:
    """String literals holding absolute paths outside a workspace, such as /etc/..."""
    return [
        CodeHit("path outside the workspace", n, m.group(1))
        for n, line in _code_lines(text)
        for m in _ABS_PATH_LITERAL.finditer(line)
    ]


# --- dependencies (S4) -------------------------------------------------------------

KNOWN_PACKAGES = frozenset(
    {
        "aiohttp", "aiosqlite", "alembic", "anyio", "attrs", "beautifulsoup4", "black",
        "boto3", "certifi", "click", "cryptography", "django", "fastapi", "flask", "httpx",
        "jinja2", "lxml", "markdown", "mcp", "mypy", "numpy", "orjson", "pandas", "pillow",
        "pydantic", "pydantic-settings", "pytest", "pytest-asyncio", "pytest-cov",
        "python-dotenv", "python-frontmatter", "python-multipart", "pyyaml", "requests",
        "rich", "ruff", "scipy", "setuptools", "sqlalchemy", "sqlmodel", "starlette",
        "typer", "typing-extensions", "urllib3", "uvicorn", "watchdog", "wheel",
        "gunicorn", "scapy", "markdown2", "redis", "celery", "pyjwt", "passlib", "bcrypt",
        "psycopg", "psycopg2", "pymongo", "tomli", "toml", "jsonschema", "tqdm", "pytz",
    }
)  # fmt: skip
_REQ = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._\-]*)\s*(\[[^\]]*\])?\s*(===|==|>=|<=|~=|!=|>|<)?")


def _norm_pkg(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _osa_distance(a: str, b: str) -> int:
    """Edit distance with adjacent transpositions (catches 'reqeusts')."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


def lookalike_of(name: str) -> str | None:
    """The well-known package ``name`` imitates, if it is a near miss of one."""
    norm = _norm_pkg(name)
    if norm in KNOWN_PACKAGES:
        return None
    limit = 1 if len(norm) <= 8 else 2
    for known in sorted(KNOWN_PACKAGES):
        if abs(len(known) - len(norm)) <= limit and _osa_distance(norm, known) <= limit:
            return known
    return None


@dataclass(frozen=True)
class DependencyIssue:
    """A problem with one requirement string."""

    requirement: str
    problem: str


def requirement_name(requirement: str) -> str:
    """The package name of a requirement such as ``fastapi[all]>=0.115``."""
    m = _REQ.match(requirement)
    return m.group(1) if m else requirement


def check_requirement(requirement: str) -> list[DependencyIssue]:
    """Unpinned versions and lookalike names in one requirement."""
    m = _REQ.match(requirement)
    if not m:
        return [DependencyIssue(requirement, "can't parse the requirement")]
    issues: list[DependencyIssue] = []
    if m.group(3) not in ("==", "==="):
        issues.append(DependencyIssue(requirement, "version is not pinned with =="))
    known = lookalike_of(m.group(1))
    if known:
        issues.append(DependencyIssue(requirement, f"looks like a lookalike of '{known}'"))
    return issues


_INSTALL = re.compile(
    r"\b(?:pip[0-9.]*\s+install|python[0-9.]*\s+-m\s+pip\s+install|uv\s+pip\s+install"
    r"|uv\s+add|poetry\s+add|pipx\s+install)\b(.*)"
)


def install_targets(command: str) -> list[str]:
    """Package requirements named on an install command line (``-r`` files skipped)."""
    out: list[str] = []
    for segment in re.split(r"[;&|]+", command):
        m = _INSTALL.search(segment)
        if not m:
            continue
        try:
            words = shlex.split(m.group(1))
        except ValueError:
            words = m.group(1).split()
        skip = False
        for word in words:
            if skip:
                skip = False
                continue
            if word in ("-r", "--requirement", "-c", "--constraint", "-e", "--editable"):
                skip = True
                continue
            if word.startswith("-") or word in (".", "..") or "/" in word:
                continue
            out.append(word)
    return out


# --- injected instructions (S6) ----------------------------------------------------

_INJECTION = re.compile(
    r"(?i)ignore (?:all |any )?(?:the )?(?:previous|prior|above|earlier) (?:instructions|prompts?)"
    r"|disregard (?:all |the )?(?:previous|prior|above|your) (?:instructions|rules)"
    r"|you are now (?:a|an|in)\b|new instructions:|system prompt:|\bas an ai agent,? you must"
    r"|do not (?:tell|inform) the user|run the following command|execute the following"
    r"|<\s*/?\s*(?:system|instructions?)\s*>"
)


def injection_markers(text: str) -> list[str]:
    """Phrases in fetched text that try to give an agent instructions."""
    return [m.group(0) for m in _INJECTION.finditer(text)]


# --- sources (R2) ------------------------------------------------------------------


def normalize_source(ref: str, workspace: str = "") -> str:
    """A URL or path in one canonical form, so a citation can be matched to the call
    that opened it."""
    ref = ref.strip()
    if re.match(r"^https?://", ref, re.IGNORECASE):
        parts = urlsplit(ref)
        path = parts.path.rstrip("/")
        query = f"?{parts.query}" if parts.query else ""
        return f"{parts.scheme.lower()}://{parts.netloc.lower()}{path}{query}"
    path = posixpath.normpath(ref.replace("\\", "/"))
    if workspace and path.startswith(posixpath.normpath(workspace) + "/"):
        path = path[len(posixpath.normpath(workspace)) + 1 :]
    return path.removeprefix("./")
