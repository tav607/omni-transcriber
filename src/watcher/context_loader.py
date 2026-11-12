"""CONTEXT.md: per-folder settings for the Dropbox watcher.

A CONTEXT.md is a YAML front matter block followed by free-text background:

    ---
    mode: meeting
    participants:
      - Jane: the client
      - Sam
    glossary: [CoWoS, CCL]
    ---
    Weekly call with the client about ...

Every CONTEXT.md from the watch folder root down to the audio file's folder
applies, root first, so the root file holds what all folders share. A folder
overrides a setting by stating it; glossary terms and background text add up
instead.

Anything the loader cannot read or does not understand raises ContextError
and the file is not processed. The format this replaces took its settings
from `## Sections` / `## Translation` headings and failed quietly: a file that
could not be fetched counted as absent, a misspelled heading was ignored, and
the HTML comments of an unfilled template were sent to the model as the
folder's background.
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Awaitable, Callable

import yaml

from ..services.sections import MODES, SECTION_TYPES, Section

logger = logging.getLogger(__name__)

CONTEXT_FILENAME = "CONTEXT.md"
DEFAULT_MODE = "meeting"
ALLOWED_KEYS = ("mode", "roles", "participants", "glossary", "translation", "output_folder", "sections")

_FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_LEGACY_HEADING_RE = re.compile(
    r"^##\s+(background|sections|output[ _]format|translation)\s*$", re.MULTILINE | re.IGNORECASE)
_SECTION_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_LABEL_FORBIDDEN_RE = re.compile(r"[*:：\n]")
_SECTION_FIELDS = ("key", "prompt", "type", "heading")


class ContextError(Exception):
    """A CONTEXT.md that cannot be read or does not validate."""


@dataclass(frozen=True)
class ContextConfig:
    """Settings in force for one audio file."""

    mode: str = DEFAULT_MODE
    background: str = ""
    # Fixed turn labels for an interview: the asking side first, the answering
    # side last (the transcribe skill's --speakers).
    roles: tuple[str, ...] = ()
    # People expected on the recording, (name, description): a hint for
    # labeling turns, not a closed list.
    participants: tuple[tuple[str, str], ...] = ()
    glossary: tuple[str, ...] = ()
    translation: bool = False
    # Subfolder of the watcher's output folder; "" writes to the output folder.
    output_folder: str = ""
    # None: the mode's own sections.
    sections: tuple[Section, ...] | None = None
    # The CONTEXT.md files that applied, root first.
    sources: tuple[str, ...] = ()

    def describe(self) -> str:
        """One line for the status message, so a run shows what it ran with."""
        parts = [f"mode: {self.mode}"]
        if self.roles:
            parts.append("roles: " + ", ".join(self.roles))
        if self.participants:
            parts.append("participants: " + ", ".join(name for name, _ in self.participants))
        if self.glossary:
            parts.append(f"glossary: {len(self.glossary)} term(s)")
        if self.sections:
            parts.append("sections: " + ", ".join(s.key for s in self.sections))
        if self.translation:
            parts.append("translation: on")
        if self.output_folder:
            parts.append(f"output: {self.output_folder}")
        parts.append("config: " + (" + ".join(self.sources) if self.sources else "none (defaults)"))
        return " · ".join(parts)


def _is_safe_dropbox_path(path: str, allow_empty: bool = False) -> bool:
    """No '..' component. Checked before os.path.normpath, which removes them."""
    if not path:
        return allow_empty
    return ".." not in PurePosixPath(path).parts


def _string_list(value, key: str, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, (str, int, float)) for v in value):
        raise ContextError(f"{path}: `{key}` must be a list of strings")
    return tuple(str(v).strip() for v in value if str(v).strip())


def _label(value, key: str, path: str) -> str:
    name = str(value).strip()
    if not name or len(name) > 40 or _LABEL_FORBIDDEN_RE.search(name):
        raise ContextError(
            f"{path}: `{key}` entry {str(value)!r} cannot be a speaker label "
            "(1-40 characters, no '*', ':' or line break)")
    return name


def _participants(value, path: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise ContextError(f"{path}: `participants` must be a list")
    out = []
    for item in value:
        if isinstance(item, dict) and len(item) == 1:
            (name, description), = item.items()
            out.append((_label(name, "participants", path), str(description or "").strip()))
        elif isinstance(item, (str, int, float)):
            out.append((_label(item, "participants", path), ""))
        else:
            raise ContextError(
                f"{path}: each `participants` entry is a name, or `Name: description`; got {item!r}")
    return tuple(out)


def _sections(value, path: str) -> tuple[Section, ...]:
    if not isinstance(value, list) or not value:
        raise ContextError(f"{path}: `sections` must be a non-empty list")
    out: list[Section] = []
    for item in value:
        if not isinstance(item, dict):
            raise ContextError(f"{path}: each `sections` entry is a mapping with key and prompt; got {item!r}")
        unknown = [k for k in item if k not in _SECTION_FIELDS]
        if unknown:
            raise ContextError(
                f"{path}: unknown field(s) {', '.join(map(str, unknown))} in a `sections` entry "
                f"(allowed: {', '.join(_SECTION_FIELDS)})")
        key = str(item.get("key", "")).strip()
        if not _SECTION_KEY_RE.match(key):
            raise ContextError(
                f"{path}: section key {key!r} must be lowercase letters, digits and underscores, "
                "starting with a letter")
        if any(s.key == key for s in out):
            raise ContextError(f"{path}: section key {key!r} appears twice")
        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            raise ContextError(f"{path}: section {key!r} needs a `prompt` saying what to write")
        kind = str(item.get("type") or ("text" if key == "title" else "list")).strip()
        if kind not in SECTION_TYPES:
            raise ContextError(
                f"{path}: section {key!r} has type {kind!r} (allowed: {', '.join(SECTION_TYPES)})")
        if key == "title" and kind != "text":
            raise ContextError(f"{path}: the title section must be of type text")
        out.append(Section(key, prompt, kind, str(item.get("heading") or "").strip()))
    return tuple(out)


def _output_folder(value, path: str) -> str:
    folder = str(value or "").strip().strip("/")
    parts = PurePosixPath(folder).parts if folder else ()
    if "\\" in folder or any(p in ("..", ".") for p in parts) or _HTML_COMMENT_RE.search(folder):
        raise ContextError(f"{path}: `output_folder` must be a plain subfolder path, got {value!r}")
    return folder


def parse_context_md(content: str, path: str = CONTEXT_FILENAME) -> tuple[dict, str]:
    """(settings, background) of one CONTEXT.md. Settings holds only the keys
    the file states, validated and normalized."""
    content = content.lstrip("﻿")
    settings: dict = {}
    match = _FRONT_MATTER_RE.match(content)
    if match:
        try:
            raw = yaml.safe_load(match.group(1))
        except yaml.YAMLError as error:
            raise ContextError(f"{path}: the front matter is not valid YAML: {error}") from None
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ContextError(f"{path}: the front matter must be `key: value` lines")
        body = content[match.end():]
    else:
        if _LEGACY_HEADING_RE.search(content):
            raise ContextError(
                f"{path}: this is the old format (## Background / ## Sections / ## Translation). "
                "Settings now go in a YAML front matter block at the top; see the README")
        raw, body = {}, content

    unknown = [k for k in raw if k not in ALLOWED_KEYS]
    if unknown:
        raise ContextError(
            f"{path}: unknown setting(s) {', '.join(map(str, unknown))} "
            f"(allowed: {', '.join(ALLOWED_KEYS)})")

    if "mode" in raw:
        mode = str(raw["mode"]).strip()
        if mode not in MODES:
            raise ContextError(f"{path}: mode {mode!r} is not one of {', '.join(MODES)}")
        settings["mode"] = mode
    if "translation" in raw:
        if not isinstance(raw["translation"], bool):
            raise ContextError(f"{path}: `translation` must be true or false")
        settings["translation"] = raw["translation"]
    if "glossary" in raw:
        settings["glossary"] = _string_list(raw["glossary"] or [], "glossary", path)
    if "roles" in raw:
        roles = tuple(_label(v, "roles", path) for v in _string_list(raw["roles"] or [], "roles", path))
        if len(roles) == 1 or len(set(roles)) != len(roles):
            raise ContextError(f"{path}: `roles` needs at least two different labels, asking side first")
        settings["roles"] = roles
    if "participants" in raw:
        settings["participants"] = _participants(raw["participants"] or [], path)
    if settings.get("roles") and settings.get("participants"):
        raise ContextError(
            f"{path}: `roles` (fixed interview labels) and `participants` (expected names) "
            "cannot both be set")
    if "output_folder" in raw:
        settings["output_folder"] = _output_folder(raw["output_folder"], path)
    if "sections" in raw:
        settings["sections"] = _sections(raw["sections"], path) if raw["sections"] else None

    return settings, _HTML_COMMENT_RE.sub("", body).strip()


def merge_layers(layers: list[tuple[str, dict, str]]) -> ContextConfig:
    """Fold (path, settings, background) layers, root first, into one config."""
    values: dict = {}
    glossary: list[str] = []
    backgrounds: list[str] = []
    for _path, settings, background in layers:
        for key, value in settings.items():
            if key == "glossary":
                glossary += [term for term in value if term not in glossary]
            else:
                values[key] = value
        # The two ways of naming speakers exclude each other, so the folder
        # that states one drops the other it inherited.
        if settings.get("roles"):
            values.pop("participants", None)
        if settings.get("participants"):
            values.pop("roles", None)
        if background:
            backgrounds.append(background)
    return ContextConfig(
        mode=values.get("mode", DEFAULT_MODE),
        background="\n\n".join(backgrounds),
        roles=values.get("roles", ()),
        participants=values.get("participants", ()),
        glossary=tuple(glossary),
        translation=values.get("translation", False),
        output_folder=values.get("output_folder", ""),
        sections=values.get("sections"),
        sources=tuple(path for path, _, _ in layers),
    )


def context_paths(dropbox_audio_path: str, watch_folder: str) -> list[str]:
    """CONTEXT.md paths that apply to an audio file, root first."""
    if not _is_safe_dropbox_path(dropbox_audio_path):
        raise ContextError(f"unsafe audio path: {dropbox_audio_path!r}")
    if not _is_safe_dropbox_path(watch_folder, allow_empty=True):
        raise ContextError(f"unsafe watch folder: {watch_folder!r}")
    root = os.path.normpath(watch_folder).rstrip("/") if watch_folder else ""
    current = os.path.dirname(os.path.normpath(dropbox_audio_path)).rstrip("/")
    if root and current != root and not current.startswith(root + "/"):
        raise ContextError(f"{dropbox_audio_path!r} is outside the watch folder")
    dirs = [current]
    while current != root:
        current = os.path.dirname(current).rstrip("/")
        dirs.append(current)
    return [f"{d}/{CONTEXT_FILENAME}" for d in reversed(dirs)]


async def load_context(
    dropbox_audio_path: str,
    watch_folder: str,
    read_file: Callable[[str], Awaitable[str | None]],
) -> ContextConfig:
    """The settings for an audio file.

    read_file(path) returns a file's text, None when there is no such file,
    and raises when it cannot tell (ContextError, so the file waits instead
    of running on defaults).
    """
    layers = []
    for path in context_paths(dropbox_audio_path, watch_folder):
        content = await read_file(path)
        if content is None:
            continue
        settings, background = parse_context_md(content, path)
        layers.append((path, settings, background))
    config = merge_layers(layers)
    logger.info(f"Context for {dropbox_audio_path}: {config.describe()}")
    return config
