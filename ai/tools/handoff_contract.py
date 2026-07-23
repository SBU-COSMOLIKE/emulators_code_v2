#!/usr/bin/env python3
"""Validate a decision-complete Architect or Red Team directive.

The mailbox carries short routing summaries, while the cited temporary note
contains the complete plan.  This tool checks that note before a thinking role
hands work to an Implementer.  It checks structure and obvious placeholders;
it cannot decide whether the scientific design itself is correct.

Examples:

  python3 ai/tools/handoff_contract.py architect ai/notes/my-ticket.md
  python3 ai/tools/handoff_contract.py redteam ai/notes/my-ticket.md
"""

import argparse
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import unicodedata

try:
    from ai.tools.role_contract import ROLE_CONTRACT
except ImportError:  # Direct execution from ai/tools/.
    from role_contract import ROLE_CONTRACT

MAX_NOTE_BYTES = 2 * 1024 * 1024
CONTRACT_REPO_ROOT = Path(__file__).resolve().parents[2]

PACKET_TITLES = {
    "architect": "Implementation directive",
    "redteam": "Repair directive",
}

REQUIRED_SECTIONS = {
    "architect": (
        "Outcome",
        "Starting point",
        "Execution checkout",
        "Character-change budget",
        "Role plan",
        "Files and symbols",
        "Ordered implementation steps",
        "Interfaces and exact behavior",
        "Failure behavior and edge cases",
        "Tests to write",
        "Validation commands",
        "Acceptance checklist",
        "Do not change",
        "Stop and ask if",
        "Parallel work plan",
    ),
    "redteam": (
        "Finding and evidence",
        "Root cause",
        "Required outcome",
        "Character-change budget",
        "Files and symbols",
        "Ordered repair steps",
        "Exact invariants",
        "Regression test",
        "Validation commands",
        "Acceptance checklist",
        "Do not change",
        "Stop and ask if",
        "Architect adjudication required",
    ),
}

ARCHITECT_ROLE_PLANS = {
    "Architect + Implementer + Red Team": {
        "route": "three-role",
        "uses_red_team": True,
    },
    "Architect + Implementer": {
        "route": "two-role",
        "uses_red_team": False,
    },
}

HEADING_RE = re.compile(
    r"^[ ]{0,3}(#{1,6})(?:[ \t]+(.*?)[ \t]*|[ \t]*)$")
FENCE_OPEN_RE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})(.*)$")
LIST_FENCE_RE = re.compile(
    r"^[ ]{0,3}(?:[-+*]|\d+[.)])[ \t]+(?:`{3,}|~{3,})")
NUMBERED_STEP_RE = re.compile(
    r"^[ \t]*1[.)][ \t]+(.+)$", re.MULTILINE)
NUMBERED_ANY_STEP_RE = re.compile(
    r"^[ \t]*\d+[.)][ \t]+(.+)$", re.MULTILINE)
CHECKBOX_RE = re.compile(
    r"^[ \t]*-[ \t]+\[[ xX]\][ \t]+(.+)$", re.MULTILINE)
SUBAGENT_HEADING_RE = re.compile(
    r"^#### Subagent `([a-z][a-z0-9-]{1,47})`$")
SUBAGENT_FIELD_RE = re.compile(
    r"^- (Mode|Ownership|Task|Return|Acceptance|Stop):[ \t]+(.+)$")
INTEGRATOR_FIELD_RE = re.compile(
    r"^- (Integration|Final validation):[ \t]+(.+)$")
PARALLEL_LAUNCH_ROW = (
    "- Launch: `required before implementation edits`")
SUBAGENTS_REQUIRED_HEADING = "#### Subagents required"
SUBAGENTS_NOT_REQUIRED_HEADING = "#### Subagents not required"
SUBAGENTS_NOT_REQUIRED_REASON_RE = re.compile(r"^- Reason:[ \t]+(.+)$")
CAPABILITY_EXCEPTION_ROWS = (
    ("Capability checked",
     re.compile(r"^- Capability checked: `([^`\n]+)`$")),
    ("Attempted operation",
     re.compile(r"^- Attempted operation: (.+)$")),
    ("Raw failure",
     re.compile(r"^- Raw failure: `([^`\n]+)`$")),
)
CAPABILITY_CHECKPOINT_CYCLE_RE = re.compile(
    r"^- Source cycle: `([a-z0-9]+(?:-[a-z0-9]+)*@[0-9a-f]{40})`$")
CAPABILITY_CHECKPOINT_SHA256_RE = re.compile(
    r"^- Source handoff SHA-256: `([0-9a-f]{64})`$")
SUBAGENT_EVIDENCE_HEADING_RE = re.compile(
    r"^(?:-[ \t]+)?#### Subagent return (?:`([a-z][a-z0-9-]{1,47})`|"
    r"([a-z][a-z0-9-]{1,47}))$")
SUBAGENT_EVIDENCE_FIELD_RE = re.compile(
    r"^- (Returned artifact|Acceptance|Evidence):[ \t]+(.+)$")
IMPLEMENTER_SUBAGENT_EVIDENCE_MARKER = "- **Subagent work:**"
IMPLEMENTER_SUBAGENT_EVIDENCE_END_FIELD = "Blockers/findings"
IMPLEMENTER_HANDOFF_FIELD_RE = re.compile(
    r"^- \*\*([^*\n]+):\*\*(?:[ \t]+.*)?$")
PLACEHOLDER_BODY_RE = re.compile(
    r"^[ \t]*(?:\[[^\n]+\]|<[^\n]+>|TBD|TODO|FIXME|"
    r"your (?:text|answer|details) here)[.!]?[ \t]*$",
    re.IGNORECASE)
EMBEDDED_PLACEHOLDER_RE = re.compile(
    r"\b(?:TODO|TBD|FIXME)\b|"
    r"\[(?:write|add|list|state|name|describe|fill|insert|your)\b[^\]\n]*\]|"
    r"<[^>\n]{1,120}>|"
    r"\b(?:command|details?|text|answer) here\b|"
    r"(?:^|[ \t])\.\.\.(?:[ \t]|$)",
    re.IGNORECASE | re.MULTILINE)
UNRESOLVED_CHOICE_RE = re.compile(
    r"\b(?:use your best judgment|use your judgement|as appropriate|"
    r"whatever works|(?:choose|pick|select) whichever|"
    r"choose (?:either|which|between|one)|"
    r"decide (?:which|whether|between|later)|"
    r"pick (?:either|which|between|one)|"
    r"(?:decide|choose|pick)(?: on)? (?:a |an |the )?"
    r"(?:format|layout|algorithm|approach|design|implementation|storage|"
    r"serialization)|"
    r"select (?:a |an |the )?(?:suitable|appropriate|best)\b|"
    r"(?:the )?implementer (?:must |should |can |may |will )?"
    r"(?:choose|decide|determine|pick|select)s?\b|"
    r"(?:the )?implementer is responsible for (?:the )?"
    r"(?:choice|decision|design|format|layout)|"
    r"defer [^.\n]{0,80}(?:choice|decision|selection) [^.\n]{0,40}"
    r"to implementation|"
    r"(?:choice|decision|design|format|layout)[^.\n]{0,40}"
    r"(?:remains? (?:an? )?open|is left open)|"
    r"leave (?:the |this )?(?:choice|decision|design) to the implementer)\b",
    re.IGNORECASE)
ALTERNATIVE_CHOICE_RE = re.compile(
    r"(?:[^.\n]{0,160}\bor\b[^.\n]{0,160}|"
    r"\bone of\b[^.\n]{1,120}\band\b[^.\n]{1,120}|"
    r"\b(?:choose|select|pick)[ \t]+from\b"
    r"[^.\n]{1,120}\band\b[^.\n]{1,120}|"
    r"[^.\n]{1,120}\balternatively\b[^.\n]{1,120}|"
    r"[^.\n]{1,80}\bversus\b[^.\n]{1,80}|"
    r"\b(?:use|choose|select|pick|format(?:[ \t]+(?:is|may[ \t]+be))?)\b"
    r"[^.\n]{0,80}\b[A-Za-z0-9_+.-]+[ \t]*/[ \t]*"
    r"[A-Za-z0-9_+.-]+\b)",
    re.IGNORECASE)
CHOICE_RESOLUTION_RE = re.compile(
    r"\b(?:(?:according to|based on|selected by|determined by|depending on|"
    r"as specified by)[ \t]+(?:the[ \t]+)?"
    r"(?:existing|explicit|named|declared|configured|stored)\b|"
    r"from[ \t]+(?:the[ \t]+)?explicit\b|"
    r"using[ \t]+(?:the[ \t]+)?(?:existing|explicit|named|declared|"
    r"configured|stored)\b|normalize both|supports? both|accepts? both)\b",
    re.IGNORECASE)
BAD_CHOICE_RESOLVER_RE = re.compile(
    r"\b(?:whoever|whatever|whichever|"
    r"judg(?:e)?ment|preferences?|prefers?|convenience|"
    r"writing the code|seems? best|open (?:choice|decision)|left open)\b|"
    r"\b(?:according to|based on|selected by|determined by|depending on|"
    r"as specified by)[ \t]+(?:the[ \t]+)?"
    r"(?:(?:existing|explicit|named|declared|configured|stored)[ \t]+)?"
    r"(?:implementer|developer|coder|architect|author|person|red[ -]?team)"
    r"(?:'s)?\b",
    re.IGNORECASE)
NON_CHOICE_OR_RE = re.compile(
    r"\b(?:refuse|reject|fail if|stop if|do not|does not|never|must not|"
    r"neither|forbid|prohibit)\b",
    re.IGNORECASE)
CHOICE_CLAUSE_SPLIT_RE = re.compile(
    r"(?:;|:[ \t]+|[\u2013\u2014]|[ \t]+-[ \t]+|"
    r"(?:,[ \t]*|[ \t]+)(?:then|and|but|while|whereas)[ \t]+"
    r"(?=(?:use|uses|using|write|writes|writing|store|stores|storing|"
    r"output|outputs|outputting|emit|emits|emitting|choose|chooses|"
    r"choosing|pick|picks|picking|select|selects|selecting|return|returns|"
    r"returning|raise|raises|raising|throw|throws|throwing|treat|treats|"
    r"treating|create|creates|creating|delete|deletes|deleting|edit|edits|"
    r"editing|change|changes|changing|support|supports|supporting|handle|"
    r"handles|handling|parse|parses|parsing)\b))",
    re.IGNORECASE)
PAIRED_CONDITION_OR_RE = re.compile(
    r"\b(?:use|uses|using|write|writes|writing|store|stores|storing|"
    r"output|outputs|outputting|return|returns|returning|exit|exits|"
    r"yield|yields|yielding|emit|emits|emitting|choose|chooses|choosing|"
    r"select|selects|selecting)\b"
    r"(?:(?!\bor\b)[^.\n;])*\b(?:if|when|on|for)\b"
    r"(?:(?!\bor\b)[^.\n;])*\bor\b"
    r"(?:(?!\bor\b)[^.\n;])*\b(?:if|when|on|for)\b",
    re.IGNORECASE)
FAILURE_CONDITION_OR_RE = re.compile(
    r"\b(?:raise|raises|throw|throws|signal|signals|report|reports)\b"
    r"[^.\n;]*\b(?:if|when|unless)\b[^.\n;]*\bor\b"
    r"(?![ \t]*(?:use|write|store|output|emit|choose|pick|select|return|"
    r"raise|throw|treat|create|delete|edit|change)\b)[^.\n;]+",
    re.IGNORECASE)
NORMALIZATION_OR_RE = re.compile(
    r"\btreat\b[^.\n;]*\bor\b[^.\n;]*\bas\b",
    re.IGNORECASE)
MAPPING_OR_RE = re.compile(
    r"\b(?:map|maps|convert|converts|translate|translates)\b"
    r"[^.\n;]*\bto\b[^.\n;]*\bor\b[^.\n;]*\bto\b",
    re.IGNORECASE)
NEGATIVE_SCOPE_BREAK_RE = re.compile(
    r"\b(?:and|because|since|although|though|while|whereas|before|after|"
    r"once|then|but|however|instead|except|until|unless|therefore|thus|"
    r"so[ \t]+that|in[ \t]+order[ \t]+to)\b[^.\n;]{0,160}\b"
    r"(?:use|uses|using|write|writes|writing|store|stores|storing|output|"
    r"outputs|outputting|emit|emits|emitting|choose|chooses|choosing|pick|"
    r"picks|picking|select|selects|selecting|return|returns|returning|"
    r"support|supports|supporting|format[ \t]+(?:is|may[ \t]+be))\b",
    re.IGNORECASE)
LOCATOR_RE = re.compile(r"`([^`\n]+)::([^`\n]+)`")
CHECKOUT_LINE_RE = re.compile(
    r"^[ \t]*(?:-[ \t]+)?(Worktree|Branch|Base):[ \t]+`([^`\n]+)`[ \t]*$",
    re.MULTILINE)
COMMAND_FENCE_TAGS = ("bash", "sh", "shell", "zsh")
RAW_HTML_START_RE = re.compile(
    r"(?:<\?|<!\[CDATA\[|<![A-Za-z]|"
    r"</?[A-Za-z][A-Za-z0-9-]*(?:[ \t]|/?>|$))")
HTML_ENTITY_RE = re.compile(
    r"&(?:#[0-9]+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);")
CANONICAL_LOCATOR_LINE_RE = re.compile(
    r"^[ ]{0,3}-[ \t]+`([^`\n]+)::([^`\n]+)`[ \t]*:[ \t]+(.+)$")
CHARACTER_BUDGET_ROWS = (
    re.compile(r"^- Limit: `([0-9]+)`$"),
    re.compile(r"^- Planned maximum: `([0-9]+)`$"),
    re.compile(r"^- Readability plan: (.+)$"),
)
ROLE_PLAN_ROWS = (
    re.compile(
        r"^- Roles: `(Architect \+ Implementer \+ Red Team|"
        r"Architect \+ Implementer)`$"),
    re.compile(
        r"^- Discovery severity: `(high|medium|low|not-used)`$"),
    re.compile(
        r"^- Review scope: `(bounded|widespread|not-used)`$"),
    re.compile(
        r"^- Ticket class: `(ordinary|protected-control-plane)`$"),
)
REDTEAM_SEVERITY_ROWS = (
    ("User severity setting",
     re.compile(r"^- User severity setting: `(high|medium|low)`$")),
    ("Red Team severity",
     re.compile(r"^- Red Team severity: `(high|medium|low)`$")),
    ("Likelihood",
     re.compile(r"^- Likelihood: `(probable|improbable)`$")),
    ("Likelihood evidence",
     re.compile(r"^- Likelihood evidence: (.+)$")),
    ("Meets user setting",
     re.compile(r"^- Meets user setting: `(yes|no)`$")),
)
DISCOVERY_SEVERITIES = ("high", "medium", "low")
DEFAULT_DISCOVERY_SEVERITY = "medium"
DISCOVERY_SEVERITY_ENVIRONMENT = "MAILBOX_DISCOVERY_SEVERITY"
TICKET_CHANGE_GUARD = ROLE_CONTRACT["protected_paths"]["trusted_tools"][
    "ticket_change_guard"]


class DirectiveError(ValueError):
    """Raised when a directive note is incomplete or ambiguous."""


def _without_leading_frontmatter(text):
    """Mask one leading YAML frontmatter block while preserving line rows.

    Frontmatter is a metadata block between ``---`` delimiters at the
    very top of a Markdown file. Its lines are blanked rather than
    removed, so every later line keeps its original row number in
    error messages. An unterminated block blanks the whole text.

    Arguments:
      text = the Markdown text.

    Returns:
      The text with any leading frontmatter blanked.
    """
    lines = text.split("\n")
    delimiter_re = re.compile(r"^(?:---|\.\.\.)[ \t]*$")
    if not lines or re.fullmatch(r"---[ \t]*", lines[0]) is None:
        return text
    end = None
    for index, line in enumerate(lines[1:], start=1):
        if delimiter_re.fullmatch(line) is not None:
            end = index
            break
    if end is None:
        return "\n".join("" for _line in lines)
    return "\n".join([""] * (end + 1) + lines[end + 1:])


def _visible_without_comments(line, in_comment):
    """Remove Markdown HTML comments from one line without losing state.

    Arguments:
      line       = the line to clean.
      in_comment = True when an earlier line opened a comment that
                   has not closed yet.

    Returns:
      ``(visible, in_comment)``: the line without comment text (each
      removed comment leaves one space so its removal cannot
      manufacture heading syntax) and whether a comment is still
      open.
    """
    visible = []
    cursor = 0
    while cursor < len(line):
        if in_comment:
            close = line.find("-->", cursor)
            if close < 0:
                return "".join(visible), True
            cursor = close + 3
            in_comment = False
            continue
        opening = line.find("<!--", cursor)
        if opening < 0:
            visible.append(line[cursor:])
            break
        visible.append(line[cursor:opening])
        # A comment is an inline Markdown node, not an empty string. Keep one
        # separator so removing it cannot manufacture heading marker syntax.
        visible.append(" ")
        cursor = opening + 4
        in_comment = True
    return "".join(visible), in_comment


def _fence_opening(line):
    """Return ``(character, width, info)`` for one valid opening fence.

    A fence is the three-backtick or ``~~~`` line that opens a code
    block. A backtick fence whose info text itself contains a
    backtick is not a valid opening.

    Arguments:
      line = the line to inspect.

    Returns:
      The fence character, its repeat count, and the lowercased info
      string, or ``None`` when the line opens no fence.
    """
    match = FENCE_OPEN_RE.match(line)
    if match is None:
        return None
    marker = match.group(1)
    suffix = match.group(2)
    if marker[0] == "`" and "`" in suffix:
        return None
    return marker[0], len(marker), suffix.strip().casefold()


def _is_fence_close(line, character, width):
    """Return whether line is a CommonMark-style matching close fence.

    CommonMark is the standardized Markdown dialect this validator
    follows: a close fence repeats the opening character at least as
    many times, with at most three leading spaces.

    Arguments:
      line      = the line to inspect.
      character = the opening fence character.
      width     = the opening fence's repeat count.

    Returns:
      True for a matching close fence.
    """
    pattern = r"^[ ]{0,3}" + re.escape(character) + "{" + str(width) + r",}[ \t]*$"
    return re.match(pattern, line) is not None


def _visible_markdown_text(text):
    """Remove HTML comments outside fences while preserving line structure.

    Text inside code fences is kept exactly; comments elsewhere are
    removed with the one-space separator rule.

    Arguments:
      text = the Markdown text.

    Returns:
      The text with comments removed, one output row per input row.
    """
    rows = []
    fence_character = None
    fence_width = 0
    in_comment = False
    for line in text.split("\n"):
        if fence_character is not None:
            rows.append(line)
            if _is_fence_close(
                    line=line,
                    character=fence_character,
                    width=fence_width):
                fence_character = None
                fence_width = 0
            continue
        visible, in_comment = _visible_without_comments(
            line=line, in_comment=in_comment)
        rows.append(visible)
        opening = _fence_opening(line=visible)
        if opening is not None:
            fence_character, fence_width, _ = opening
    return "\n".join(rows)


def _indent_columns(line):
    """Return CommonMark indentation columns using four-column tab stops.

    Arguments:
      line = the line to measure.

    Returns:
      The indentation width in columns, counting a tab as advancing
      to the next multiple of four.
    """
    columns = 0
    for character in line:
        if character == " ":
            columns += 1
        elif character == "\t":
            columns += 4 - (columns % 4)
        else:
            break
    return columns


def _structural_markdown_text(text):
    """Mask fenced and indented code examples for prose-structure checks.

    Everything a Markdown renderer would treat as an example rather
    than prose — fenced blocks, four-column indented code, and
    blockquotes — is blanked, so a structure check cannot be fooled
    by heading-like text inside an example.

    Arguments:
      text = the Markdown text.

    Returns:
      The prose-only text, one output row per input row.
    """
    rows = []
    fence_character = None
    fence_width = 0
    in_comment = False
    in_blockquote = False
    for line in text.split("\n"):
        if fence_character is not None:
            rows.append("")
            if _is_fence_close(
                    line=line,
                    character=fence_character,
                    width=fence_width):
                fence_character = None
                fence_width = 0
            continue
        visible, in_comment = _visible_without_comments(
            line=line, in_comment=in_comment)
        if in_blockquote:
            rows.append("")
            if not visible.strip():
                in_blockquote = False
            continue
        opening = _fence_opening(line=visible)
        if opening is not None:
            fence_character, fence_width, _ = opening
            rows.append("")
            continue
        if _indent_columns(line=visible) >= 4:
            rows.append("")
            continue
        if re.match(r"^[ ]{0,3}>", visible) is not None:
            rows.append("")
            in_blockquote = True
            continue
        rows.append(visible)
    return "\n".join(rows)


def _binding_markdown_text(text):
    """Return visible non-example prose used to satisfy binding fields.

    A binding field is a directive row whose content carries
    authority, so its prose must be exactly what a reader sees:
    link-reference definitions are blanked, and inline links, images,
    HTML entities, and invisible control or format characters are
    refused outright rather than interpreted.

    Arguments:
      text = the Markdown text.

    Returns:
      The binding prose, one output row per input row.

    Raises:
      DirectiveError: for links, images, entities, or invisible
        characters inside binding prose.
    """
    rows = []
    in_reference = False
    for line in _structural_markdown_text(text=text).split("\n"):
        if in_reference:
            rows.append("")
            if not line.strip():
                in_reference = False
            continue
        # A link-reference definition is metadata, not rendered directive
        # prose.  Use a deliberately conservative row rule so escaped closing
        # brackets cannot smuggle a hidden body into a binding field.
        if (re.match(r"^[ ]{0,3}\[", line) is not None
                and re.search(r"(?<!\\)\]:", line) is not None):
            rows.append("")
            in_reference = True
            continue
        # Inline link destinations, image alt text, and entity spellings can
        # all contain characters that are absent from the rendered prose.
        # Binding fields use a small, canonical Markdown subset instead of
        # attempting to reproduce a complete renderer.  Supplemental links
        # and images belong outside the directive packet.
        if "![" in line or "](" in line:
            raise DirectiveError(
                "binding directive fields may not use inline Markdown links "
                "or images; write visible prose instead")
        entity = HTML_ENTITY_RE.search(line)
        if entity is not None:
            raise DirectiveError(
                "binding directive fields may not use HTML entities; write "
                "the visible character directly")
        for character in line:
            if (character not in ("\t", "\n")
                    and unicodedata.category(character) in ("Cc", "Cf")):
                raise DirectiveError(
                    "binding directive fields contain an invisible control "
                    "or format character")
        rows.append(line)
    return "\n".join(rows)


def _reject_setext_headings(text):
    """Keep binding packets on canonical ATX headings only.

    ATX headings start with ``#`` marks; setext headings instead
    underline the previous line with ``=`` or ``-``. Only the ATX
    form is accepted, so a heading can never appear by accident when
    prose happens to precede a dash row.

    Arguments:
      text = the Markdown text.

    Raises:
      DirectiveError: when a setext-style underline follows prose.
    """
    previous = ""
    for line in _structural_markdown_text(text=text).split("\n"):
        if (previous.strip()
                and re.fullmatch(r"[ ]{0,3}(?:=+|-+)[ \t]*", line)
                is not None):
            raise DirectiveError(
                "directive packets may not use Setext headings; use # "
                "heading markers")
        previous = line


def _reject_list_item_fences(text):
    """Refuse container-nested fences outside the packet's canonical subset.

    A fence indented as a list item renders as code but sits inside a
    container whose boundaries different renderers disagree on, so
    packets refuse the form instead of guessing.

    Arguments:
      text = the Markdown text.

    Raises:
      DirectiveError: for a fence nested inside a list item.
    """
    fence_character = None
    fence_width = 0
    in_comment = False
    for line in text.split("\n"):
        if fence_character is not None:
            if _is_fence_close(
                    line=line,
                    character=fence_character,
                    width=fence_width):
                fence_character = None
                fence_width = 0
            continue
        visible, in_comment = _visible_without_comments(
            line=line, in_comment=in_comment)
        if LIST_FENCE_RE.match(visible) is not None:
            raise DirectiveError(
                "directive packets may not use a fenced code block nested "
                "inside a list item")
        opening = _fence_opening(line=visible)
        if opening is not None:
            fence_character, fence_width, _ = opening


def _reject_display_math_blocks(text):
    """Refuse GFM display-math containers around binding Markdown rows.

    A GitHub-flavored-Markdown ``$$`` block swallows the rows inside
    it into rendered mathematics, which could hide binding text.

    Arguments:
      text = the Markdown text.

    Raises:
      DirectiveError: for a ``$$`` block delimiter outside a fence.
    """
    fence_character = None
    fence_width = 0
    in_comment = False
    for line in text.split("\n"):
        if fence_character is not None:
            if _is_fence_close(
                    line=line,
                    character=fence_character,
                    width=fence_width):
                fence_character = None
                fence_width = 0
            continue
        visible, in_comment = _visible_without_comments(
            line=line, in_comment=in_comment)
        opening = _fence_opening(line=visible)
        if opening is not None:
            fence_character, fence_width, _ = opening
            continue
        if re.fullmatch(r"[ ]{0,3}\$\$[ \t]*", visible) is not None:
            raise DirectiveError(
                "directive packets may not use display-math blocks around "
                "binding instructions")


def _reject_raw_html_blocks(text):
    """Refuse raw HTML block syntax outside code fences and comments.

    Raw HTML renders invisibly or inconsistently across viewers, so a
    directive may not carry it as prose.

    Arguments:
      text = the Markdown text.

    Raises:
      DirectiveError: for an HTML block start outside a fence or
        indented code.
    """
    fence_character = None
    fence_width = 0
    in_comment = False
    for line in text.split("\n"):
        if fence_character is not None:
            if _is_fence_close(
                    line=line,
                    character=fence_character,
                    width=fence_width):
                fence_character = None
                fence_width = 0
            continue
        visible, in_comment = _visible_without_comments(
            line=line, in_comment=in_comment)
        opening = _fence_opening(line=visible)
        if opening is not None:
            fence_character, fence_width, _ = opening
            continue
        if _indent_columns(line=visible) >= 4:
            continue
        if RAW_HTML_START_RE.search(visible) is not None:
            raise DirectiveError(
                "directive note contains raw HTML syntax; use visible "
                "Markdown prose instead")


def _heading_rows(text):
    """Return ``(line, level, title)`` rows for Markdown ATX headings.

    Headings inside code fences do not count; a trailing run of
    ``#`` marks is stripped from the title as CommonMark specifies.

    Arguments:
      text = the Markdown text.

    Returns:
      List of ``(line_number, level, title)`` for each real heading,
      where level counts the leading ``#`` marks.
    """
    rows = []
    fence_character = None
    fence_width = 0
    in_comment = False
    for line_number, line in enumerate(text.split("\n"), start=1):
        if fence_character is not None:
            if _is_fence_close(
                    line=line,
                    character=fence_character,
                    width=fence_width):
                fence_character = None
                fence_width = 0
            continue
        visible, in_comment = _visible_without_comments(
            line=line, in_comment=in_comment)
        opening = _fence_opening(line=visible)
        if opening is not None:
            fence_character, fence_width, _ = opening
            continue
        match = HEADING_RE.match(visible)
        if match is not None:
            title = (match.group(2) or "").rstrip()
            closing = re.match(r"^(.*?)[ \t]+#+$", title)
            if closing is not None and closing.group(1).strip():
                title = closing.group(1).rstrip()
            rows.append((line_number, len(match.group(1)), title))
    return rows


def _packet_bounds(text, title):
    """Return the line interval below one exact level-two packet heading.

    A packet is one machine-validated block of the note under a
    ``## `` heading; the interval runs to the next heading of level
    two or one, or to the end of the file.

    Arguments:
      text  = the Markdown text.
      title = the packet's exact heading title, compared
              case-insensitively.

    Returns:
      ``(lines, start, end)``: the split lines and the packet's line
      interval.

    Raises:
      DirectiveError: when the heading does not appear exactly once.
    """
    lines = text.split("\n")
    matches = []
    for line_number, level, heading in _heading_rows(text=text):
        if level == 2 and heading.casefold() == title.casefold():
            matches.append(line_number)
    if len(matches) != 1:
        raise DirectiveError(
            "expected exactly one '## " + title + "' heading; found "
            + str(len(matches)))
    start = matches[0]
    end = len(lines) + 1
    for line_number, level, _ in _heading_rows(text=text):
        if line_number > start and level <= 2:
            end = line_number
            break
    return lines, start, end


def _section_bodies(text, title, required):
    """Return exact required level-three section bodies in declared order.

    Arguments:
      text     = the Markdown text.
      title    = the packet's level-two heading title.
      required = the exact ordered ``###`` headings the packet must
                 contain, no more and no fewer.

    Returns:
      Mapping from heading to its comment-free body with surrounding
      blank rows trimmed.

    Raises:
      DirectiveError: when the headings differ from the required
        order in any way.
    """
    lines, packet_start, packet_end = _packet_bounds(text=text, title=title)
    headings = []
    for line_number, level, heading in _heading_rows(text=text):
        if packet_start < line_number < packet_end and level == 3:
            headings.append((line_number, heading))

    actual = tuple(heading for _, heading in headings)
    if actual != required:
        raise DirectiveError(
            "'## " + title + "' requires these level-three headings in "
            "this exact order: " + ", ".join(required) + "; found: "
            + (", ".join(actual) if actual else "none"))

    bodies = {}
    for index, (line_number, heading) in enumerate(headings):
        next_line = packet_end
        if index + 1 < len(headings):
            next_line = headings[index + 1][0]
        body = "\n".join(lines[line_number:next_line - 1])
        visible_rows = _visible_markdown_text(text=body).split("\n")
        while visible_rows and not visible_rows[0].strip():
            visible_rows.pop(0)
        while visible_rows and not visible_rows[-1].strip():
            visible_rows.pop()
        bodies[heading] = "\n".join(visible_rows)
    return bodies


def _require_evidence_destination(text, packet_title):
    """Require and return the Architect packet's sibling evidence body.

    The directive packet must be followed immediately by the
    ``## Implementation evidence / resume state`` heading, and that
    heading may repeat only consecutively, so evidence can never be
    scattered around the note.

    Arguments:
      text         = the note's Markdown text.
      packet_title = the directive packet's heading title.

    Returns:
      The combined evidence body, or ``None`` when the note has no
      such packet.

    Raises:
      DirectiveError: for a missing or misplaced evidence heading.
    """
    boundary_rows = [(line, level, heading)
                     for line, level, heading in _heading_rows(text=text)
                     if level <= 2]
    packet_rows = [index for index, (_line, level, heading)
                   in enumerate(boundary_rows)
                   if (level == 2
                       and heading.casefold() == packet_title.casefold())]
    if len(packet_rows) != 1:
        return
    packet_index = packet_rows[0]
    expected = "Implementation evidence / resume state"
    if (packet_index + 1 >= len(boundary_rows)
            or boundary_rows[packet_index + 1][1] != 2
            or boundary_rows[packet_index + 1][2].casefold()
            != expected.casefold()):
        raise DirectiveError(
            "'## " + packet_title + "' must be followed immediately by "
            "the sibling '## " + expected + "' heading")
    evidence_rows = [
        index for index, (_line, level, heading) in enumerate(boundary_rows)
        if level == 2 and heading.casefold() == expected.casefold()]
    consecutive = []
    index = packet_index + 1
    while (index < len(boundary_rows)
           and boundary_rows[index][1] == 2
           and boundary_rows[index][2].casefold() == expected.casefold()):
        consecutive.append(index)
        index += 1
    if evidence_rows != consecutive:
        raise DirectiveError(
            "'## " + expected + "' may repeat only consecutively after the "
            "directive packet")
    evidence_line = boundary_rows[consecutive[-1]][0]
    end_line = len(text.split("\n")) + 1
    for line_number, _level, _heading in boundary_rows[index:]:
        end_line = line_number
        break
    lines = text.split("\n")
    return "\n".join(lines[evidence_line:end_line - 1])


def _require_prior_capability_checkpoint(evidence_body,
                                         parallel_work_plan):
    """Bind a no-subagent exception to a prior Implementer checkpoint.

    A capability-unavailable plan is honest only when a real prior
    Implementer turn recorded the failed launch. The evidence must
    hold exactly one checkpoint heading whose rows repeat the plan's
    capability, operation, and raw failure word for word, plus the
    source cycle and handoff digest that identify the prior turn.

    Arguments:
      evidence_body      = the evidence section's text.
      parallel_work_plan = the parsed plan claiming the exception.

    Returns:
      Mapping with the checkpoint's ``cycle`` and
      ``handoff_sha256``.

    Raises:
      DirectiveError: for a missing, duplicated, or inexact
        checkpoint.
    """
    structural = _binding_markdown_text(text=evidence_body)
    lines = [line.strip() for line in structural.split("\n")
             if line.strip()]
    heading = "### Prior Implementer subagent launch failure"
    starts = [index for index, line in enumerate(lines) if line == heading]
    if len(starts) != 1:
        raise DirectiveError(
            "a capability-unavailable Parallel work plan requires exactly "
            "one '### Prior Implementer subagent launch failure' checkpoint "
            "under 'Implementation evidence / resume state'")
    start = starts[0]
    if start + 2 >= len(lines):
        raise DirectiveError(
            "the prior Implementer capability checkpoint must name its "
            "full Source cycle and Source handoff SHA-256")
    cycle_match = CAPABILITY_CHECKPOINT_CYCLE_RE.fullmatch(lines[start + 1])
    sha_match = CAPABILITY_CHECKPOINT_SHA256_RE.fullmatch(lines[start + 2])
    if cycle_match is None or sha_match is None:
        raise DirectiveError(
            "the prior Implementer capability checkpoint must name exact "
            "Source cycle and Source handoff SHA-256 rows")
    expected = [
        heading,
        lines[start + 1],
        lines[start + 2],
        "- Source: `prior same-cycle IMPLEMENTER_HANDOFF checkpoint`",
        "- Capability checked: `"
        + parallel_work_plan["capability_checked"] + "`",
        "- Attempted operation: "
        + parallel_work_plan["attempted_operation"],
        "- Raw failure: `" + parallel_work_plan["raw_failure"] + "`",
    ]
    if lines[start:start + len(expected)] != expected:
        raise DirectiveError(
            "the prior Implementer capability checkpoint must repeat the "
            "exact Capability checked, Attempted operation, and Raw failure "
            "rows from the revised Parallel work plan")
    return {
        "cycle": cycle_match.group(1),
        "handoff_sha256": sha_match.group(1),
    }


def _require_substance(bodies):
    """Refuse empty, placeholder, or explicitly delegated design choices.

    Every section must carry enough visible prose to execute, no
    template placeholder may remain, and — except in the sections
    validated separately — no unresolved design choice may be left to
    the Implementer.

    Arguments:
      bodies = mapping from section heading to its body text.

    Raises:
      DirectiveError: naming the failing section and the exact
        placeholder or choice.
    """
    for heading, body in bodies.items():
        binding_body = _binding_markdown_text(text=body)
        evaluated_body = (body if heading == "Validation commands"
                          else binding_body)
        compact = " ".join(evaluated_body.split())
        alphanumeric_count = sum(
            1 for character in compact if character.isalnum())
        if len(compact) < 12 or alphanumeric_count < 6:
            raise DirectiveError(
                "section '" + heading + "' is too short to be executable")
        if PLACEHOLDER_BODY_RE.fullmatch(evaluated_body) is not None:
            raise DirectiveError(
                "section '" + heading + "' still contains a placeholder")
        placeholder = EMBEDDED_PLACEHOLDER_RE.search(evaluated_body)
        if placeholder is not None:
            raise DirectiveError(
                "section '" + heading + "' contains unresolved placeholder '"
                + placeholder.group(0).strip() + "'")
        if heading == "Execution checkout":
            choice = None
        else:
            choice = _unresolved_choice(
                body=binding_body,
                ignore_locator_spans=heading in (
                    "Files and symbols", "Tests to write",
                    "Regression test"))
        if choice is not None:
            raise DirectiveError(
                "section '" + heading + "' delegates an unresolved design "
                "choice with '" + choice.group(0) + "'")


def _unresolved_choice(body, ignore_locator_spans=False):
    """Return one explicit unruled choice, allowing a named resolver.

    An unresolved choice is prose that offers the Implementer
    alternatives ("X or Y", "either ... or ...") without one named
    machine rule that decides them. Concrete condition patterns and
    negated scopes do not count; one named resolver may bless exactly
    one alternative pair per clause.

    Arguments:
      body                 = the binding prose to scan.
      ignore_locator_spans = True in sections whose ``path::symbol``
                             code spans are validated separately.

    Returns:
      The first offending regular-expression match, or ``None`` when
      every alternative is ruled.
    """
    # Inline-code spans are visible binding prose. Remove their Markdown
    # delimiters, never their content, so ``Use `JSON or YAML` `` cannot hide
    # an unresolved design choice. Canonical ``path::symbol`` code spans are
    # structural locators, not prose alternatives, and may be masked only in
    # the locator sections whose rows are validated separately. Fenced
    # examples were already masked by ``_binding_markdown_text``.
    def inline_code(match):
        """Unwrap one code span, masking only locator spans."""
        content = match.group(1)
        if ignore_locator_spans and "::" in content:
            return ""
        return content

    choice_text = re.sub(r"`([^`\n]*)`", inline_code, body)
    choice_text = re.sub(
        r"\bvs\.", " versus ", choice_text, flags=re.IGNORECASE)
    sentences = re.split(r"[.\n]+", choice_text)
    clauses = (
        clause.strip()
        for sentence in sentences
        for clause in CHOICE_CLAUSE_SPLIT_RE.split(sentence)
        if clause.strip())
    for clause in clauses:
        direct = UNRESOLVED_CHOICE_RE.search(clause)
        if direct is not None:
            return direct
        bad_resolver = BAD_CHOICE_RESOLVER_RE.search(clause)
        concrete_rules = [
            match
            for pattern in (
                PAIRED_CONDITION_OR_RE, FAILURE_CONDITION_OR_RE,
                NORMALIZATION_OR_RE, MAPPING_OR_RE)
            for match in pattern.finditer(clause)]
        or_matches = list(re.finditer(r"\bor\b", clause, re.IGNORECASE))
        resolution = CHOICE_RESOLUTION_RE.search(clause)
        negative_matches = list(NON_CHOICE_OR_RE.finditer(clause))
        for alternative in or_matches:
            if bad_resolver is not None:
                return alternative
            if any(rule.start() <= alternative.start() < rule.end()
                   for rule in concrete_rules):
                continue
            governing_negative = [
                match for match in negative_matches
                if match.end() <= alternative.start()]
            if governing_negative:
                last_negative = governing_negative[-1]
                scope = clause[last_negative.end():alternative.start()]
                if NEGATIVE_SCOPE_BREAK_RE.search(scope) is None:
                    continue
            # One named machine rule may resolve one alternative pair. More
            # than one ``or`` needs separate clauses/rules so a later resolver
            # cannot accidentally bless an earlier open choice.
            if resolution is not None and len(or_matches) == 1:
                continue
            return alternative
        if or_matches:
            continue
        for alternative in ALTERNATIVE_CHOICE_RE.finditer(clause):
            if bad_resolver is not None:
                return alternative
            governing_negative = [
                match for match in negative_matches
                if match.end() <= alternative.start()]
            if governing_negative:
                last_negative = governing_negative[-1]
                scope = clause[last_negative.end():alternative.start()]
                if NEGATIVE_SCOPE_BREAK_RE.search(scope) is None:
                    continue
            if resolution is None:
                return alternative
    return None


def _has_substantive_payload(
        text, minimum_alphanumeric=6, minimum_words=2):
    """Return whether one structured row carries visible executable prose.

    Arguments:
      text                 = the row's text.
      minimum_alphanumeric = least alphanumeric characters required.
      minimum_words        = least words required.

    Returns:
      True when the row meets both minimums.
    """
    compact = " ".join(text.split())
    words = re.findall(r"[^\W_]+", compact, flags=re.UNICODE)
    return (
        sum(1 for character in compact if character.isalnum())
        >= minimum_alphanumeric
        and len(words) >= minimum_words)


def _valid_locator_rows(body):
    """Return concrete ``repo/path::symbol`` locator pairs from one body.

    A locator names one repository file and one symbol inside it.
    Only concrete pairs survive: absolute paths, drive letters, URL
    schemes, backslashes, parent-directory parts, glob characters,
    placeholder names like ``some_file`` or ``example.py``, and
    generic templates like ``path/to/file`` are all dropped.

    Arguments:
      body = the text to scan.

    Returns:
      List of ``(path, symbol)`` pairs.
    """
    rows = []
    for match in LOCATOR_RE.finditer(body):
        path = match.group(1).strip()
        symbol = match.group(2).strip()
        normalized = path.replace("\\", "/")
        canonical = PurePosixPath(normalized).as_posix()
        parts = PurePosixPath(normalized).parts
        basename = PurePosixPath(normalized).name
        scheme_path = re.match(
            r"^[A-Za-z][A-Za-z0-9+.-]*:", path) is not None
        glob_syntax = re.search(r"[*?\[\]{}]", path + symbol) is not None
        placeholder_path = (
            any(re.fullmatch(r"(?:some|your)[_-].*", part.casefold())
                is not None for part in parts)
            or (len(parts) == 1 and basename.casefold() == "example.py"))
        placeholder_symbol = re.fullmatch(
            r"(?:some|your)[_-].*|whatever|all[_ -]?functions?|"
            r"all[_ -]?symbols?|anything|relevant(?:[_ -].*)?",
            symbol.casefold()) is not None
        generic_path = (
            normalized.casefold() in {
                "repo/path", "path/to/file", "path/to/test",
                "path/to/source", "some/path", "your/file.py"}
            or normalized.casefold().startswith("path/to/")
            or any(part.casefold() in ("some", "your") for part in parts))
        if (not path or not symbol or normalized != path
                or canonical != normalized or normalized.startswith("/")
                or re.match(r"^[A-Za-z]:", normalized) is not None
                or scheme_path or "\\" in path
                or ".." in parts or generic_path or glob_syntax
                or placeholder_path or placeholder_symbol
                or ("." not in basename
                    and basename not in ("Dockerfile", "Makefile"))):
            continue
        rows.append((path, symbol))
    return rows


def _require_locator(bodies, heading):
    """Require one canonical visible repository locator bullet.

    Every nonempty row of the section must be one bullet of the exact
    form ``- `repo/path::symbol`: exact edit`` with a concrete pair
    and a substantive description; anything else — a bare bullet,
    loose prose, a generic path or symbol, or a placeholder — is
    refused by name.

    Arguments:
      bodies  = the parsed section bodies.
      heading = the locator section to validate.

    Returns:
      The list of ``(path, symbol)`` pairs, at least one.

    Raises:
      DirectiveError: naming the first offending row.
    """
    structural = _binding_markdown_text(text=bodies[heading])
    rows = []
    generic_paths = {"repo/path", "path/to/file", "path/to/test"}
    generic_symbols = {
        "symbol", "section", "symbol-or-section", "test", "test-name",
        "function", "function-name", "method", "method-name", "class",
        "class-name", "some_symbol", "your_symbol"}
    generic_descriptions = {
        "exact edit", "exact test", "edit here", "test here"}
    for line in structural.split("\n"):
        if not line.strip():
            continue
        match = CANONICAL_LOCATOR_LINE_RE.match(line)
        if match is not None:
            spans = list(LOCATOR_RE.finditer(line))
            if (len(spans) != 1
                    or spans[0].start() != match.start(1) - 1):
                raise DirectiveError(
                    "section '" + heading + "' requires exactly one "
                    "locator at the start of every canonical bullet")
            path = match.group(1).strip()
            symbol = match.group(2).strip()
            description = match.group(3).strip()
            pairs = _valid_locator_rows(
                body="`" + path + "::" + symbol + "`")
            generic = (
                path.casefold() in generic_paths
                or symbol.casefold() in generic_symbols
                or description.casefold().rstrip(".") in generic_descriptions)
            if (pairs and not generic
                    and _has_substantive_payload(
                        description,
                        minimum_alphanumeric=16,
                        minimum_words=3)
                    and PLACEHOLDER_BODY_RE.fullmatch(description) is None
                    and EMBEDDED_PLACEHOLDER_RE.search(description) is None):
                rows.extend(pairs)
            else:
                raise DirectiveError(
                    "section '" + heading + "' has a generic or "
                    "non-substantive visible bullet locator; every locator "
                    "must name one concrete file, symbol, and exact edit")
        elif re.match(r"^[ ]{0,3}-[ \t]+", line) is not None:
            raise DirectiveError(
                "section '" + heading + "' has a locator bullet outside "
                "the canonical `repo/path::symbol`: exact edit form")
        else:
            raise DirectiveError(
                "section '" + heading + "' may contain only a canonical "
                "visible bullet locator in `repo/path::symbol`: exact edit "
                "form")
    if not rows:
        raise DirectiveError(
            "section '" + heading + "' requires a visible bullet locator "
            "that starts with `repo/path::symbol-or-section`: followed by "
            "its exact edit")
    return rows


def _require_execution_checkout(body):
    """Require the exact worktree, branch, and base selected by Architect.

    The section may contain only the three backticked rows, each
    exactly once. The worktree must be a literal absolute path with
    no traversal or shell characters; the branch may not be main and
    must be a plain reference name; the base must be one full commit.

    Arguments:
      body = the section's text.

    Returns:
      Mapping with the ``Worktree``, ``Branch``, and ``Base`` values.

    Raises:
      DirectiveError: naming the first violated row rule.
    """
    values = {field: [] for field in ("Worktree", "Branch", "Base")}
    structural = _binding_markdown_text(text=body)
    for line in structural.split("\n"):
        if not line.strip():
            continue
        match = CHECKOUT_LINE_RE.fullmatch(line)
        if match is None:
            raise DirectiveError(
                "section 'Execution checkout' may contain only the exact "
                "Worktree, Branch, and Base rows; contradictory or extra "
                "prose is not allowed")
        values[match.group(1)].append(match.group(2).strip())
    invalid_counts = [field for field, rows in values.items()
                      if len(rows) != 1]
    if invalid_counts:
        raise DirectiveError(
            "section 'Execution checkout' requires exactly one backticked "
            "field for each of: Worktree, Branch, Base; invalid: "
            + ", ".join(invalid_counts))

    worktree = values["Worktree"][0]
    worktree_parts = PurePosixPath(worktree).parts
    if (not PurePosixPath(worktree).is_absolute()
            or ".." in worktree_parts
            or re.search(r"[\x00-\x1f\x7f$`*?\[\]{};|&<>()\\]",
                         worktree) is not None):
        raise DirectiveError(
            "Execution checkout Worktree must be one literal absolute path "
            "without parent traversal or shell expansion characters")

    branch = values["Branch"][0]
    if (branch in ("main", "refs/heads/main")
            or re.fullmatch(r"[A-Za-z0-9._/-]+", branch) is None
            or ".." in branch or "//" in branch
            or branch.startswith("/") or branch.endswith("/")
            or branch.endswith(".lock")):
        raise DirectiveError(
            "Execution checkout Branch must name one well-formed non-main "
            "branch")

    base = values["Base"][0]
    if re.fullmatch(r"[0-9a-fA-F]{40}", base) is None:
        raise DirectiveError(
            "Execution checkout Base must be one full 40-hex commit")
    return {
        "Worktree": worktree,
        "Branch": branch,
        "Base": base.lower(),
    }


def _require_character_change_budget(body, expected_max):
    """Return one exact, policy-matched character-change budget.

    The directive stores the run-time limit so it survives mailbox relays.
    Only the three canonical visible rows are accepted; supplemental budget
    reasoning belongs in the readability-plan row rather than in free-form
    fields that a lower-capability Implementer would have to interpret.
    """
    if (expected_max is not None
            and (isinstance(expected_max, bool)
                 or not isinstance(expected_max, int)
                 or expected_max < 0)):
        raise DirectiveError(
            "expected character-change limit must be a nonnegative integer")
    structural = _binding_markdown_text(text=body)
    rows = [line for line in structural.split("\n") if line.strip()]
    if len(rows) != len(CHARACTER_BUDGET_ROWS):
        raise DirectiveError(
            "section 'Character-change budget' requires exactly these rows "
            "in order: - Limit: `N`; - Planned maximum: `K`; "
            "- Readability plan: visible prose")
    matches = []
    for row, pattern in zip(rows, CHARACTER_BUDGET_ROWS):
        match = pattern.fullmatch(row)
        if match is None:
            raise DirectiveError(
                "section 'Character-change budget' requires exactly these "
                "rows in order: - Limit: `N`; - Planned maximum: `K`; "
                "- Readability plan: visible prose")
        matches.append(match)
    limit_text = matches[0].group(1)
    planned_text = matches[1].group(1)
    limit = int(limit_text)
    planned_maximum = int(planned_text)
    readability_plan = matches[2].group(1).strip()
    if limit_text != str(limit) or planned_text != str(planned_maximum):
        raise DirectiveError(
            "section 'Character-change budget' decimal values must use "
            "their exact canonical spelling without leading zeros")
    if expected_max is not None and limit != expected_max:
        raise DirectiveError(
            "section 'Character-change budget' limit " + str(limit)
            + " does not match the run-time --max value "
            + str(expected_max))
    if limit > 0 and planned_maximum > limit:
        raise DirectiveError(
            "section 'Character-change budget' planned maximum "
            + str(planned_maximum) + " exceeds limit " + str(limit))
    if not _has_substantive_payload(
            readability_plan,
            minimum_alphanumeric=16,
            minimum_words=3):
        raise DirectiveError(
            "section 'Character-change budget' needs a substantive visible "
            "readability plan")
    return {
        "limit": limit,
        "planned_maximum": planned_maximum,
        "readability_plan": readability_plan,
    }


def _require_architect_role_plan(body):
    """Return the exact role plan written by the Architect.

    The manual router may verify command-line confirmations against this
    section, but it must never let those confirmations replace the plan in
    the source note.
    """
    structural = _binding_markdown_text(text=body)
    rows = [line for line in structural.split("\n") if line.strip()]
    if len(rows) != len(ROLE_PLAN_ROWS):
        raise DirectiveError(
            "section 'Role plan' requires exactly these rows in order: "
            "- Roles: `...`; - Discovery severity: `...`; "
            "- Review scope: `...`; - Ticket class: `...`")
    matches = []
    for row, pattern in zip(rows, ROLE_PLAN_ROWS):
        match = pattern.fullmatch(row)
        if match is None:
            raise DirectiveError(
                "section 'Role plan' requires one supported Roles value "
                "followed by one Discovery severity value and one Review "
                "scope value and one Ticket class value")
        matches.append(match)

    roles = matches[0].group(1)
    severity = matches[1].group(1)
    review_scope = matches[2].group(1)
    ticket_class = matches[3].group(1)
    plan = dict(ARCHITECT_ROLE_PLANS[roles])
    if plan["uses_red_team"] and severity == "not-used":
        raise DirectiveError(
            "section 'Role plan' must name high, medium, or low discovery "
            "severity when the Red Team is included")
    if not plan["uses_red_team"] and severity != "not-used":
        raise DirectiveError(
            "section 'Role plan' must use discovery severity `not-used` "
            "when the Red Team is not included")
    if plan["uses_red_team"] and review_scope == "not-used":
        raise DirectiveError(
            "section 'Role plan' must use review scope `bounded` or "
            "`widespread` when the Red Team is included")
    if not plan["uses_red_team"] and review_scope != "not-used":
        raise DirectiveError(
            "section 'Role plan' must use review scope `not-used` when the "
            "Red Team is not included")
    if review_scope == "widespread" and severity != "low":
        raise DirectiveError(
            "section 'Role plan' widespread review scope requires discovery "
            "severity `low`")
    if ticket_class == "protected-control-plane":
        raise DirectiveError(
            "section 'Role plan' protected-control-plane is reserved for "
            "Architect-owned ai/notes administration and is not an "
            "Implementer ticket class; keep an ai/tools ticket Open for "
            "external maintenance")
    plan["roles"] = roles
    plan["discovery_severity"] = severity
    plan["review_scope"] = review_scope
    plan["ticket_class"] = ticket_class
    return plan


def _parallel_payload(field, value, minimum_alphanumeric=24,
                      minimum_words=5,
                      context="section 'Parallel work plan'"):
    """Require one concrete, visible action or observable result.

    Arguments:
      field                = the plan field being checked.
      value                = its text.
      minimum_alphanumeric = least alphanumeric characters required.
      minimum_words        = least words required.
      context              = section name for error messages.

    Raises:
      DirectiveError: for thin, placeholder, or vague wording such as
        "as needed" or "use best judgment".
    """
    if (not _has_substantive_payload(
            value,
            minimum_alphanumeric=minimum_alphanumeric,
            minimum_words=minimum_words)
            or PLACEHOLDER_BODY_RE.fullmatch(value) is not None
            or EMBEDDED_PLACEHOLDER_RE.search(value) is not None):
        raise DirectiveError(
            context + " field '" + field
            + "' needs concrete, non-placeholder detail")
    generic = re.search(
        r"\b(?:as needed|where useful|if useful|use best judg(?:e)?ment|"
        r"help with|work on|handle (?:it|this|the task)|"
        r"do (?:the )?(?:work|task|thing)|report results?|see above|"
        r"same as above)\b",
        value,
        re.IGNORECASE)
    if generic is not None:
        raise DirectiveError(
            context + " field '" + field
            + "' is vague at '" + generic.group(0) + "'")


def _subagent_ownership(value, mode, name):
    """Return exact repository locators owned by one named subagent.

    Ownership is either the literal ``none (read-only)`` (valid only
    for a read-only subagent) or a comma-separated list of exact
    backticked ``repo/path::symbol`` entries with no duplicates.

    Arguments:
      value = the Ownership field text.
      mode  = the subagent's declared mode.
      name  = the subagent's name, for error messages.

    Returns:
      The list of owned ``(path, symbol)`` pairs; empty when
      read-only.

    Raises:
      DirectiveError: for malformed, generic, or repeated entries.
    """
    if value == "`none (read-only)`":
        if mode != "read-only":
            raise DirectiveError(
                "edit Subagent '" + name + "' cannot use "
                "`none (read-only)` ownership")
        return []

    tokens = re.findall(r"`([^`\n]+)`", value)
    canonical = ", ".join("`" + token + "`" for token in tokens)
    if not tokens or canonical != value:
        raise DirectiveError(
            "Subagent '" + name + "' Ownership must be "
            "`none (read-only)` or a comma-separated list of exact "
            "backticked `repo/path::symbol` entries")
    ownership = []
    for token in tokens:
        rows = _valid_locator_rows(body="`" + token + "`")
        if (token.count("::") != 1 or len(rows) != 1
                or token != rows[0][0] + "::" + rows[0][1]):
            raise DirectiveError(
                "Subagent '" + name + "' has a generic or malformed "
                "Ownership entry `" + token + "`")
        if rows[0] in ownership:
            raise DirectiveError(
                "Subagent '" + name + "' repeats Ownership `"
                + token + "`")
        ownership.append(rows[0])
    return ownership


def _capability_exception(lines):
    """Parse the sole permitted exception to mandatory subagent launch.

    A plan with no launched subagents must record exactly the three
    rows — Capability checked, Attempted operation, Raw failure — and
    each must be concrete: the capability names the launch operation,
    the operation describes really attempting a subagent launch
    before implementation edits, and the failure is the runtime's raw
    text.

    Arguments:
      lines = the plan's remaining rows.

    Returns:
      Mapping with the three verified field values.

    Raises:
      DirectiveError: naming the first missing or vague row.
    """
    if len(lines) != len(CAPABILITY_EXCEPTION_ROWS):
        raise DirectiveError(
            "section 'Parallel work plan' without launched subagents "
            "requires exactly Capability checked, Attempted operation, and "
            "Raw failure rows")
    values = {}
    for line, (field, pattern) in zip(lines, CAPABILITY_EXCEPTION_ROWS):
        match = pattern.fullmatch(line)
        if match is None:
            raise DirectiveError(
                "section 'Parallel work plan' capability exception requires "
                "the exact ordered fields Capability checked, Attempted "
                "operation, and Raw failure")
        values[field] = match.group(1).strip()

    capability = values["Capability checked"]
    if (re.fullmatch(r"[A-Za-z0-9_.:/-]{3,120}", capability) is None
            or capability.casefold() in {
                "none", "unknown", "unavailable", "not-applicable"}):
        raise DirectiveError(
            "Capability checked must name the exact launch operation that "
            "the runtime was asked to provide")

    attempted = values["Attempted operation"]
    _parallel_payload("Attempted operation", attempted)
    if (re.search(r"\b(?:launch|spawn|delegate|invoke|call)\w*\b",
                  attempted, re.IGNORECASE) is None
            or re.search(r"\bsubagent\b", attempted, re.IGNORECASE) is None
            or re.search(r"\bbefore implementation edits\b", attempted,
                         re.IGNORECASE) is None):
        raise DirectiveError(
            "Attempted operation must state the concrete subagent launch "
            "that was tried before implementation edits")

    raw_failure = values["Raw failure"]
    if (not _has_substantive_payload(
            raw_failure, minimum_alphanumeric=12, minimum_words=3)
            or PLACEHOLDER_BODY_RE.fullmatch(raw_failure) is not None
            or EMBEDDED_PLACEHOLDER_RE.search(raw_failure) is not None
            or re.fullmatch(
                r"(?:error|failed|failure|unavailable|unsupported|"
                r"no support|unknown)(?:[.!])?",
                raw_failure,
                re.IGNORECASE) is not None):
        raise DirectiveError(
            "Raw failure must preserve the concrete runtime failure, not a "
            "summary or placeholder")
    return {
        "mode": "capability-unavailable",
        "capability_checked": capability,
        "attempted_operation": attempted,
        "raw_failure": raw_failure,
        "subagents": [],
    }


def _subagents_not_required(lines, context="section 'Parallel work plan'"):
    """Parse one Architect-owned decision that a helper adds no value.

    The decision is exactly one heading plus one Reason row, and the
    reason must actually argue the point: it must say why a separate
    helper would produce no independent, non-overlapping work.

    Arguments:
      lines   = the plan's remaining rows.
      context = section name for error messages.

    Returns:
      The parsed not-required plan mapping.

    Raises:
      DirectiveError: for a malformed heading, row, or reason.
    """
    if len(lines) != 2 or lines[0] != SUBAGENTS_NOT_REQUIRED_HEADING:
        raise DirectiveError(
            context + " must contain exactly the heading "
            + SUBAGENTS_NOT_REQUIRED_HEADING + " and one Reason row")
    match = SUBAGENTS_NOT_REQUIRED_REASON_RE.fullmatch(lines[1])
    if match is None:
        raise DirectiveError(
            context + " must place the Architect's explanation in one exact "
            "- Reason: row")
    reason = match.group(1).strip()
    _parallel_payload("Reason", reason, context=context)
    no_independent_result = re.search(
        r"\b(?:no|not|without|cannot)\b[^.!?\n]{0,120}\bindependent\b",
        reason, re.IGNORECASE)
    if (no_independent_result is None
            or re.search(
                r"\b(?:repeat|duplicate|overlap|same|indivisible|separate|"
                r"non-overlapping)\w*\b", reason, re.IGNORECASE) is None):
        raise DirectiveError(
            context + " Reason must explain why a separate helper would not "
            "produce independent, non-overlapping work or evidence")
    return {
        "mode": "not-required",
        "reason": reason,
        "subagents": [],
    }


def _require_parallel_subagent_plan(body):
    """Parse the Architect's explicit helper decision.

    The Architect either provides bounded helper work or explains why a
    separate helper would only duplicate this ticket.  The Implementer cannot
    choose or rewrite that decision.
    """
    structural = _binding_markdown_text(text=body)
    lines = [line.strip() for line in structural.split("\n") if line.strip()]
    if not lines:
        raise DirectiveError(
            "section 'Parallel work plan' requires a structured subagent "
            "plan")
    if lines[0].startswith("- Capability checked:"):
        return _capability_exception(lines=lines)
    if lines[0] == SUBAGENTS_NOT_REQUIRED_HEADING:
        return _subagents_not_required(lines=lines)
    if lines[0] != SUBAGENTS_REQUIRED_HEADING:
        raise DirectiveError(
            "section 'Parallel work plan' must start with "
            + SUBAGENTS_REQUIRED_HEADING + " or "
            + SUBAGENTS_NOT_REQUIRED_HEADING)
    if len(lines) < 2 or lines[1] != PARALLEL_LAUNCH_ROW:
        raise DirectiveError(
            "section 'Parallel work plan' with required subagents must place "
            "the exact launch row after its heading: " + PARALLEL_LAUNCH_ROW)

    subagents = []
    names = set()
    edit_owners = {}
    index = 2
    expected_fields = (
        "Mode", "Ownership", "Task", "Return", "Acceptance", "Stop")
    while index < len(lines) and lines[index] != "#### Integrator":
        heading = SUBAGENT_HEADING_RE.fullmatch(lines[index])
        if heading is None:
            raise DirectiveError(
                "each Parallel work plan task must start with the exact "
                "heading #### Subagent `descriptive-name`")
        name = heading.group(1)
        if name in {"agent", "helper", "subagent", "worker", "integrator"}:
            raise DirectiveError(
                "Subagent name '" + name
                + "' is generic; use a bounded responsibility name")
        if name in names:
            raise DirectiveError(
                "Parallel work plan repeats Subagent name '" + name + "'")
        names.add(name)
        index += 1

        fields = {}
        for expected_field in expected_fields:
            if index >= len(lines):
                raise DirectiveError(
                    "Subagent '" + name + "' is missing field '"
                    + expected_field + "'")
            field_match = SUBAGENT_FIELD_RE.fullmatch(lines[index])
            if (field_match is None
                    or field_match.group(1) != expected_field):
                raise DirectiveError(
                    "Subagent '" + name + "' requires the exact ordered "
                    "fields Mode, Ownership, Task, Return, Acceptance, Stop; "
                    "expected '" + expected_field + "'")
            fields[expected_field] = field_match.group(2).strip()
            index += 1

        mode_value = fields["Mode"]
        if mode_value not in ("`read-only`", "`edit`"):
            raise DirectiveError(
                "Subagent '" + name
                + "' Mode must be exactly `read-only` or `edit`")
        mode = mode_value[1:-1]
        ownership = _subagent_ownership(
            value=fields["Ownership"], mode=mode, name=name)
        for field in ("Task", "Return", "Acceptance", "Stop"):
            _parallel_payload(field, fields[field])

        if re.match(
                r"^(?:Run|Read|Compare|Write|Add|Edit|Change|Implement|"
                r"Create|Remove|Replace|Measure|Reproduce|Verify|Audit|"
                r"Inspect|Trace|Review|Execute)\b",
                fields["Task"], re.IGNORECASE) is None:
            raise DirectiveError(
                "Subagent '" + name + "' Task must begin with one concrete "
                "imperative action, not an open investigation")
        if not ownership and re.search(r"`[^`\n]+`", fields["Task"]) is None:
            raise DirectiveError(
                "read-only Subagent '" + name + "' with no file Ownership "
                "must name its exact command or artifact in backticks")
        if re.search(
                r"\b(?:command|output|diff|patch|file|assertion|report|"
                r"result|evidence|line|list|commit|artifact)\w*\b",
                fields["Return"], re.IGNORECASE) is None:
            raise DirectiveError(
                "Subagent '" + name + "' Return must name the exact "
                "artifact or evidence to send back")
        if re.search(
                r"\b(?:exit|contains?|equals?|matches?|passes?|fails?|"
                r"created|unchanged|reports?|shows?|diff|output|file|"
                r"assertion)\w*\b",
                fields["Acceptance"], re.IGNORECASE) is None:
            raise DirectiveError(
                "Subagent '" + name + "' Acceptance must name an "
                "observable result")
        if (re.match(r"^(?:Stop|Block)\b", fields["Stop"], re.IGNORECASE)
                is None
                or re.search(r"\b(?:if|when|unless)\b",
                             fields["Stop"], re.IGNORECASE) is None):
            raise DirectiveError(
                "Subagent '" + name + "' Stop must state a concrete "
                "blocker condition beginning with Stop or Block")

        if mode == "edit":
            if not ownership:
                raise DirectiveError(
                    "edit Subagent '" + name
                    + "' requires at least one exact Ownership locator")
            for path in sorted({owner[0] for owner in ownership}):
                if path in edit_owners:
                    raise DirectiveError(
                        "edit Ownership file `" + path
                        + "` is duplicated by Subagents '"
                        + edit_owners[path] + "' and '" + name + "'")
                edit_owners[path] = name
        subagents.append({
            "name": name,
            "mode": mode,
            "ownership": [path + "::" + symbol
                          for path, symbol in ownership],
            "task": fields["Task"],
            "return": fields["Return"],
            "acceptance": fields["Acceptance"],
            "stop": fields["Stop"],
        })

    if not subagents:
        raise DirectiveError(
            "section 'Parallel work plan' must define at least one named "
            "Subagent before the Integrator")
    if index >= len(lines) or lines[index] != "#### Integrator":
        raise DirectiveError(
            "section 'Parallel work plan' requires the exact heading "
            "#### Integrator after all Subagent blocks")
    index += 1
    integrator = {}
    for expected_field in ("Integration", "Final validation"):
        if index >= len(lines):
            raise DirectiveError(
                "Integrator is missing field '" + expected_field + "'")
        match = INTEGRATOR_FIELD_RE.fullmatch(lines[index])
        if match is None or match.group(1) != expected_field:
            raise DirectiveError(
                "Integrator requires exactly the ordered fields Integration "
                "and Final validation; expected '" + expected_field + "'")
        integrator[expected_field] = match.group(2).strip()
        _parallel_payload(expected_field, integrator[expected_field])
        index += 1
    if index != len(lines):
        raise DirectiveError(
            "section 'Parallel work plan' contains extra text after the "
            "Integrator contract")
    if (re.search(r"\b(?:each|every|all)\b",
                  integrator["Integration"], re.IGNORECASE) is None
            or re.search(r"\b(?:subagent|return)\w*\b",
                         integrator["Integration"], re.IGNORECASE) is None):
        raise DirectiveError(
            "Integrator Integration must explain how every named subagent "
            "return is combined")
    if (re.search(r"`[^`\n]+`", integrator["Final validation"]) is None
            or re.search(
                r"\b(?:exit|pass|fail|output|reports?|equals?|matches?)\w*\b",
                integrator["Final validation"], re.IGNORECASE) is None):
        raise DirectiveError(
            "Integrator Final validation must name an exact backticked "
            "command and its observable required result")
    return {
        "mode": "subagents",
        "launch": "required before implementation edits",
        "subagents": subagents,
        "integrator": {
            "integration": integrator["Integration"],
            "final_validation": integrator["Final validation"],
        },
    }


def _require_integrator_validation_command(parallel_work_plan,
                                           validation_commands_body):
    """Bind final integration to one command already named by Architect.

    Arguments:
      parallel_work_plan       = the parsed plan; only the subagents
                                 mode is checked.
      validation_commands_body = the directive's Validation commands
                                 section text.

    Raises:
      DirectiveError: when the Integrator's final validation does not
        repeat exactly one command from that section.
    """
    if parallel_work_plan.get("mode") != "subagents":
        return
    final_validation = parallel_work_plan["integrator"]["final_validation"]
    named = re.findall(r"`([^`\n]+)`", final_validation)
    available = set()
    for _tag, command_lines in _command_blocks(
            body=validation_commands_body):
        available.update(_logical_shell_commands(lines=command_lines))
    if len(named) != 1 or named[0] not in available:
        raise DirectiveError(
            "Integrator Final validation must repeat exactly one command "
            "from the directive's Validation commands section")


def _subagent_evidence_fields(lines, index, name):
    """Read the three ordered fields, including wrapped prose lines.

    Arguments:
      lines = the evidence rows.
      index = the row where this return's fields begin.
      name  = the subagent's name, for error messages.

    Returns:
      ``(fields, index)``: the mapping of the three field values —
      continuation lines joined into their field — and the row after
      the last one consumed.

    Raises:
      DirectiveError: for a missing or misordered field.
    """
    fields = {}
    for expected in ("Returned artifact", "Acceptance", "Evidence"):
        if index >= len(lines):
            raise DirectiveError(
                "Subagent return '" + name + "' is missing field '"
                + expected + "'")
        match = SUBAGENT_EVIDENCE_FIELD_RE.fullmatch(lines[index])
        if match is None or match.group(1) != expected:
            raise DirectiveError(
                "Subagent return '" + name + "' requires exactly the "
                "ordered fields Returned artifact, Acceptance, Evidence")
        parts = [match.group(2).strip()]
        index += 1
        while index < len(lines):
            if (SUBAGENT_EVIDENCE_FIELD_RE.fullmatch(lines[index]) is not None
                    or SUBAGENT_EVIDENCE_HEADING_RE.fullmatch(
                        lines[index]) is not None
                    or lines[index].startswith(("- ", "#### "))):
                break
            parts.append(lines[index])
            index += 1
        fields[expected] = " ".join(parts)
    return fields, index


def _visible_subagent_evidence(text):
    """Expose only canonical evidence rows nested in a Markdown list.

    A return written as a nested list item indents its rows; up to
    four leading spaces are stripped from rows that are canonical
    evidence headings or fields so the validator sees them, while
    deeper or foreign rows stay untouched.

    Arguments:
      text = the evidence text.

    Returns:
      The text with shallow canonical rows unindented.
    """
    rows = []
    for line in text.split("\n"):
        stripped = line.lstrip(" ")
        if (len(line) - len(stripped) <= 4
                and (SUBAGENT_EVIDENCE_HEADING_RE.fullmatch(stripped)
                     is not None
                     or SUBAGENT_EVIDENCE_FIELD_RE.fullmatch(stripped)
                     is not None)):
            line = stripped
        rows.append(line)
    return "\n".join(rows)


def validate_implementer_subagent_evidence(parallel_work_plan, text):
    """Validate an Implementer return against its parsed Architect plan.

    ``parallel_work_plan`` is the dictionary returned under that key by
    :func:`validate_directive_text`.  For an ordinary plan, ``text`` contains
    one block per planned name in the same order::

        #### Subagent return `name`
        - Returned artifact: exact artifact description
        - Acceptance: `pass`
        - Evidence: exact command, output, path, or other observable evidence

    ``blocked`` is also accepted as an Acceptance value so a truthful
    checkpoint can return without pretending that a subagent passed.
    """
    if not isinstance(parallel_work_plan, dict):
        raise DirectiveError(
            "parallel_work_plan must be the parsed Architect plan")
    if not isinstance(text, str):
        raise DirectiveError("subagent evidence text must be a native string")
    structural = _binding_markdown_text(
        text=_visible_subagent_evidence(text=text))
    lines = [line.strip() for line in structural.split("\n") if line.strip()]

    if parallel_work_plan.get("mode") == "capability-unavailable":
        evidence = _capability_exception(lines=lines)
        for key in ("capability_checked", "attempted_operation", "raw_failure"):
            if evidence[key] != parallel_work_plan.get(key):
                raise DirectiveError(
                    "Implementer capability evidence does not match the "
                    "Architect's Parallel work plan field '" + key + "'")
        # This mode is reachable only after the Architect directive embeds
        # the exact prior blocked cycle/SHA checkpoint.  Repeating the same
        # capability failure is therefore the mechanically authorized
        # no-helper fallback, not another request to loop back to Architect.
        evidence["completion_ready"] = True
        return evidence
    if parallel_work_plan.get("mode") == "not-required":
        evidence = _subagents_not_required(
            lines=lines, context="IMPLEMENTER_HANDOFF subagent evidence")
        if evidence["reason"] != parallel_work_plan.get("reason"):
            raise DirectiveError(
                "IMPLEMENTER_HANDOFF must repeat the Architect's Subagents "
                "not required Reason exactly")
        evidence["completion_ready"] = True
        return evidence
    if parallel_work_plan.get("mode") != "subagents":
        raise DirectiveError("parallel_work_plan has an unknown mode")

    planned = [row.get("name")
               for row in parallel_work_plan.get("subagents", [])]
    if (not planned or any(not isinstance(name, str) for name in planned)
            or len(set(planned)) != len(planned)):
        raise DirectiveError(
            "parallel_work_plan has invalid planned Subagent names")
    returned = []
    records = []
    index = 0
    for planned_name in planned:
        if index >= len(lines):
            raise DirectiveError(
                "IMPLEMENTER_HANDOFF lacks Subagent return '"
                + planned_name + "'")
        heading = SUBAGENT_EVIDENCE_HEADING_RE.fullmatch(lines[index])
        if heading is None:
            raise DirectiveError(
                "subagent evidence requires #### Subagent return `name`, "
                "optionally as one Markdown list item")
        name = heading.group(1) or heading.group(2)
        returned.append(name)
        index += 1
        fields, index = _subagent_evidence_fields(
            lines=lines, index=index, name=name)
        if fields["Acceptance"] not in (
                "`pass`", "`blocked`", "pass", "blocked"):
            raise DirectiveError(
                "Subagent return '" + name
                + "' Acceptance must be exactly `pass` or `blocked`")
        evidence_context = "IMPLEMENTER_HANDOFF subagent evidence"
        _parallel_payload(
            "Returned artifact", fields["Returned artifact"],
            context=evidence_context)
        _parallel_payload(
            "Evidence", fields["Evidence"], context=evidence_context)
        records.append({
            "name": name,
            "returned_artifact": fields["Returned artifact"],
            "acceptance": fields["Acceptance"].strip("`"),
            "evidence": fields["Evidence"],
        })
    if index != len(lines):
        raise DirectiveError(
            "IMPLEMENTER_HANDOFF subagent evidence contains an unplanned or "
            "duplicate return")
    if returned != planned:
        raise DirectiveError(
            "IMPLEMENTER_HANDOFF Subagent returns must match the planned "
            "names and order; planned " + repr(planned)
            + ", returned " + repr(returned))
    return {
        "mode": "subagents",
        "returns": records,
        "completion_ready": all(
            record["acceptance"] == "pass" for record in records),
    }


def extract_implementer_subagent_evidence(handoff_text):
    """Extract the one bounded subagent-evidence region from a handoff.

    The handoff must carry exactly one handoff heading, then exactly
    one Subagent work marker row, and the evidence runs from that
    marker to the closing field; duplicates or reordering are
    refused so evidence can never be ambiguous.

    Arguments:
      handoff_text = the full IMPLEMENTER_HANDOFF text.

    Returns:
      The evidence region's text.

    Raises:
      DirectiveError: for a missing, doubled, or reordered heading,
        marker, or closing field.
    """
    if not isinstance(handoff_text, str):
        raise DirectiveError("IMPLEMENTER_HANDOFF must be a native string")
    normalized = handoff_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    headers = [index for index, line in enumerate(lines)
               if line.startswith("### IMPLEMENTER_HANDOFF:")]
    if len(headers) != 1:
        raise DirectiveError(
            "IMPLEMENTER_HANDOFF requires exactly one handoff heading")
    markers = [index for index, line in enumerate(lines)
               if line == IMPLEMENTER_SUBAGENT_EVIDENCE_MARKER]
    if len(markers) != 1:
        raise DirectiveError(
            "IMPLEMENTER_HANDOFF requires exactly one marker row "
            + IMPLEMENTER_SUBAGENT_EVIDENCE_MARKER)
    header_index = headers[0]
    marker_index = markers[0]
    if header_index >= marker_index:
        raise DirectiveError(
            "IMPLEMENTER_HANDOFF heading and Subagent work marker are "
            "reordered")
    fields = []
    for index, line in enumerate(lines):
        match = IMPLEMENTER_HANDOFF_FIELD_RE.fullmatch(line)
        if match is not None:
            fields.append((index, match.group(1)))
    subagent_fields = [index for index, name in fields
                       if name == "Subagent work"]
    if subagent_fields != [marker_index]:
        raise DirectiveError(
            "IMPLEMENTER_HANDOFF requires one exact marker-only Subagent "
            "work field")
    end_fields = [index for index, name in fields
                  if name == IMPLEMENTER_SUBAGENT_EVIDENCE_END_FIELD]
    if len(end_fields) != 1:
        raise DirectiveError(
            "IMPLEMENTER_HANDOFF requires exactly one "
            "- **Blockers/findings:** field after subagent evidence")
    end_index = end_fields[0]
    if marker_index >= end_index:
        raise DirectiveError(
            "IMPLEMENTER_HANDOFF subagent marker and Blockers/findings "
            "field are reordered")
    following_fields = [(index, name) for index, name in fields
                        if index > marker_index]
    if not following_fields or following_fields[0] != (
            end_index, IMPLEMENTER_SUBAGENT_EVIDENCE_END_FIELD):
        next_name = ("none" if not following_fields
                     else following_fields[0][1])
        raise DirectiveError(
            "Blockers/findings must be the next exact handoff field after "
            "subagent evidence; found " + next_name)
    fragment = "\n".join(lines[marker_index + 1:end_index]).strip("\n")
    if not fragment.strip():
        raise DirectiveError(
            "IMPLEMENTER_HANDOFF Subagent work marker has no structured "
            "evidence")
    return fragment


def extract_blocked_implementer_capability_evidence(handoff_text):
    """Return the exact launch failure saved by one blocked handoff.

    This parser deliberately does not use the later Architect plan. The prior
    handoff is the evidence source: it must contain one or more unique,
    well-formed Subagent returns, at least one blocked return, and then the
    three canonical capability-failure rows as the final rows of its bounded
    Subagent-work fragment.
    """
    evidence = extract_implementer_subagent_evidence(
        handoff_text=handoff_text)
    visible_evidence = _visible_subagent_evidence(text=evidence)
    has_blocked_return = any(
        line.strip() in ("- Acceptance: `blocked`", "- Acceptance: blocked")
        for line in _binding_markdown_text(
            text=visible_evidence).split("\n"))
    structural = _binding_markdown_text(text=visible_evidence)
    lines = [line.strip() for line in structural.split("\n")
             if line.strip()]
    records = []
    names = set()
    index = 0
    while index < len(lines):
        heading = SUBAGENT_EVIDENCE_HEADING_RE.fullmatch(lines[index])
        if heading is None:
            break
        name = heading.group(1) or heading.group(2)
        if name in names:
            raise DirectiveError(
                "blocked IMPLEMENTER_HANDOFF repeats Subagent return '"
                + name + "'")
        names.add(name)
        index += 1
        fields, index = _subagent_evidence_fields(
            lines=lines, index=index, name=name)
        if fields["Acceptance"] not in (
                "`pass`", "`blocked`", "pass", "blocked"):
            raise DirectiveError(
                "Subagent return '" + name
                + "' Acceptance must be exactly `pass` or `blocked`")
        _parallel_payload(
            "Returned artifact", fields["Returned artifact"],
            context="blocked IMPLEMENTER_HANDOFF subagent evidence")
        _parallel_payload(
            "Evidence", fields["Evidence"],
            context="blocked IMPLEMENTER_HANDOFF subagent evidence")
        records.append({
            "name": name,
            "returned_artifact": fields["Returned artifact"],
            "acceptance": fields["Acceptance"].strip("`"),
            "evidence": fields["Evidence"],
        })
    if not records:
        raise DirectiveError(
            "blocked IMPLEMENTER_HANDOFF requires at least one well-formed "
            "Subagent return before its capability failure rows")
    if not any(record["acceptance"] == "blocked" for record in records):
        raise DirectiveError(
            "blocked IMPLEMENTER_HANDOFF capability evidence requires at "
            "least one blocked Subagent return")
    capability = _capability_exception(lines=lines[index:])
    return {
        "returns": records,
        "capability_checked": capability["capability_checked"],
        "attempted_operation": capability["attempted_operation"],
        "raw_failure": capability["raw_failure"],
    }


def validate_implementer_handoff_subagent_evidence(parallel_work_plan,
                                                    handoff_text):
    """Extract and validate one full handoff against its Architect plan.

    Ordinary per-subagent returns are tried first; when the plan
    demanded subagents and the returns do not complete it, the
    blocked-capability form is accepted instead — but only with its
    exact rows, so a handoff cannot half-claim both shapes.

    Arguments:
      parallel_work_plan = the parsed Architect plan.
      handoff_text       = the full IMPLEMENTER_HANDOFF text.

    Returns:
      The validated evidence mapping from whichever form matched.

    Raises:
      DirectiveError: when neither form validates.
    """
    evidence = extract_implementer_subagent_evidence(
        handoff_text=handoff_text)
    has_blocked_return = any(
        line.strip() == "- Acceptance: `blocked`"
        for line in _binding_markdown_text(
            text=_visible_subagent_evidence(text=evidence)).split("\n"))
    ordinary = None
    ordinary_error = None
    try:
        ordinary = validate_implementer_subagent_evidence(
            parallel_work_plan=parallel_work_plan, text=evidence)
    except DirectiveError as exc:
        ordinary_error = exc
    if ordinary is not None and ordinary.get("completion_ready"):
        return ordinary
    if parallel_work_plan.get("mode") != "subagents":
        if ordinary_error is not None:
            raise ordinary_error
        return ordinary
    try:
        blocked = extract_blocked_implementer_capability_evidence(
            handoff_text=handoff_text)
    except DirectiveError as capability_error:
        if ordinary is not None or has_blocked_return:
            raise DirectiveError(
                "a blocked IMPLEMENTER_HANDOFF must end its Subagent work "
                "with exact Capability checked, Attempted operation, and "
                "Raw failure rows") from capability_error
        raise ordinary_error
    planned = [row.get("name")
               for row in parallel_work_plan.get("subagents", [])]
    returned = [record["name"] for record in blocked["returns"]]
    if returned != planned:
        raise DirectiveError(
            "blocked IMPLEMENTER_HANDOFF Subagent returns must match "
            "the planned names and order; planned " + repr(planned)
            + ", returned " + repr(returned))
    return {
        "mode": "subagents",
        "returns": blocked["returns"],
        "completion_ready": False,
        "capability_failure": {
            key: blocked[key] for key in (
                "capability_checked", "attempted_operation", "raw_failure")
        },
    }


def _require_redteam_severity_assessment(body, expected_user_severity=None):
    """Return the five ordered discovery-assessment fields.

    Each row — User severity setting, Red Team severity, Likelihood,
    Likelihood evidence, Meets user setting — must appear exactly
    once, in order, in canonical form, with substantive likelihood
    evidence; the recorded user setting must match the dispatched
    one when the caller knows it.

    Arguments:
      body                   = the Finding and evidence section text.
      expected_user_severity = the dispatched severity, or ``None``.

    Returns:
      Mapping from row name to its value.

    Raises:
      DirectiveError: naming the first malformed or mismatched row.
    """
    structural = _binding_markdown_text(text=body)
    lines = [line for line in structural.split("\n") if line.strip()]
    positions = []
    values = {}
    for name, pattern in REDTEAM_SEVERITY_ROWS:
        reserved = re.compile(
            r"^- " + re.escape(name) + r"[ \t]*:", re.IGNORECASE)
        candidates = [(index, line) for index, line in enumerate(lines)
                      if reserved.match(line) is not None]
        matches = [(index, pattern.fullmatch(line))
                   for index, line in candidates
                   if pattern.fullmatch(line) is not None]
        if len(candidates) != 1 or len(matches) != 1:
            raise DirectiveError(
                "section 'Finding and evidence' requires exactly one "
                "canonical '- " + name + ": ...' row")
        index, match = matches[0]
        positions.append(index)
        values[name] = match.group(1).strip()
    if positions != sorted(positions):
        raise DirectiveError(
            "section 'Finding and evidence' severity rows are out of order")
    if not _has_substantive_payload(
            values["Likelihood evidence"],
            minimum_alphanumeric=16,
            minimum_words=3):
        raise DirectiveError(
            "section 'Finding and evidence' needs substantive likelihood "
            "evidence")
    user_setting = values["User severity setting"]
    if (expected_user_severity is not None
            and user_setting != expected_user_severity):
        raise DirectiveError(
            "section 'Finding and evidence' User severity setting "
            + user_setting + " does not match the run-time --severity value "
            + expected_user_severity)
    redteam_severity = values["Red Team severity"]
    likelihood = values["Likelihood"]
    qualifies = (
        user_setting == "low"
        or redteam_severity == "high"
        or (user_setting == "medium"
            and redteam_severity == "medium"
            and likelihood == "probable"))
    expected_meets = "yes" if qualifies else "no"
    if values["Meets user setting"] != expected_meets:
        raise DirectiveError(
            "section 'Finding and evidence' says Meets user setting "
            + values["Meets user setting"] + " but its severity and "
            "likelihood require " + expected_meets)
    return {
        "user_setting": user_setting,
        "redteam_severity": redteam_severity,
        "likelihood": likelihood,
        "likelihood_evidence": values["Likelihood evidence"],
        "meets_user_setting": values["Meets user setting"],
    }


def _command_blocks(body):
    """Return ``(tag, lines)`` for closed shell fences in one section.

    Only fences whose info tag is a recognized command tag count, and
    only when they close; an unclosed fence contributes nothing.

    Arguments:
      body = the section text.

    Returns:
      List of ``(tag, lines)`` pairs, one per closed command fence.
    """
    blocks = []
    character = None
    width = 0
    info = ""
    lines = []
    in_comment = False
    for line in body.split("\n"):
        if character is not None:
            if _is_fence_close(line=line, character=character, width=width):
                if info in COMMAND_FENCE_TAGS:
                    blocks.append((info, tuple(lines)))
                character = None
                width = 0
                info = ""
                lines = []
            else:
                lines.append(line)
            continue
        visible, in_comment = _visible_without_comments(
            line=line, in_comment=in_comment)
        opening = _fence_opening(line=visible)
        if opening is not None:
            character, width, info = opening
            lines = []
    return blocks


def _logical_shell_commands(lines):
    """Return simple visible commands, joining backslash continuations.

    Arguments:
      lines = the fence's raw lines.

    Returns:
      The logical commands, with blank and comment lines dropped and
      backslash-continued lines joined into one command.
    """
    commands = []
    pending = ""
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            pending += line[:-1].rstrip() + " "
            continue
        commands.append((pending + line).strip())
        pending = ""
    if pending:
        commands.append(pending.strip())
    return commands


def _has_shell_control_flow(commands):
    """Return whether a command block can hide a guard behind shell flow.

    Shell control keywords (if, for, case, function bodies, grouping
    braces) could make a required guard command conditional, so their
    presence disqualifies the block from carrying guards.

    Arguments:
      commands = the logical commands.

    Returns:
      True when any command opens shell control flow.
    """
    control_re = re.compile(
        r"^(?:if|then|elif|else|fi|for|select|while|until|do|done|case|"
        r"esac|function)\b|^(?:\(|\)|\{|\})|"
        r"^[A-Za-z_][A-Za-z0-9_]*[ \t]*\([ \t]*\)[ \t]*\{")
    return any(control_re.search(command.strip()) is not None
               for command in commands)


def _parse_ticket_change_guard(command, authoritative_tool=None):
    """Parse one direct, literal ticket-size guard command.

    Shell variables and compound shell expressions are deliberately refused.
    The packet must preserve the exact worktree, starting commit, and limit so
    a lower-capability Implementer can copy the command without inference.
    """
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return None
    if not tokens:
        return None

    script_index = None
    script_token = None
    manual_absolute = str(CONTRACT_REPO_ROOT / TICKET_CHANGE_GUARD)
    allowed_tools = {
        TICKET_CHANGE_GUARD,
        "./" + TICKET_CHANGE_GUARD,
        manual_absolute,
    }
    if authoritative_tool is not None:
        allowed_tools = {authoritative_tool}
    if (len(tokens) >= 2
            and Path(tokens[0]).name in ("python", "python3")
            and tokens[1] in allowed_tools):
        script_index = 1
        script_token = tokens[1]
    if script_index is None:
        return None

    arguments = tokens[script_index + 1:]
    if len(arguments) != 6:
        return None
    values = {}
    for index in range(0, len(arguments), 2):
        option = arguments[index]
        value = arguments[index + 1]
        if option not in ("--repo", "--base", "--max") or option in values:
            return None
        values[option] = value
    if set(values) != {"--repo", "--base", "--max"}:
        return None

    repo = values["--repo"]
    if (not PurePosixPath(repo).is_absolute()
            or ".." in PurePosixPath(repo).parts
            or re.search(r"[\x00-\x1f\x7f$`*?\[\]{};|&<>()\\]",
                         repo) is not None):
        return None
    base = values["--base"]
    if re.fullmatch(r"[0-9a-fA-F]{40}", base) is None:
        return None
    maximum = values["--max"]
    if re.fullmatch(r"[0-9]+", maximum) is None:
        return None
    return {
        "tool": script_token,
        "repo": repo,
        "base": base.lower(),
        "max": int(maximum),
        "max_literal": maximum,
    }


def _require_ticket_change_guard(
        bodies, expected_max, execution_checkout=None):
    """Require a literal positive-limit guard command and acceptance check.

    A ticket with a positive character budget must carry exactly one
    direct, literal size-guard command in its Validation commands —
    outside any shell control flow, with the exact worktree, base,
    and limit spelled out — so a lower-capability Implementer can
    copy it without inference.

    Arguments:
      bodies             = the parsed section bodies.
      expected_max       = the ticket's character limit; zero means
                           unlimited and skips the requirement.
      execution_checkout = the parsed checkout, when the guard must
                           name it.

    Returns:
      The parsed guard invocation, or ``None`` for an unlimited
      ticket.

    Raises:
      DirectiveError: for a missing, duplicated, indirect, or
        mismatched guard command.
    """
    if expected_max == 0:
        return None

    authoritative_tool = os.environ.get("MAILBOX_TICKET_CHANGE_GUARD")
    if authoritative_tool is not None and (
            not os.path.isabs(authoritative_tool)
            or Path(authoritative_tool).name != "ticket_change_guard.py"):
        raise DirectiveError(
            "MAILBOX_TICKET_CHANGE_GUARD must name the authoritative "
            "absolute ticket_change_guard.py path")

    parsed = []
    saw_guard_text = False
    for _tag, lines in _command_blocks(body=bodies["Validation commands"]):
        commands = _logical_shell_commands(lines=lines)
        if _has_shell_control_flow(commands=commands):
            continue
        for command in commands:
            if TICKET_CHANGE_GUARD in command:
                saw_guard_text = True
            invocation = _parse_ticket_change_guard(
                command=command,
                authoritative_tool=authoritative_tool)
            if invocation is not None:
                parsed.append(invocation)
    if len(parsed) != 1:
        detail = ("one direct literal command" if saw_guard_text
                  else "a direct literal command")
        raise DirectiveError(
            "positive Character-change budget requires " + detail + " in "
            "'Validation commands': python3 "
            + (authoritative_tool or TICKET_CHANGE_GUARD)
            + " --repo ABSOLUTE_WORKTREE --base FULL_40_HEX_COMMIT "
              "--max " + str(expected_max))

    invocation = parsed[0]
    if (invocation["max"] != expected_max
            or invocation["max_literal"] != str(expected_max)):
        raise DirectiveError(
            "ticket_change_guard.py command --max "
            + invocation["max_literal"]
            + " does not match the exact run-time --max "
            + str(expected_max))
    if execution_checkout is not None:
        if invocation["repo"] != execution_checkout["Worktree"]:
            raise DirectiveError(
                "ticket_change_guard.py command --repo does not match the "
                "Execution checkout Worktree")
        if invocation["base"] != execution_checkout["Base"]:
            raise DirectiveError(
                "ticket_change_guard.py command --base does not match the "
                "Execution checkout Base")

    checklist = _binding_markdown_text(text=bodies["Acceptance checklist"])
    conditions = CHECKBOX_RE.findall(checklist)

    def is_positive_condition(condition):
        """Accept only a within-limit condition with no negation."""
        return (
            TICKET_CHANGE_GUARD in condition
            and re.search(r"\bwithin[ -]limit\b", condition,
                          flags=re.IGNORECASE) is not None
            and re.search(
                r"\b(?:not|never|without|fail(?:s|ed)?|refus(?:e|es|ed)|"
                r"over[ -]?limit)\b",
                condition,
                flags=re.IGNORECASE) is None)

    if not any(is_positive_condition(condition) for condition in conditions):
        raise DirectiveError(
            "positive Character-change budget requires an Acceptance "
            "checklist condition that ticket_change_guard.py reports "
            "'within limit' for the exact candidate")
    return {key: value for key, value in invocation.items()
            if key != "max_literal"}


def _require_commands(body):
    """Require one syntactically valid fence led by a real command.

    Each shell fence's script must pass the shell's own syntax check,
    and at least one fence must lead with a real command — not a
    comment, placeholder, assignment, or bare shell builtin.

    Arguments:
      body = the Validation commands section text.

    Raises:
      DirectiveError: when no fence carries a valid runnable command.
    """
    shell_by_tag = {
        "bash": "bash",
        "sh": "sh",
        "shell": "sh",
        "zsh": "zsh",
    }
    shell_builtins = {
        ".", "cd", "command", "export", "false", "printf", "read",
        "set", "source", "test", "true", "type", "ulimit", "umask",
        "unset", "wait", "[",
    }
    assignment_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
    for tag, block in _command_blocks(body=body):
        candidates = []
        for line in block:
            candidate = re.sub(r"^[ \t]*(?:[-*]|\d+[.)])[ \t]+", "", line)
            candidate = candidate.strip()
            if not candidate or candidate.startswith("#"):
                continue
            if (HTML_ENTITY_RE.search(candidate) is not None
                    or any(unicodedata.category(character) in ("Cc", "Cf")
                           for character in candidate)
                    or not any(character.isalnum()
                               for character in candidate)):
                continue
            if EMBEDDED_PLACEHOLDER_RE.search(candidate) is not None:
                continue
            if candidate in ("...", "[command]", "<command>"):
                continue
            candidates.append(candidate)
        if not candidates:
            continue

        script = "\n".join(block) + "\n"
        shell = shutil.which(shell_by_tag[tag])
        if shell is None:
            continue
        try:
            syntax = subprocess.run(
                [shell, "-n"],
                input=script,
                text=True,
                capture_output=True,
                timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if syntax.returncode != 0:
            continue

        try:
            tokens = shlex.split(candidates[0], posix=True)
        except ValueError:
            continue
        while tokens and assignment_re.fullmatch(tokens[0]) is not None:
            tokens.pop(0)
        if not tokens:
            continue
        command = tokens[0]
        if "/" in command:
            command_path = command
            if not os.path.isabs(command_path):
                command_path = os.path.join(
                    str(CONTRACT_REPO_ROOT), command_path)
            command_resolves = (
                os.path.isfile(command_path)
                and os.access(command_path, os.X_OK))
        else:
            command_resolves = (
                command in shell_builtins
                or shutil.which(command) is not None)
        if command_resolves:
            return
    raise DirectiveError(
        "section 'Validation commands' requires a closed bash/sh/shell/zsh "
        "fence with at least one non-comment, non-placeholder, "
        "syntax-valid, resolvable command")


def validate_directive_text(role, text, expected_max=0,
                            expected_severity=None):
    """Validate one note's Architect or Red Team directive packet.

    Arguments:
      role = ``architect`` for a binding implementation directive, or
             ``redteam`` for an advisory repair directive.
      text = complete Markdown note text.
      expected_max = exact nonnegative run-time character-change limit. None
                     accepts the canonical limit saved by an interrupted
                     Architect directive during restart recovery.
      expected_severity = exact user setting for a Red Team discovery. When
                          omitted, only the row's internal consistency is
                          checked.

    Returns:
      A parsed dictionary containing the role, packet title, character-change
      budget, and the Architect's execution checkout and role plan when
      applicable.
      ``DirectiveError`` is raised when the packet is incomplete or its
      character-change limit differs from ``expected_max``.
    """
    if role not in PACKET_TITLES:
        raise DirectiveError("unknown directive role: " + repr(role))
    if not isinstance(text, str):
        raise DirectiveError("directive text must be a native string")
    if text.startswith("\ufeff"):
        text = text[1:]
    if "\x00" in text:
        raise DirectiveError("directive text contains a NUL byte")
    for character in text:
        if (character not in ("\n", "\r", "\t")
                and unicodedata.category(character) in (
                    "Cc", "Cf", "Zl", "Zp")):
            raise DirectiveError(
                "directive text contains a non-Markdown control, format, or "
                "line-separator character")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _without_leading_frontmatter(text=text)
    _reject_list_item_fences(text=text)
    _reject_display_math_blocks(text=text)
    _reject_setext_headings(text=text)
    _reject_raw_html_blocks(text=text)

    title = PACKET_TITLES[role]
    required = REQUIRED_SECTIONS[role]
    bodies = _section_bodies(text=text, title=title, required=required)
    _require_substance(bodies=bodies)

    file_rows = _require_locator(
        bodies=bodies, heading="Files and symbols")
    result = {"role": role, "packet_title": title}
    result["character_change_budget"] = _require_character_change_budget(
        body=bodies["Character-change budget"],
        expected_max=expected_max)
    effective_max = result["character_change_budget"]["limit"]
    execution_checkout = None
    if role == "architect":
        execution_checkout = _require_execution_checkout(
            body=bodies["Execution checkout"])
        result["execution_checkout"] = execution_checkout
        result["role_plan"] = _require_architect_role_plan(
            body=bodies["Role plan"])
        result["parallel_work_plan"] = _require_parallel_subagent_plan(
            body=bodies["Parallel work plan"])
        test_rows = _require_locator(bodies=bodies, heading="Tests to write")
        result["allowed_paths"] = sorted({
            path for path, _symbol in file_rows + test_rows})
        tool_paths = [path for path in result["allowed_paths"]
                      if path.startswith("ai/tools/")]
        if tool_paths:
            raise DirectiveError(
                "ai/tools/ is external-maintainer-only; keep the backlog "
                "ticket Open and do not send it to the Implementer")
        evidence_body = _require_evidence_destination(
            text=text, packet_title=title)
        if result["parallel_work_plan"]["mode"] == "capability-unavailable":
            result["capability_checkpoint"] = (
                _require_prior_capability_checkpoint(
                evidence_body=evidence_body,
                parallel_work_plan=result["parallel_work_plan"]))
    else:
        _require_locator(bodies=bodies, heading="Regression test")
        result["discovery_severity_assessment"] = (
            _require_redteam_severity_assessment(
                body=bodies["Finding and evidence"],
                expected_user_severity=expected_severity))

    step_heading = ("Ordered implementation steps" if role == "architect"
                    else "Ordered repair steps")
    structural_steps = _binding_markdown_text(text=bodies[step_heading])
    numbered = NUMBERED_STEP_RE.search(structural_steps)
    all_numbered = list(NUMBERED_ANY_STEP_RE.finditer(structural_steps))
    if (numbered is None or not all_numbered
            or any(not _has_substantive_payload(match.group(1))
                   for match in all_numbered)):
        raise DirectiveError(
            "section '" + step_heading + "' must start a numbered procedure "
            "with a visible alphanumeric instruction")
    structural_checks = _binding_markdown_text(
        text=bodies["Acceptance checklist"])
    checkboxes = list(CHECKBOX_RE.finditer(structural_checks))
    if (not checkboxes
            or any(not _has_substantive_payload(match.group(1))
                   for match in checkboxes)):
        raise DirectiveError(
            "section 'Acceptance checklist' must contain a Markdown checkbox "
            "with a visible alphanumeric condition")
    _require_commands(body=bodies["Validation commands"])
    if role == "architect":
        _require_integrator_validation_command(
            parallel_work_plan=result["parallel_work_plan"],
            validation_commands_body=bodies["Validation commands"])
    guard = _require_ticket_change_guard(
        bodies=bodies,
        expected_max=effective_max,
        execution_checkout=execution_checkout)
    if guard is not None:
        result["ticket_change_guard"] = guard
    return result


def validate_directive_file(role, path, expected_max=0,
                            expected_severity=None):
    """Read and validate one bounded UTF-8 directive note.

    The file must sit under the authoritative shared-notes folder
    when that environment binding exists, this validator must be the
    authoritative contract copy, and the note must be one bounded
    regular UTF-8 file before role validation runs.

    Arguments:
      role              = ``"architect"``, ``"implementer"``, or
                          ``"redteam"``.
      path              = the directive note file.
      expected_max      = the ticket's character limit; zero is
                          unlimited.
      expected_severity = dispatched discovery severity, Red Team
                          role only.

    Returns:
      The role validator's parsed result.

    Raises:
      DirectiveError: for an unauthoritative copy, an unsafe path or
        read, or any failed packet rule.
    """
    if role == "redteam":
        expected_severity = resolve_discovery_severity(
            cli_value=expected_severity)
    elif expected_severity is not None:
        raise DirectiveError(
            "--severity is valid only for a Red Team directive")
    authoritative_contract = os.environ.get("MAILBOX_HANDOFF_CONTRACT")
    if authoritative_contract is not None:
        actual_contract = os.path.realpath(os.path.abspath(__file__))
        if (not os.path.isabs(authoritative_contract)
                or os.path.abspath(authoritative_contract) != actual_contract
                or os.path.realpath(authoritative_contract)
                != actual_contract):
            raise DirectiveError(
                "this validator is not the authoritative absolute "
                "MAILBOX_HANDOFF_CONTRACT program")
    note = Path(path)
    shared_notes = os.environ.get("MAILBOX_SHARED_NOTES")
    if shared_notes is not None:
        shared_path = Path(shared_notes)
        if (not shared_path.is_absolute()
                or ".." in shared_path.parts
                or os.path.realpath(str(shared_path))
                != os.path.abspath(str(shared_path))):
            raise DirectiveError(
                "MAILBOX_SHARED_NOTES must name one authoritative absolute "
                "notes directory without a redirected path")
        if not note.is_absolute() or ".." in note.parts:
            raise DirectiveError(
                "a mailbox directive note must use its absolute path below "
                "MAILBOX_SHARED_NOTES, not a relative or parent-traversing "
                "path")
        note_absolute = os.path.abspath(str(note))
        try:
            inside_shared_notes = (
                os.path.commonpath((note_absolute, str(shared_path)))
                == str(shared_path))
        except ValueError:
            inside_shared_notes = False
        if not inside_shared_notes:
            raise DirectiveError(
                "mailbox directive note is outside MAILBOX_SHARED_NOTES")
        if os.path.realpath(note_absolute) != note_absolute:
            raise DirectiveError(
                "mailbox directive note uses a redirected path instead of "
                "the authoritative MAILBOX_SHARED_NOTES file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags = flags | os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags = flags | os.O_NONBLOCK
    try:
        descriptor = os.open(str(note), flags)
    except OSError as exc:
        raise DirectiveError("cannot open directive note safely: " + str(exc))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DirectiveError(
                "directive note is not a regular file: " + str(note))
        if before.st_size > MAX_NOTE_BYTES:
            raise DirectiveError(
                "directive note exceeds " + str(MAX_NOTE_BYTES) + " bytes")
        chunks = []
        remaining = MAX_NOTE_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining = remaining - len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise DirectiveError("cannot read directive note safely: " + str(exc))
    finally:
        os.close(descriptor)

    identity_before = (before.st_dev, before.st_ino, before.st_size,
                       before.st_mtime_ns, before.st_ctime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size,
                      after.st_mtime_ns, after.st_ctime_ns)
    if identity_before != identity_after or len(payload) != before.st_size:
        raise DirectiveError("directive note changed while it was being read")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise DirectiveError("cannot read directive note as UTF-8: " + str(exc))
    return validate_directive_text(
        role=role, text=text, expected_max=expected_max,
        expected_severity=expected_severity)


def nonnegative_character_limit(value):
    """Parse one command-line decimal character limit.

    Arguments:
      value = the command-line text; only plain ASCII digits are
              accepted.

    Returns:
      The parsed limit.

    Raises:
      argparse.ArgumentTypeError: for anything else.
    """
    if re.fullmatch(r"[0-9]+", value) is None:
        raise argparse.ArgumentTypeError(
            "--max must be a nonnegative decimal character count")
    return int(value)


def resolve_character_limit(cli_value, environment_value=None):
    """Bind a CLI limit to the mailbox environment without silent fallback.

    Arguments:
      cli_value         = the ``--max`` value, or ``None`` when
                          omitted.
      environment_value = mailbox environment text, or ``None`` to
                          read the real environment.

    Returns:
      The limit to enforce; zero means unlimited.

    Raises:
      DirectiveError: for a malformed value or a disagreement between
        the command line and the environment.
    """
    if cli_value is not None and (
            isinstance(cli_value, bool)
            or not isinstance(cli_value, int)
            or cli_value < 0):
        raise DirectiveError(
            "command-line --max must be a nonnegative integer")
    if environment_value is None:
        environment_value = os.environ.get("MAILBOX_MAX_CHARACTERS")

    environment_limit = None
    if environment_value is not None:
        if (not isinstance(environment_value, str)
                or re.fullmatch(r"[0-9]+", environment_value) is None):
            raise DirectiveError(
                "MAILBOX_MAX_CHARACTERS must contain only ASCII decimal "
                "digits")
        environment_limit = int(environment_value)

    if cli_value is None:
        return 0 if environment_limit is None else environment_limit
    if environment_limit is not None and cli_value != environment_limit:
        raise DirectiveError(
            "command-line --max " + str(cli_value)
            + " does not match MAILBOX_MAX_CHARACTERS "
            + str(environment_limit))
    return cli_value


def resolve_discovery_severity(cli_value, environment_value=None):
    """Bind a Red Team severity value to the mailbox run setting.

    Arguments:
      cli_value         = the ``--severity`` value, or ``None``.
      environment_value = the run's inherited severity, or ``None``
                          to read the real environment.

    Returns:
      The severity to enforce: the explicit value when it matches any
      inherited one, otherwise the inherited or default severity.

    Raises:
      DirectiveError: for an invalid value or a disagreement with the
        environment.
    """
    if (cli_value is not None
            and cli_value not in DISCOVERY_SEVERITIES):
        raise DirectiveError(
            "command-line --severity must be high, medium, or low")
    if environment_value is None:
        environment_value = os.environ.get(
            DISCOVERY_SEVERITY_ENVIRONMENT)
    if (environment_value is not None
            and environment_value not in DISCOVERY_SEVERITIES):
        raise DirectiveError(
            DISCOVERY_SEVERITY_ENVIRONMENT
            + " must be exactly high, medium, or low")
    if cli_value is None:
        return (DEFAULT_DISCOVERY_SEVERITY
                if environment_value is None else environment_value)
    if environment_value is not None and cli_value != environment_value:
        raise DirectiveError(
            "command-line --severity " + cli_value + " does not match "
            + DISCOVERY_SEVERITY_ENVIRONMENT + " " + environment_value)
    return cli_value


def parse_args(argv=None):
    """Parse the read-only directive-validation command line.

    Arguments:
      argv = argument list, or ``None`` for the process arguments.

    Returns:
      The parsed options: the role, the note path, and the optional
      ``--max`` and ``--severity`` bindings.
    """
    parser = argparse.ArgumentParser(
        description="Validate a complete Architect or Red Team directive")
    parser.add_argument("role", choices=tuple(PACKET_TITLES))
    parser.add_argument("note", help="Markdown ticket note containing the packet")
    parser.add_argument(
        "--max", metavar="characters",
        type=nonnegative_character_limit, default=None,
        help="character-change limit that the directive must match; when "
             "omitted, use MAILBOX_MAX_CHARACTERS if present, otherwise 0")
    parser.add_argument(
        "--severity", choices=DISCOVERY_SEVERITIES, default=None,
        help="user's discovery threshold for a Red Team directive; when "
             "omitted, use MAILBOX_DISCOVERY_SEVERITY if present, "
             "otherwise medium")
    return parser.parse_args(argv)


def main(argv=None):
    """Run the read-only directive validator.

    Arguments:
      argv = argument list, or ``None`` for the process arguments.

    Returns:
      The process exit code: 0 with a VALID line, or 1 with an
      INVALID line naming the exact failed rule.
    """
    args = parse_args(argv=argv)
    try:
        expected_max = resolve_character_limit(cli_value=args.max)
        if args.severity is not None and args.role != "redteam":
            raise DirectiveError(
                "--severity is valid only for a Red Team directive")
        expected_severity = (
            resolve_discovery_severity(cli_value=args.severity)
            if args.role == "redteam" else None)
        validate_directive_file(
            role=args.role, path=args.note, expected_max=expected_max,
            expected_severity=expected_severity)
    except DirectiveError as exc:
        print(args.role + " directive: INVALID: " + str(exc))
        return 1
    print(args.role + " directive: VALID: " + args.note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
