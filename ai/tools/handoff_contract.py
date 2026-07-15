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


class DirectiveError(ValueError):
    """Raised when a directive note is incomplete or ambiguous."""


def _without_leading_frontmatter(text):
    """Mask one leading YAML frontmatter block while preserving line rows."""
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
    """Remove Markdown HTML comments from one line without losing state."""
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
    """Return ``(character, width, info)`` for one valid opening fence."""
    match = FENCE_OPEN_RE.match(line)
    if match is None:
        return None
    marker = match.group(1)
    suffix = match.group(2)
    if marker[0] == "`" and "`" in suffix:
        return None
    return marker[0], len(marker), suffix.strip().casefold()


def _is_fence_close(line, character, width):
    """Return whether line is a CommonMark-style matching close fence."""
    pattern = r"^[ ]{0,3}" + re.escape(character) + "{" + str(width) + r",}[ \t]*$"
    return re.match(pattern, line) is not None


def _visible_markdown_text(text):
    """Remove HTML comments outside fences while preserving line structure."""
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
    """Return CommonMark indentation columns using four-column tab stops."""
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
    """Mask fenced and indented code examples for prose-structure checks."""
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
    """Return visible non-example prose used to satisfy binding fields."""
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
    """Keep binding packets on canonical ATX headings only."""
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
    """Refuse container-nested fences outside the packet's canonical subset."""
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
    """Refuse GFM display-math containers around binding Markdown rows."""
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
    """Refuse raw HTML block syntax outside code fences and comments."""
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
    """Return ``(line, level, title)`` rows for Markdown ATX headings."""
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
    """Return the line interval below one exact level-two packet heading."""
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
    """Return exact required level-three section bodies in declared order."""
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
    """Require the Architect packet's immediate sibling evidence heading."""
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
    matches = [heading for _line, level, heading in boundary_rows
               if (level == 2
                   and heading.casefold() == expected.casefold())]
    if len(matches) != 1:
        raise DirectiveError(
            "expected exactly one sibling '## " + expected
            + "' heading; found " + str(len(matches)))


def _require_substance(bodies):
    """Refuse empty, placeholder, or explicitly delegated design choices."""
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
    """Return one explicit unruled choice, allowing a named resolver."""
    # Inline-code spans are visible binding prose. Remove their Markdown
    # delimiters, never their content, so ``Use `JSON or YAML` `` cannot hide
    # an unresolved design choice. Canonical ``path::symbol`` code spans are
    # structural locators, not prose alternatives, and may be masked only in
    # the locator sections whose rows are validated separately. Fenced
    # examples were already masked by ``_binding_markdown_text``.
    def inline_code(match):
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
    """Return whether one structured row carries visible executable prose."""
    compact = " ".join(text.split())
    words = re.findall(r"[^\W_]+", compact, flags=re.UNICODE)
    return (
        sum(1 for character in compact if character.isalnum())
        >= minimum_alphanumeric
        and len(words) >= minimum_words)


def _valid_locator_rows(body):
    """Return concrete ``repo/path::symbol`` locator pairs from one body."""
    rows = []
    for match in LOCATOR_RE.finditer(body):
        path = match.group(1).strip()
        symbol = match.group(2).strip()
        normalized = path.replace("\\", "/")
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
        if (not path or not symbol or normalized.startswith("/")
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
    """Require one canonical visible repository locator bullet."""
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


def _require_execution_checkout(body):
    """Require the exact worktree, branch, and base selected by Architect."""
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
            or ".." in worktree_parts):
        raise DirectiveError(
            "Execution checkout Worktree must be one absolute path without "
            "parent traversal")

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


def _command_blocks(body):
    """Return ``(tag, lines)`` for closed shell fences in one section."""
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


def _require_commands(body):
    """Require one syntactically valid fence led by a real command."""
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


def validate_directive_text(role, text):
    """Validate one note's Architect or Red Team directive packet.

    Arguments:
      role = ``architect`` for a binding implementation directive, or
             ``redteam`` for an advisory repair directive.
      text = complete Markdown note text.

    Returns:
      None.  ``DirectiveError`` is raised when the packet is incomplete.
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

    _require_locator(bodies=bodies, heading="Files and symbols")
    result = {"role": role, "packet_title": title}
    if role == "architect":
        result["execution_checkout"] = _require_execution_checkout(
            body=bodies["Execution checkout"])
        _require_locator(bodies=bodies, heading="Tests to write")
        _require_evidence_destination(text=text, packet_title=title)
    else:
        _require_locator(bodies=bodies, heading="Regression test")

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
    return result


def validate_directive_file(role, path):
    """Read and validate one bounded UTF-8 directive note."""
    note = Path(path)
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
    return validate_directive_text(role=role, text=text)


def parse_args(argv=None):
    """Parse the read-only directive-validation command line."""
    parser = argparse.ArgumentParser(
        description="Validate a complete Architect or Red Team directive")
    parser.add_argument("role", choices=tuple(PACKET_TITLES))
    parser.add_argument("note", help="Markdown ticket note containing the packet")
    return parser.parse_args(argv)


def main(argv=None):
    """Run the read-only directive validator."""
    args = parse_args(argv=argv)
    try:
        validate_directive_file(role=args.role, path=args.note)
    except DirectiveError as exc:
        print(args.role + " directive: INVALID: " + str(exc))
        return 1
    print(args.role + " directive: VALID: " + args.note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
