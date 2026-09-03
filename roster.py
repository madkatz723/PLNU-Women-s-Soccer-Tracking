"""Canonical roster: the single source of truth that joins CMJ names,
GPS/Catapult names, and player photos.

The CMJ sheet uses legal first names ("Nicole Ross", 'Hayley "Lehua"
Hanawahine') while the Catapult export uses the name on the pod, which is
usually the nickname ("Nikki Ross", "Lehua Hanawahine"). Fuzzy matching is
not safe here -- "Nikki"/"Nicole" scores 0.36 on the first name, low enough
that any threshold loose enough to catch it would start pairing up unrelated
athletes. In a fatigue tool a silent mis-join shows a coach the wrong
athlete's readiness, so every alias is listed explicitly below and anything
unrecognised is surfaced rather than guessed at.

Add a row here when a player joins; add an alias when a file spells someone a
new way. `suggest_alias()` proposes the likely canonical name for an unknown
spelling, but a human still pastes it into ALIASES.
"""

import os
import re
import unicodedata

IMAGE_DIR = os.path.join(os.path.dirname(__file__), "assets", "player_images")

# canonical name -> (extra spellings seen in the wild, photo filename or None)
# The canonical name is the CMJ-sheet spelling, since that sheet carries the
# season history. Aliases only need to list spellings that differ from it.
ROSTER = {
    "Molly Bachman":              ([], "MollyB_pic.webp"),
    "Jenna Bisset":               ([], "JennaB_pic.webp"),
    "Hannah Boelscher":           ([], "HannahB_pic.webp"),
    "Abby Buys":                  ([], "AbbyB_pic.webp"),
    "Kaleia Coughlin":            ([], "KaleiaC_pic.webp"),
    "Shaylen Greff":              (["Shay Greff"], "ShaylenG_pic.webp"),
    'Hayley "Lehua" Hanawahine':  (["Lehua Hanawahine", "Hayley Hanawahine"], "LehuaH_pic.webp"),
    "Ava Heil":                   ([], "AvaH_pic.webp"),
    "Kylee Jerome":               ([], "KyleeJ_pic.webp"),
    "Riley Johnson":              ([], "RileyJ_pic.webp"),
    "Lila Jones":                 ([], "LilaJ_pic.webp"),
    "Madison Khan":               (["Maddie Khan"], "MadisonK_pic.webp"),
    "Gianna Masinter":            ([], "GiannaM_pic.webp"),
    "Kody McKinney":              ([], "KodyM_pic.webp"),
    "Emma Naftzger":              ([], "EmmaN_pic.webp"),
    "Grace Nelson":               ([], "GraceN_pic.webp"),
    "Nicole Ross":                (["Nikki Ross"], "NikkiRoss_pic.webp"),
    "Madelyn Saruwatari":         (["Maddy Saruwatari"], "MaddyS_pic.webp"),
    "Juliet Thrapp":              ([], "JulietT_pic.webp"),
    "Priya Torres":               ([], "PriyaT_pic.webp"),
    "Leah Uezato":                ([], "LeahU_pic.webp"),
    "Jade Vacheck":               (["Jade Vachek"], "JadeV_pic.webp"),
    "Abby Wright":                ([], "AbbyW_pic.webp"),
    # On the CMJ sheet but no GPS pod data so far this season.
    "Tessa Anastasi":             ([], "Tessa_pic.webp"),
    "Madilyn Audet":              ([], "MadiA_pic.webp"),
    "Emma Blakely":               ([], "EmmaB_pic.webp"),
    "Beatrice Levi":              (["Bea Levi"], "Bea_pic.webp"),
    "Cameron Simmons":            (["Cam Simmons"], "CameronS_pic.webp"),
}


def normalize(name):
    """Fold a raw name to a comparison key: strip accents, quotes and
    punctuation, lowercase, collapse whitespace. This alone absorbs the
    'Hayley "Lehua"' quoting and any stray double spaces or trailing tabs
    that come out of Excel."""
    if name is None:
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"[\"'`.,]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _build_lookup():
    lookup = {}
    for canonical, (aliases, _img) in ROSTER.items():
        for variant in [canonical, *aliases]:
            key = normalize(variant)
            if key in lookup and lookup[key] != canonical:
                raise ValueError(
                    f"Roster alias collision: {variant!r} maps to both "
                    f"{lookup[key]!r} and {canonical!r}"
                )
            lookup[key] = canonical
    return lookup


_LOOKUP = _build_lookup()


def resolve(name):
    """Raw name from any file -> canonical roster name, or None if unknown."""
    return _LOOKUP.get(normalize(name))


def canonicalize(series):
    """Map a pandas Series of raw names to canonical names. Unknown names are
    left as-is so they stay visible in the UI instead of silently vanishing."""
    return series.map(lambda n: resolve(n) or n)


def unresolved(names):
    """The raw names in `names` that the roster does not recognise, sorted.
    The dashboard shows these so a new signing or a re-spelled export is
    noticed on the day it appears rather than months later."""
    return sorted({str(n).strip() for n in names if n and resolve(n) is None})


def image_path(name):
    """Absolute path to a player's photo, or None if the roster has no photo
    for them or the file has not been uploaded yet."""
    canonical = resolve(name)
    if canonical is None:
        return None
    filename = ROSTER[canonical][1]
    if not filename:
        return None
    path = os.path.join(IMAGE_DIR, filename)
    return path if os.path.exists(path) else None


PLAYER_IMAGES = {canonical: image_path(canonical) for canonical in ROSTER}


def photo_audit():
    """(players with no usable photo, image files no player claims).

    A missing photo degrades silently to an initials disc, which looks
    deliberate -- so a typo'd filename is invisible until someone notices a
    player who never shows a face. It bites hardest on deploy: the Procfile
    host is Linux and case-sensitive, so `GraceN_Pic.webp` resolves fine on a
    Windows laptop and vanishes in production. Call this after touching the
    roster or the image folder.
    """
    missing = sorted(name for name, path in PLAYER_IMAGES.items() if path is None)
    declared = {filename for _aliases, filename in ROSTER.values() if filename}
    try:
        on_disk = {f for f in os.listdir(IMAGE_DIR) if not f.startswith(".")}
    except OSError:
        on_disk = set()
    return missing, sorted(on_disk - declared)


def suggest_alias(name, cutoff=0.80):
    """QA helper, not used at runtime: given an unrecognised spelling, propose
    the canonical name it most likely belongs to. Surnames are the stable part
    of these files (every real mismatch this season scored >=0.92 on surname
    and as low as 0.36 on first name), so match on surname first and only fall
    back to the whole string. The answer is a suggestion for a human to accept
    into ALIASES, never an automatic join."""
    import difflib

    key = normalize(name)
    if not key:
        return None
    surname = key.split()[-1]

    best, best_score = None, 0.0
    for canonical in ROSTER:
        c_key = normalize(canonical)
        score = difflib.SequenceMatcher(None, surname, c_key.split()[-1]).ratio()
        if score < cutoff:
            score = max(score, difflib.SequenceMatcher(None, key, c_key).ratio())
        if score > best_score:
            best, best_score = canonical, score
    return best if best_score >= cutoff else None
