#!/usr/bin/env python3
"""Anatomic site taxonomy for pitcher IL placements (prompt B.3).

Extends the bat-tracking `site` / `site_fine` / `laterality` scheme
(`Bat-Tracking + Injuries/scripts/build_stints_2026-09-08.py`) to the pitcher
sites this study needs, rather than inventing a second taxonomy.  Every site
name that exists in the bat-tracking scheme keeps its meaning; the elbow region
is split into the four structures the prompt names (`elbow_other`,
`forearm_flexor`, `ulnar_nerve`, `ucl_graft`) and the shoulder girdle into
`shoulder`, `lat_teres`, `pec`, `tos`.

One category is added beyond the prompt's list: `arm_other`, for placements that
name the throwing arm without localising it ("arm surgery", "right arm injury",
upper-extremity nerve and vascular problems, "dead arm").  Assigning these to
`unspecified` would drop real throwing-arm placements out of the arm family;
assigning them to a named structure would invent a diagnosis.  Their count is
reported in Supplementary Table S2.

The rules below were written from the regex and then corrected by hand-reading
every arm-site placement and every `unspecified` placement in the cohort
(prompt B.3); `HAND_CORRECTIONS` lists the readings the regex could not make.
"""
import re

RULES = [
    # --- specific arm structures, most specific first ---------------------
    ("tos",               r"thoracic outlet|first rib resection|subclavian|effort thrombosis"),
    ("ulnar_nerve",       r"ulnar ner|ulnar neur|ulnar transposition|cubital tunnel|"
                          r"ulnar.{0,12}(?:irritat|entrap)"),
    ("ucl_graft",         r"ulnar (?:collateral|colateral)|\bucl\b|ucl construction|tommy john|"
                          r"elbow ligament|ligament.{0,25}(?:e[lb]bow|eblow)|graft|"
                          r"(?:e[lb]bow|eblow) sprain|sprained.{0,15}(?:e[lb]bow|eblow)|"
                          r"ligament reconstruction"),
    ("forearm_flexor",    r"forearm|flexor|pronator"),
    ("lat_teres",         r"latissimus|\blat\b(?!eral)|lat strain|teres major|\bteras\b"),
    ("pec",               r"pectoral|\bpec\b"),
    # hip precedes shoulder so "hip labral tear" is not swept up by the
    # shoulder labrum keyword (bat-tracking taxonomy note)
    ("hip_groin",         r"\bhip\b|groin|adductor|pelvis|abductor|pubalgia|sports hernia|"
                          r"inguinal|glute"),
    ("shoulder",          r"shoulder|shouder|should(?= strain| tendin| inflam)|rotator cuff|"
                          r"labrum|labral|\bslap\b|a[./ ]?c[. ]joint|acromio|subscap|"
                          r"infraspinatus|supraspinatus|glenohumeral|scapul|capsul|deltoid|"
                          r"rhomboid|trapezius|\btrap\b|\bbicep", ),
    ("elbow_other",       r"e[lb]bow|eblow|olecranon|triceps|loose bod|epicondyl"),
    # --- non-arm --------------------------------------------------------
    ("hand_wrist_finger", r"wrist|hand(?!string)|finger|thumb|hamm?ate|metacarp|carpal|"
                          r"knuckle|\bpinky\b|distal radius|\bnail\b"),
    ("core_oblique",      r"obliqu|oliqu|intercostal|\bribs?\b|rib ?cage|abdomin|core muscle|"
                          r"core surgery|costochondr|sternum|serratus|torso|trunk|flank|"
                          r"(?:left|right)\s+side\b|\bside (?:strain|tightness|discomfort)|"
                          r"\bchest\b"),
    ("back",              r"\bback\b|lumbar|spine|spinal|\bdisc\b|neck|cervical|sacro|sciatic"),
    ("hamstring_quad",    r"hamstring|quad|quadriceps|\bthigh\b"),
    ("knee",              r"knee|\bacl\b|\bmcl\b|\bpcl\b|meniscus|patell"),
    ("calf_achilles",     r"calf|achilles|soleus|gastroc|\bshin\b|fibula|tibia|lower (?:right |left )?leg"),
    ("ankle_foot",        r"ankle|\bfoot\b|\btoe\b|plantar|heel|peroneal|metatars|lisfranc"),
    ("arm_other",         r"\barm\b|upper extremity|axillar|armpit|radial ner|median ner|"
                          r"brachial|blood clot"),
    ("illness_other",     r"illness|covid|virus|viral|infection|infected|personal|anxiety|"
                          r"migraine|vertigo|appendec|appendicitis|diverticul|bronchitis|"
                          r"influenza|\bflu\b|kidney|gastro|food poisoning|dental|tooth|"
                          r"\bskin\b|blister|dehydrat|cardiac|heartbeat|arrhythmia|"
                          r"pericard|heart\b|\blung\b|cancer|lymphoma|aneurysm|abscess|"
                          r"concussion|\bhead\b|skull|facial|\bface\b|\bjaw\b|\bnose\b|"
                          r"\beye\b|orbital|raynaud|non-baseball"),
]

ARM_SITES = {"shoulder", "elbow_other", "forearm_flexor", "ulnar_nerve",
             "ucl_graft", "lat_teres", "pec", "tos", "arm_other"}
# The throwing-arm family for Aim 1, as the prompt defines it, plus arm_other.
THROWING_ARM_FAMILY = {"shoulder", "elbow_other", "forearm_flexor", "ulnar_nerve",
                       "ucl_graft", "lat_teres", "arm_other"}
# pec and TOS are reported separately: they are shoulder-girdle but neither is
# one of the six sub-sites the prompt prespecifies.
SHOULDER_GIRDLE_OTHER = {"pec", "tos"}
NONARM_FAMILY = {"core_oblique", "back", "hip_groin", "hamstring_quad", "knee",
                 "calf_achilles", "ankle_foot", "hand_wrist_finger",
                 "illness_other", "unspecified"}
ALL_SITES = [s for s, _ in RULES] + ["unspecified"]

# Hand readings the regex cannot make, keyed on the lower-cased diagnosis text.
# Each entry is (site, note).  Applied after the regex and before the
# adjudicated-cohort overrides; every application is logged.
HAND_CORRECTIONS = {
    "cutaneous lymphoma in right triceps.":
        ("illness_other", "oncologic, not a musculoskeletal elbow injury"),
    "right arm fatigue.":
        ("arm_other", "arm fatigue is not an illness; unlocalised throwing-arm"),
    "streaa reaction in right arm.":
        ("arm_other", "typo for stress reaction; site within the arm not stated"),
    "surgery recovery.":
        ("unspecified", "no site named"),
    "non-baseball related injury.":
        ("illness_other", "no site named, explicitly non-baseball"),
    "non-baseball related medical matter.":
        ("illness_other", "no site named, explicitly non-baseball"),
    "right upper extremity blood clot.":
        ("arm_other", "vascular; not thoracic outlet unless stated"),
    "blood clot left armpit.":
        ("arm_other", "vascular; not thoracic outlet unless stated"),
    "axillary abscess on right side.":
        ("illness_other", "abscess, not a musculoskeletal arm injury"),
    "raynaud\u2019s syndrome.":
        ("illness_other", "systemic vasospastic disorder"),
    "shingles.":
        ("illness_other", "herpes zoster"),
    "right elbow capsule strain.":
        ("elbow_other", "capsule keyword is a shoulder term; the text names the elbow"),
}

# site_fine splits that matter for the prespecified contrasts
FINE = {
    "ucl_graft": [("ucl_named", r"ulnar (?:collateral|colateral)|\bucl\b|tommy john|graft|"
                                r"ligament"),
                  ("elbow_sprain_generic", r".")],
    "forearm_flexor": [("flexor_pronator", r"flexor|pronator"),
                       ("forearm_other", r".")],
    "shoulder": [("biceps_unspecified", r"\bbicep(?!.{0,20}(?:shoulder|tendin))"),
                 ("scapulothoracic", r"rhomboid|trapezius|\btrap\b|scapul"),
                 ("rotator_cuff_labrum", r"rotator cuff|labrum|labral|\bslap\b|"
                                         r"supraspinatus|infraspinatus|subscap"),
                 ("shoulder_other", r".")],
}

SIDE_RE = re.compile(r"\b(left|right)\b", re.I)
# "RHP", "LHP", "right-hander" are HANDEDNESS, not the side of the injury
# (memory rule "Laterality is not handedness").
HAND_RE = re.compile(r"\b(?:RHP|LHP|RHS|LHS|RH|LH|"
                     r"(?:right|left)[- ]hand(?:ed|er|ers)?|"
                     r"(?:right|left)[- ]hander)\b", re.I)

_COMPILED = [(s, re.compile(p, re.I)) for s, p in RULES]
_FINE = {k: [(n, re.compile(p, re.I)) for n, p in v] for k, v in FINE.items()}


def classify_site(txt):
    t = (txt or "").lower()
    for site, pat in _COMPILED:
        if pat.search(t):
            return site
    return "unspecified"


def classify_fine(site, txt):
    t = (txt or "").lower()
    for name, pat in _FINE.get(site, []):
        if pat.search(t):
            return name
    return site


def read_side(dx):
    """Side of the injury from the diagnosis text only, handedness stripped."""
    t = HAND_RE.sub(" ", str(dx or ""))
    m = SIDE_RE.search(t)
    return m.group(1).lower() if m else None


# A placement whose diagnosis explicitly names the structure the regex assigned
# is protected from the adjudicated-cohort override: the cohort tables are the
# truth for their own site, but they are joined on date, and a date join cannot
# override a text that names a different structure outright ("Right UCL
# injury.", "Left hamstring strain.").  Every blocked override is logged as a
# conflict rather than silently dropped.
PROTECTED = {
    "ucl_graft":        r"ulnar (?:collateral|colateral)|\bucl\b|tommy john|elbow ligament",
    "lat_teres":        r"latissimus|\blat\b(?!eral)|teres major",
    "pec":              r"pectoral",
    "ulnar_nerve":      r"ulnar ner|ulnar neur",
    "shoulder":         r"rotator cuff|labrum|labral|\bslap\b",
    "knee":             r"meniscus|\bacl\b|\bmcl\b",
    "core_oblique":     r"obliqu|intercostal",
    "hamstring_quad":   r"hamstring|quadriceps",
    "hip_groin":        r"groin|\bhip\b",
    "forearm_flexor":   r"forearm|flexor|pronator",
    "tos":              r"thoracic outlet",
}
_PROT = {k: re.compile(v, re.I) for k, v in PROTECTED.items()}


def is_protected(site, dx):
    p = _PROT.get(site)
    return bool(p and p.search(str(dx or "")))
