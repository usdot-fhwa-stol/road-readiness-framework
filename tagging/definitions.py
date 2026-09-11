"""Standards-grounded definitions for the 43-tag road-readiness taxonomy.

Same 6 dimensions / 43 tags as taxonomy.py (so tag_codes.py and all downstream
encoding stay valid) -- this module ADDS, per tag, a precise definition with:
  - cue : the positive visual evidence that licenses the tag
  - neg : what it is NOT / when to withhold it (exclusions)
and, per dimension, a `rules` block: the decision logic, how many tags to emit,
and the disambiguation boundaries that were the main source of model
disagreement (esp. faded-but-present vs unmarked).

Sources (see memory/tag-definition-standards.md):
  - MUTCD 11th ed. Part 3 (markings), Parts 6/8 & Ch 2e/2f (scenarios)
  - FHWA Highway Functional Classification 2023 + HPMS Appendix B
  - Maryland MSMT-729 nighttime pavement-marking condition scale
  - FHWA pavement tech briefs (asphalt vs concrete); FHWA lighting handbook (contrast)

Consumed by tag_gateway.py --defs standard to build a stronger prompt. The tag
strings here MUST stay byte-identical to taxonomy.TAXONOMY keys.
"""

# dimension -> {"rules": <decision logic for the whole dimension>,
#               "tags": {tag: {"cue": ..., "neg": ...}}}
DEFINITIONS = {
    "Operational Scenario": {
        "rules": (
            "Identify special road situations the vehicle is AT or IMMINENTLY "
            "ENTERING in THIS frame. Multi-label. Always tag 'General Roadway "
            "Segments'; ADD a special feature only when the vehicle is on/at it or "
            "about to enter it within the near foreground -- NOT for features merely "
            "visible far in the distance. Judge from road geometry, signs, and "
            "markings, not from surroundings alone. When unsure, omit the special tag."
        ),
        "tags": {
            "Intersections": {
                "cue": "the vehicle is AT or immediately approaching a junction the "
                       "driver would act on: a stop bar/limit line just ahead, a "
                       "crosswalk across our lanes at the near corner, a signal head "
                       "governing our approach, or a cross street meeting our road in "
                       "the near foreground",
                "neg": "NOT a junction far down the road, cross traffic in the "
                       "distance, or scattered city lights; NOT a driveway/parking "
                       "entrance; NOT a roundabout (use Roundabouts). If the junction "
                       "is distant or you must infer it, do NOT tag it.",
            },
            "Roundabouts": {
                "cue": "a circular junction with a central island and curved approaches "
                       "where entering traffic yields to circulating traffic",
                "neg": "NOT an ordinary signalized/stop intersection; requires the "
                       "circular geometry or central island to be visible.",
            },
            "Ramps / Merges": {
                "cue": "the vehicle is on or directly at a highway on-ramp, off-ramp, "
                       "or an active merge/diverge point -- a painted gore (chevron "
                       "neutral area) at our lane, a ramp we are on/entering, or a lane "
                       "actively merging into ours here",
                "neg": "NOT an ordinary at-grade intersection turn; NOT an overpass or "
                       "distant interchange merely visible ahead. Require ramp/gore "
                       "geometry at our position, not on the horizon.",
            },
            "At-Grade Rail Crossings": {
                "cue": "railroad tracks crossing the road at grade; an 'X' + 'RR' pavement "
                       "marking, a crossbuck sign, or flashing-light gates",
                "neg": "NOT an overpass/underpass where rail is grade-separated; the "
                       "tracks must cross the driving surface.",
            },
            "Tolling Stations": {
                "cue": "a toll booth or tolling plaza: overhead lane-guide signs, per-lane "
                       "toll gates/canopies, or lane-use control signals above each lane",
                "neg": "NOT ordinary overhead gantry signs; requires toll-collection "
                       "infrastructure (booths/canopy) or explicit toll signage.",
            },
            "Work Zones": {
                "cue": "active road work: traffic cones, drums/barrels, barricades, vertical "
                       "panels, temporary barriers, or construction equipment channelizing traffic",
                "neg": "NOT a single stray cone or a parked work vehicle with no "
                       "channelization; needs a temporary-traffic-control pattern.",
            },
            "Managed Lanes": {
                "cue": "an HOV / express / managed lane: a painted white diamond symbol, the "
                       "word HOV or EXPRESS on the pavement, or a buffer/barrier-separated special lane",
                "neg": "NOT an ordinary lane; requires the diamond marking, HOV/EXPRESS "
                       "text, or explicit managed-lane separation.",
            },
            "General Roadway Segments": {
                "cue": "an ordinary open road segment carrying through traffic with no "
                       "special junction, ramp, toll, work zone, or managed-lane feature",
                "neg": "still emit this even when a special feature is also present, if the "
                       "frame is dominated by ordinary through-road; it is the default/catch-all.",
            },
        },
    },
    "Roadway Context & Facility Type": {
        "rules": (
            "TWO independent axes -- tag ONE functional class AND ONE setting. "
            "(A) Functional class, decided by ACCESS + separation cues visible in "
            "frame: Freeway/Expressway = a DIVIDED highway (opposing traffic separated "
            "by a median or barrier) with NO driveways or at-grade cross streets in "
            "view, traffic entering/leaving via ramps -- tag this whenever the road "
            "LOOKS like a limited-access highway (multiple lanes each way, median/"
            "barrier, guardrails, high-speed feel, overpasses), even if you cannot "
            "prove full access control. Arterial = major through-road that DOES serve "
            "abutting land (driveways, at-grade signals, storefront frontage). Collector "
            "= moderate road gathering local traffic toward arterials. Local Street/Road "
            "= residential/low-speed, direct property access. When a road is clearly a "
            "divided grade-separated highway, prefer Freeway/Expressway over Arterial. "
            "(B) Setting: Urban vs Rural."
        ),
        "tags": {
            "Freeway/Expressway": {
                "cue": "looks like a limited-access highway: multiple lanes each way, "
                       "opposing directions separated by a median or barrier, guardrails, "
                       "overpasses/ramps, high-speed feel, and NO driveways or at-grade "
                       "cross streets in view",
                "neg": "if you can see driveways, at-grade cross streets, traffic "
                       "signals, or storefront access on the road, it is an Arterial. "
                       "Absence of those on a divided high-speed road => Freeway/Expressway.",
            },
            "Arterial Roadway": {
                "cue": "a major multi-lane through-road that still serves abutting land: "
                       "driveways, at-grade signalized intersections, commercial frontage",
                "neg": "NOT grade-separated (that is Freeway); NOT a quiet residential "
                       "street (that is Local).",
            },
            "Collector Roadway": {
                "cue": "a moderate road that gathers traffic from local streets toward "
                       "arterials; fewer lanes and lower speed than an arterial, some access",
                "neg": "hard to confirm from one frame -- prefer Arterial or Local unless "
                       "the collecting role is evident; low-confidence by nature.",
            },
            "Local Street / Road": {
                "cue": "a residential or low-speed street fronting homes/small businesses, "
                       "narrow, direct property access, little through movement",
                "neg": "NOT a wide multi-lane through-road (Arterial); NOT grade-separated.",
            },
            "Urban": {
                "cue": "dense adjacent development, buildings, sidewalks, curbs/gutters, "
                       "on-street parking, pedestrians, frequent access points",
                "neg": "NOT open countryside; pick Rural instead when land is open.",
            },
            "Rural": {
                "cue": "open-country setting: fields, forest, sparse development, minimal "
                       "access points, no sidewalks",
                "neg": "NOT a built-up area with continuous buildings/sidewalks (Urban).",
            },
        },
    },
    "Roadway Surface Type": {
        "rules": (
            "Judge the DRIVING surface material, not the shoulder. DECISIVE TEST: look "
            "for regularly spaced TRANSVERSE joints (evenly spaced lines running across "
            "all lanes, every few meters) -- these indicate jointed CONCRETE slabs and "
            "OVERRIDE color, because aged/dirty/shadowed concrete often looks dark gray "
            "and can be mistaken for asphalt. Only call asphalt when the surface is a "
            "continuous mat with NO repeating transverse slab joints. Tag one primary "
            "type; add a second only if the frame genuinely transitions. Use "
            "'Undetermined' only when the surface truly cannot be read."
        ),
        "tags": {
            "Paved - bituminous / asphalt": {
                "cue": "a CONTINUOUS surface with a granular/aggregate mat texture and NO "
                       "regularly spaced transverse slab joints; usually dark but may be "
                       "gray when aged/faded",
                "neg": "if you see evenly spaced transverse joints crossing the lanes "
                       "(slab panels), it is CONCRETE regardless of how dark it looks. "
                       "Color alone does not make it asphalt.",
            },
            "Paved - non-bituminous / concrete": {
                "cue": "rectangular slabs/panels marked by regularly spaced transverse "
                       "joints (and a longitudinal joint), often with tined/grooved "
                       "texture; commonly lighter gray but can be dark when dirty or shadowed",
                "neg": "a single longitudinal paving seam alone is not enough -- require "
                       "the repeating TRANSVERSE panel joints. Do not reject concrete just "
                       "because the surface looks dark.",
            },
            "Unpaved": {
                "cue": "dirt, soil, or loose gravel/crushed-rock driving surface; dust, ruts, "
                       "no continuous paved mat",
                "neg": "NOT a paved road with a dirty/patched surface; requires genuinely "
                       "unpaved material.",
            },
            "Undetermined": {
                "cue": "the surface material genuinely cannot be determined (heavy darkness, "
                       "snow cover, glare, or full occlusion)",
                "neg": "do NOT use as a lazy default -- only when asphalt/concrete/unpaved "
                       "truly cannot be told apart.",
            },
        },
    },
    "Pavement Marking Type & Configuration": {
        "rules": (
            "List EVERY painted marking type visible ANYWHERE in the frame -- center "
            "lines, lane dividers, shoulder/edge lines, and special markings. A faded or "
            "worn line that is still identifiable AS a line COUNTS as that line type "
            "(its condition is scored separately under Observed Marking Visibility). "
            "'Unmarked Pavement' is a PARTIAL-COVERAGE signal, NOT a mutually exclusive "
            "catch-all: add it WHENEVER a substantial part of the drivable surface has "
            "no markings -- this INCLUDES roads that also have some lines (e.g. a center "
            "line but bare lane edges, a wide unmarked shoulder/parking area, or lanes "
            "that lose their markings). So 'Unmarked Pavement' may co-occur with "
            "'Dashed Line', 'Solid Line', etc. Emit 'Unmarked Pavement' ALONE only when "
            "there is no paint anywhere. Only when the surface is fully and clearly "
            "marked with no meaningful bare area should you omit it."
        ),
        "tags": {
            "Solid Line": {
                "cue": "a continuous unbroken painted line (white or yellow), e.g. a no-pass "
                       "center line or a lane line where crossing is discouraged",
                "neg": "NOT a broken/dashed line (use Dashed Line); a right-edge solid line "
                       "may ALSO be tagged Edge Line.",
            },
            "Dashed Line": {
                "cue": "a broken line of painted segments with gaps -- a permissive lane "
                       "divider or passing-allowed center line",
                "neg": "NOT a continuous line (Solid Line).",
            },
            "Double Line": {
                "cue": "two parallel adjacent longitudinal lines (double solid yellow no-pass, "
                       "solid+broken, or double white)",
                "neg": "NOT a single line; requires two parallel lines close together.",
            },
            "Edge Line": {
                "cue": "a longitudinal line marking the edge of the travel way / shoulder "
                       "(solid white on the right, solid yellow on the left of divided/one-way)",
                "neg": "NOT the center/lane-divider line; specifically the roadway edge.",
            },
            "Curved Line": {
                "cue": "lane/edge markings that visibly curve because the road bends "
                       "(horizontal curve)",
                "neg": "NOT straight markings viewed in perspective; the road itself must curve.",
            },
            "Gore Markings": {
                "cue": "chevron / diagonal neutral-area markings bounded by channelizing "
                       "lines where a ramp splits from or merges to the mainline",
                "neg": "NOT ordinary lane lines; requires the chevron/diagonal neutral area.",
            },
            "Crosswalks": {
                "cue": "white transverse or ladder/continental bars marking a pedestrian "
                       "crossing across the travel lanes",
                "neg": "NOT a stop bar alone; requires the crossing-area markings.",
            },
            "Arrows": {
                "cue": "white directional arrows painted on the lane (through, turn, or "
                       "through-or-turn movement)",
                "neg": "NOT text legends or other symbols; specifically movement arrows.",
            },
            "Unmarked Pavement": {
                "cue": "a substantial part of the drivable surface has NO markings: a "
                       "fully bare road, OR a road with some lines but notable unmarked "
                       "areas (bare lane edges with no edge line, a wide unmarked "
                       "shoulder/parking lane, or lanes whose markings drop out). This "
                       "may co-occur with line tags.",
                "neg": "omit only when the surface is fully and clearly marked with no "
                       "meaningful bare area. Faint-but-identifiable paint is a line "
                       "(tag its type + mark faded under Visibility), not 'unmarked'.",
            },
        },
    },
    "Observed Marking Visibility": {
        "rules": (
            "Rate the CONDITION/readability of the lane markings that are present "
            "(Maryland MSMT-729 style scale). Be CRITICAL: reserve 'Clearly Visible' "
            "for markings that are genuinely bright, crisp, and effortless to follow. "
            "If markings are at all weak, thin, worn, dull, or need any concentration, "
            "tag 'Faded / Worn / Not Visible'. Faded/worn markings are ALMOST ALWAYS "
            "also low in contrast against the pavement, so when you tag 'Faded / Worn / "
            "Not Visible' you should normally ALSO tag 'Low Contrast' unless the paint "
            "is still bright but merely broken. Multiple condition tags routinely "
            "co-apply (Faded + Low Contrast; Low Contrast + Partially Missing). "
            "'Not Applicable' = genuinely NO markings anywhere to assess (a fully bare "
            "road)."
        ),
        "tags": {
            "Clearly Visible": {
                "cue": "markings are genuinely bright, sharp, high-contrast, and "
                       "effortless to follow (MSMT-729 'Good')",
                "neg": "do NOT use for markings that are dull, thin, worn, or need any "
                       "concentration -- those are Faded / Worn / Not Visible. Be strict.",
            },
            "Faded / Worn / Not Visible": {
                "cue": "markings still identifiable as markings but weak, worn, thin, "
                       "dull, or requiring concentration to follow (MSMT-729 "
                       "'Fair'/'Poor'). This is common -- prefer it over 'Clearly "
                       "Visible' whenever there is any degradation.",
                "neg": "NOT when markings are genuinely bright and crisp; NOT when there "
                       "are no markings at all (Not Applicable). Normally pair with "
                       "'Low Contrast'.",
            },
            "Low Contrast": {
                "cue": "the marking has little luminance/color difference from the "
                       "pavement and is hard to distinguish; typically accompanies faded "
                       "or worn paint",
                "neg": "add this alongside 'Faded / Worn / Not Visible' in the usual case "
                       "where worn paint also blends into the road; omit only if faded "
                       "paint is still bright against a dark surface.",
            },
            "Partially Missing": {
                "cue": "the marking exists but has gaps/breaks beyond normal dash pattern -- "
                       "sections eroded or patched away",
                "neg": "NOT normal dashed-line gaps; requires abnormal missing segments.",
            },
            "Occluded": {
                "cue": "markings are present but hidden by vehicles, glare, shadow, snow, "
                       "water, or debris in this frame",
                "neg": "the marking exists but is BLOCKED -- distinct from being worn/faded.",
            },
            "Not Applicable": {
                "cue": "there are genuinely no lane markings present to assess",
                "neg": "pairs with 'Unmarked Pavement'. Do NOT use if any marking (even "
                       "faint) is present -- use a condition tag instead.",
            },
        },
    },
    "Lighting & Weather": {
        "rules": (
            "Tag EXACTLY ONE ambient-light level (Daylight / Nighttime / Dusk-Dawn) PLUS "
            "every additional condition clearly visible. Distinguish causes: dim overall "
            "scene near sunrise/sunset => Dusk / Dawn (not Daylight+Shadow). Localized "
            "dark patches under bright sky => Shadow. Reflective/darkened wet road => "
            "Wet Pavement (and Rain only if precipitation/droplets are actually visible)."
        ),
        "tags": {
            "Daylight": {
                "cue": "clear, evenly bright daytime scene; sky bright, full ambient light",
                "neg": "NOT dim low-sun conditions (Dusk / Dawn); NOT dark (Nighttime).",
            },
            "Nighttime": {
                "cue": "dark scene lit mainly by headlights/streetlights; dark sky",
                "neg": "NOT a dim-but-pre/post-sunset scene (Dusk / Dawn).",
            },
            "Dusk / Dawn": {
                "cue": "low sun near the horizon, warm/dim light, twilight sky -- transitional "
                       "light that is neither full day nor full dark",
                "neg": "NOT a bright daytime scene that merely has shadows.",
            },
            "Shadow": {
                "cue": "distinct localized dark areas cast across the road by trees, "
                       "buildings, overpasses under otherwise bright conditions",
                "neg": "NOT overall dimness of dusk/night; requires a bright scene with "
                       "cast shadows. Do not over-apply to mild dappling.",
            },
            "Glare": {
                "cue": "bright sun or oncoming headlights washing out part of the image / "
                       "reflecting off the windshield or road",
                "neg": "requires a visible bright-source washout, not merely a sunny day.",
            },
            "Rain": {
                "cue": "visible precipitation: raindrops on the windshield/lens, streaks, or "
                       "actively falling rain",
                "neg": "a merely wet road with no visible droplets is Wet Pavement, NOT Rain.",
            },
            "Fog": {
                "cue": "hazy atmosphere reducing distance visibility, muted colors/contrast "
                       "from mist",
                "neg": "NOT lens blur or dirt; requires atmospheric haze.",
            },
            "Snow": {
                "cue": "snow on the road, shoulders, or actively falling snow",
                "neg": "NOT light-gray concrete; requires actual snow cover/precipitation.",
            },
            "Wet Pavement": {
                "cue": "road surface is darkened and reflective from water; specular "
                       "reflections of lights/sky on the pavement",
                "neg": "tag even in daytime; if droplets/precipitation are visible add Rain too.",
            },
            "Icy Road": {
                "cue": "visible ice sheen, packed snow/ice, or frozen surface on the roadway",
                "neg": "NOT merely wet (Wet Pavement); requires ice/frozen appearance.",
            },
        },
    },
}


def validate_against(taxonomy):
    """Ensure DEFINITIONS covers exactly the same dims/tags as taxonomy.TAXONOMY.
    Returns list of mismatch strings (empty = OK)."""
    problems = []
    for dim, tags in taxonomy.items():
        if dim not in DEFINITIONS:
            problems.append(f"missing dimension: {dim}")
            continue
        deftags = set(DEFINITIONS[dim]["tags"])
        taxtags = set(tags)
        for t in taxtags - deftags:
            problems.append(f"[{dim}] tag missing a definition: {t!r}")
        for t in deftags - taxtags:
            problems.append(f"[{dim}] definition has extra tag not in taxonomy: {t!r}")
    for dim in DEFINITIONS:
        if dim not in taxonomy:
            problems.append(f"extra dimension not in taxonomy: {dim}")
    return problems
