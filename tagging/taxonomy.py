"""Tag taxonomy for the road-readiness categorization.

Six dimensions, 43 tags total -- identical to the columns in
TO_25-203_All_Datasets_Categorized.xlsx. Each tag carries a natural-language
prompt (SigLIP/CLIP zero-shot works far better with descriptive sentences than
bare tag names). `multi_label` / `rel_thresh` control how many tags a dimension
may emit per image (see tag_images.py for the selection rule).
"""

# dimension -> {tag: prompt}
TAXONOMY = {
    "Operational Scenario": {
        "Intersections": "a forward-facing dashcam photo at a road intersection or junction with cross traffic",
        "Roundabouts": "a forward-facing dashcam photo of a roundabout or traffic circle",
        "Ramps / Merges": "a forward-facing dashcam photo of a highway on-ramp, off-ramp, or merging lane",
        "At-Grade Rail Crossings": "a forward-facing dashcam photo of a railroad level crossing with tracks across the road",
        "Tolling Stations": "a forward-facing dashcam photo of a toll booth or tolling plaza",
        "Work Zones": "a forward-facing dashcam photo of a road work zone with cones, barriers, or construction",
        "Managed Lanes": "a forward-facing dashcam photo of a managed lane such as an HOV or express lane",
        "General Roadway Segments": "a forward-facing dashcam photo of an ordinary open road segment with no special features",
    },
    "Roadway Context & Facility Type": {
        "Freeway/Expressway": "a forward-facing dashcam photo of a freeway or expressway with high-speed traffic",
        "Arterial Roadway": "a forward-facing dashcam photo of a major multi-lane arterial road",
        "Collector Roadway": "a forward-facing dashcam photo of a smaller collector road connecting neighborhoods",
        "Local Street / Road": "a forward-facing dashcam photo of a local residential street",
        "Urban": "a forward-facing dashcam photo of a road in a dense urban city environment with buildings",
        "Rural": "a forward-facing dashcam photo of a road in a rural countryside area with open land",
    },
    "Roadway Surface Type": {
        "Paved - bituminous / asphalt": "a forward-facing dashcam photo of a road paved with dark asphalt",
        "Paved - non-bituminous / concrete": "a forward-facing dashcam photo of a road paved with light gray concrete",
        "Unpaved": "a forward-facing dashcam photo of an unpaved dirt or gravel road",
        "Undetermined": "a forward-facing dashcam photo where the road surface material cannot be determined",
    },
    "Pavement Marking Type & Configuration": {
        "Solid Line": "a forward-facing dashcam photo of a road with solid continuous painted lane lines",
        "Dashed Line": "a forward-facing dashcam photo of a road with dashed or broken lane lines",
        "Double Line": "a forward-facing dashcam photo of a road with double painted center lines",
        "Edge Line": "a forward-facing dashcam photo of a road with painted edge lines along the shoulder",
        "Curved Line": "a forward-facing dashcam photo of a road with curving lane markings",
        "Gore Markings": "a forward-facing dashcam photo of gore-area chevron markings where lanes split",
        "Crosswalks": "a forward-facing dashcam photo of a painted pedestrian crosswalk on the road",
        "Arrows": "a forward-facing dashcam photo of directional turn arrows painted on the road",
        "Unmarked Pavement": "a forward-facing dashcam photo of a road with no painted lane markings",
    },
    "Observed Marking Visibility": {
        "Clearly Visible": "a forward-facing dashcam photo where the lane markings are bright and clearly visible",
        "Faded / Worn / Not Visible": "a forward-facing dashcam photo where the lane markings are faded, worn, or barely visible",
        "Low Contrast": "a forward-facing dashcam photo where the lane markings have low contrast against the road",
        "Partially Missing": "a forward-facing dashcam photo where the lane markings are partially missing or broken up",
        "Occluded": "a forward-facing dashcam photo where the lane markings are occluded by vehicles or objects",
        "Not Applicable": "a forward-facing dashcam photo with no lane markings present at all",
    },
    "Lighting & Weather": {
        "Daylight": "a forward-facing dashcam photo taken during clear daylight",
        "Nighttime": "a forward-facing dashcam photo taken at night in the dark",
        "Dusk / Dawn": "a forward-facing dashcam photo taken at dusk or dawn with dim light",
        "Shadow": "a forward-facing dashcam photo with strong shadows cast across the road",
        "Glare": "a forward-facing dashcam photo with bright sun glare or oncoming headlight glare",
        "Rain": "a forward-facing dashcam photo taken in the rain",
        "Fog": "a forward-facing dashcam photo taken in foggy or hazy conditions",
        "Snow": "a forward-facing dashcam photo with snow on the road or snow falling",
        "Wet Pavement": "a forward-facing dashcam photo of wet reflective road pavement",
        "Icy Road": "a forward-facing dashcam photo of an icy road surface",
    },
}

# Per-dimension selection config.
#   multi_label: may emit more than one tag.
#   rel_thresh : besides the top-1 (always emitted), also emit any tag whose
#                within-dimension softmax probability >= rel_thresh.
DIMENSION_CONFIG = {
    "Operational Scenario": {"multi_label": True, "rel_thresh": 0.35},
    "Roadway Context & Facility Type": {"multi_label": True, "rel_thresh": 0.30},
    "Roadway Surface Type": {"multi_label": True, "rel_thresh": 0.40},
    "Pavement Marking Type & Configuration": {"multi_label": True, "rel_thresh": 0.20},
    "Observed Marking Visibility": {"multi_label": True, "rel_thresh": 0.40},
    "Lighting & Weather": {"multi_label": True, "rel_thresh": 0.25},
}

# Flat helpers.
ALL_TAGS = [(dim, tag, prompt)
            for dim, tags in TAXONOMY.items()
            for tag, prompt in tags.items()]
TAG_TO_DIM = {tag: dim for dim, tags in TAXONOMY.items() for tag in tags}
