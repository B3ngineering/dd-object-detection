# Maps model labels to military-relevant categories
# Each category has an associated threat level and color for bounding boxes

# Custom model classes (from arctic_military.yaml)
MILITARY_TAXONOMY_MAP = {
    # Personnel
    "camouflage_soldier": "Military Personnel",
    "soldier": "Military Personnel",
    "civilian": "Civilian",
    
    # Vehicles
    "military_tank": "Military Vehicle",
    "military_truck": "Military Vehicle",
    "military_vehicle": "Military Vehicle",
    "military_aircraft": "Military Vehicle",
    "military_warship": "Military Vehicle",
    "civilian_vehicle": "Civilian Vehicle",


    # Weapons
    "weapon": "Weapon",
    "military_artillery": "Artillery",

    # Weapons from weapons model
    "Gun": "Weapon",
    "Explosive": "Explosive",
    "Grenade": "Explosive",
    "Knife": "Weapon",

    # Structures
    "trench": "Fortification",
}

# COCO dataset classes mapped to taxonomy
COCO_TAXONOMY_MAP = {
    "person": "Civilian",

    # COCO vehicles should generally be civilian vehicles
    "car": "Civilian Vehicle",
    "truck": "Civilian Vehicle",
    "bus": "Civilian Vehicle",
    "motorcycle": "Civilian Vehicle",
    "bicycle": "Civilian Vehicle",
    "airplane": "Civilian Vehicle",
    "boat": "Civilian Vehicle",
}


def get_taxonomy(label):
    """
    Map a label to its taxonomy category.
    Checks military map first, then COCO, then defaults to 'Other'.
    """
    if label in MILITARY_TAXONOMY_MAP:
        return MILITARY_TAXONOMY_MAP[label]
    if label in COCO_TAXONOMY_MAP:
        return COCO_TAXONOMY_MAP[label]
    return "Other"


# Threat levels for prioritization
THREAT_LEVELS = {
    "Artillery": 4,
    "Weapon": 4,
    "Explosive": 4,
    "Military Personnel": 3,
    "Military Vehicle": 3,
    "Fortification": 2,
    "Civilian Vehicle": 1,
    "Civilian": 1,
    "Other": 0,
}

# Colors by threat level
THREAT_COLORS = {
    4: (0, 0, 255),       # Red - Critical
    3: (0, 128, 255),     # Orange - High
    2: (0, 255, 255),     # Yellow - Medium
    1: (0, 255, 0),       # Green - Minimal
    0: (200, 200, 200),   # Gray - None
}