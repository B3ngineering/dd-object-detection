# Maps model labels to military-relevant categories
# Each category has an associated threat level and color for bounding boxes

MILITARY_TAXONOMY_MAP = {
    "camouflage_soldier": "Military Personnel",
    "soldier": "Military Personnel",
    "civilian": "Civilian",
    
    "military_tank": "Military Vehicle",
    "military_truck": "Military Vehicle",
    "military_vehicle": "Military Vehicle",
    "military_aircraft": "Military Vehicle",
    "military_warship": "Military Vehicle",
    "civilian_vehicle": "Civilian Vehicle",

    "weapon": "Weapon",
    "military_artillery": "Weapon",

    "trench": "Fortification",
}

# Threat levels for prioritization
THREAT_LEVELS = {
    "Armored Vehicle": 5,
    "Artillery": 5,
    "Aircraft": 4,
    "Naval Vessel": 4,
    "Weapon": 4,
    "Military Personnel": 3,
    "Military Transport": 3,
    "Military Vehicle": 3,
    "Fortification": 2,
    "Civilian Personnel": 1,
    "Civilian Vehicle": 1,
    "Other": 0,
}

# Colors by threat level
THREAT_COLORS = {
    5: (0, 0, 255),       # Red
    4: (0, 128, 255),     # Orange
    3: (0, 255, 255),     # Yellow
    2: (0, 255, 128),     # Light green
    1: (0, 255, 0),       # Green
    0: (200, 200, 200),   # Gray
}