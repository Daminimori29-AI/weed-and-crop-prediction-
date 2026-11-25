"""Weed knowledge base: names, precautions and control methods.

NOTE: Herbicide names are general examples only (no doses). Always confirm
product, dose and timing with your local Krishi Vigyan Kendra (KVK) /
agriculture officer, since they depend on crop, soil and region.
"""

# ---------------------------------------------------------------------------
# Species database. Keys must match the class names your model predicts
# (see class_names.json) once you train a multi-class model.
# ---------------------------------------------------------------------------
WEEDS = {
    "Parthenium": {
        "common_name": "Congress Grass / Carrot Weed",
        "hindi_name": "गाजर घास (पार्थेनियम)",
        "scientific_name": "Parthenium hysterophorus",
        "risk": "High",
        "description": "Fast-spreading annual weed with deeply lobed, carrot-like leaves and small white flowers.",
        "health_hazard": "Causes skin allergy, dermatitis, asthma and hay fever; toxic to livestock.",
        "precautions": [
            "Wear gloves, full-sleeve clothes and a mask when handling it.",
            "Remove it BEFORE flowering and seed set; one plant makes thousands of seeds.",
            "Do not burn it or let animals graze on it.",
            "Clean tools and tractor tyres so seeds do not spread to other fields.",
        ],
        "control": [
            "Hand-pull (with protection) before flowering, then bury or compost away from the field.",
            "Competitive crops/plants like Cassia (Chakunda) can help suppress it.",
            "Post-emergence herbicides (e.g. glyphosate on non-crop areas) as advised locally.",
        ],
    },
    "Nutsedge": {
        "common_name": "Purple Nutsedge",
        "hindi_name": "मोथा (नागरमोथा)",
        "scientific_name": "Cyperus rotundus",
        "risk": "High",
        "description": "Perennial sedge with triangular stem; spreads mainly through underground tubers.",
        "health_hazard": "No direct health hazard, but one of the world's worst crop weeds.",
        "precautions": [
            "Do not just cut the top; tubers regrow quickly.",
            "Avoid spreading tuber-contaminated soil between fields.",
            "Keep fields well drained; it thrives in wet soil.",
        ],
        "control": [
            "Deep summer ploughing to expose tubers to sun.",
            "Repeated hoeing/cultivation every 2-3 weeks to exhaust tubers.",
            "Crop rotation and mulching; selective herbicides (e.g. halosulfuron) as per local advice.",
        ],
    },
    "Bermuda Grass": {
        "common_name": "Bermuda Grass",
        "hindi_name": "दूब घास",
        "scientific_name": "Cynodon dactylon",
        "risk": "Medium",
        "description": "Perennial creeping grass spreading via stolons and rhizomes.",
        "health_hazard": "Pollen can trigger allergies in some people.",
        "precautions": [
            "Remove the full rhizome network, not only the green part.",
            "Avoid rotavating infested patches; it cuts and spreads rhizomes.",
        ],
        "control": [
            "Repeated deep tillage in hot dry weather.",
            "Solarization (clear plastic mulch) in summer.",
            "Grass-selective or non-selective herbicides on fallow land as per local advice.",
        ],
    },
    "Amaranth": {
        "common_name": "Slender / Green Amaranth (Pigweed)",
        "hindi_name": "जंगली चौलाई",
        "scientific_name": "Amaranthus viridis",
        "risk": "Medium",
        "description": "Fast-growing broadleaf annual producing very large numbers of seeds.",
        "health_hazard": "Generally low; may accumulate nitrates.",
        "precautions": [
            "Remove before seeding; seeds stay viable in soil for years.",
            "Do not compost mature seed-bearing plants.",
        ],
        "control": [
            "Timely hand weeding or inter-row hoeing at 15-30 days after sowing.",
            "Pre-emergence herbicide (e.g. pendimethalin) where recommended for your crop.",
            "Mulching to block light for germination.",
        ],
    },
    "Barnyard Grass": {
        "common_name": "Barnyard Grass",
        "hindi_name": "सांवा घास",
        "scientific_name": "Echinochloa crus-galli",
        "risk": "High",
        "description": "Annual grass that mimics rice seedlings; common in paddy and wet fields.",
        "health_hazard": "No direct hazard; severe yield competitor in rice.",
        "precautions": [
            "Weed early; it looks like rice before heading, so check carefully.",
            "Use clean, certified crop seed to avoid contamination.",
        ],
        "control": [
            "Maintain standing water in paddy to suppress germination.",
            "Hand weeding at 20-30 days; transplanting instead of direct seeding.",
            "Rice-safe herbicides (e.g. butachlor/bispyribac) only as advised locally.",
        ],
    },
    "Lamb's Quarters": {
        "common_name": "Lamb's Quarters / Goosefoot",
        "hindi_name": "बथुआ (जंगली)",
        "scientific_name": "Chenopodium album",
        "risk": "Medium",
        "description": "Broadleaf winter annual with mealy, diamond-shaped leaves.",
        "health_hazard": "Edible when young but can be high in oxalates; pollen allergen.",
        "precautions": [
            "Remove before seed set in rabi crops (wheat, mustard).",
            "Do not let it shade young crop seedlings.",
        ],
        "control": [
            "Hand weeding 25-35 days after sowing.",
            "Broadleaf herbicides in wheat (e.g. 2,4-D / metsulfuron) strictly per label and local advice.",
        ],
    },
    "Lantana": {
        "common_name": "Lantana",
        "hindi_name": "लैंटाना (राईमुनिया)",
        "scientific_name": "Lantana camara",
        "risk": "High",
        "description": "Woody invasive shrub with multicoloured flower clusters.",
        "health_hazard": "Berries and leaves are toxic to humans and livestock; skin irritation possible.",
        "precautions": [
            "Wear gloves and long sleeves; keep children and animals away.",
            "Uproot with roots; cut stumps regrow.",
        ],
        "control": [
            "Uproot young plants; cut-and-treat stumps with suitable herbicide per local advice.",
            "Dry the removed material and dispose away from water bodies.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Generic info used when the model only says "weed" (binary model).
# ---------------------------------------------------------------------------
GENERIC_WEED = {
    "title": "Weed detected (species not identified)",
    "note": "The current model only separates crop from weed, so it cannot tell "
            "which weed species this is. The advice below is general.",
    "precautions": [
        "Remove weeds early (first 3-4 weeks) - this is when they hurt yield the most.",
        "Remove before flowering/seed set to stop them spreading.",
        "Wear gloves; some weeds cause skin allergies or are toxic.",
        "Clean tools and machinery after working in weedy patches.",
        "Do not compost weeds that already carry seeds or tubers.",
    ],
    "control": [
        "Manual: hand weeding or hoeing between rows.",
        "Cultural: crop rotation, correct seed rate and spacing, mulching.",
        "Chemical (last resort): use only the herbicide recommended for your crop and weed type.",
    ],
}

GENERIC_CROP = {
    "title": "Crop detected",
    "precautions": [
        "Do not spray non-selective herbicide near this plant.",
        "Keep the area around crop plants free of weeds in the early growth stage.",
    ],
}

HERBICIDE_SAFETY = [
    "Read the label fully; use the exact dose and the crop stage mentioned on it.",
    "Wear gloves, mask, goggles and full-cover clothing while spraying.",
    "Spray in calm weather (no strong wind, no rain expected within hours).",
    "Keep away from water sources, children and animals; store in original containers.",
    "Wash hands/clothes after spraying and never eat or smoke while spraying.",
]

DISCLAIMER = (
    "Advice is general guidance. Confirm herbicide choice, dose and timing with your "
    "local Krishi Vigyan Kendra (KVK) or agriculture officer."
)
