"""Anomaly-definition verbalizer — port of LaGoVAD's ``DatasetSpecVerbalizer``.

Maps class names to natural-language definition sentences per dataset; one
definition is sampled per call (the baseline's training-time conditioning,
carried here for TAD/DoTA/DADA as well so definitions are available during
training, not just eval). Datasets the project does not target (xd, sht,
ubnormal, nwpu, ubif, lad) are not ported.
"""

from __future__ import annotations

import random

# Special class name marking "abnormal, but unlabeled category": the verbalizer
# returns one sampled definition for every class of the active dataset.
SPECIAL_ABNORMAL_CLS = "Abnormal"

# Baseline's literal normal query caption (prepended to the caption set fed to
# the caption branch; see LaGoVAD training_step).
NORMAL_QUERY_CAPTION = (
    "Normal behavior, views or surveillance content captured by camera that "
    "not involve any unexpected or unusual events."
)

# Dataset-directory names (core.constants) -> verbalizer keys.
DATASET_NAME_TO_ABBR = {
    "PreVAD": "prevad",
    "UCF-Crime": "ucf",
    "MSAD": "msad",
    "MSAD-full": "msad",  # entire MSAD benchmark; same class taxonomy/definitions
    "DoTA": "dota",
    "DADA2000": "dada",
    "TAD": "tad",
}


def dataset_abbr(dataset_name: str) -> str:
    """Map a dataset directory name (or an abbr itself) to the verbalizer key."""
    if dataset_name in DATASET_NAME_TO_ABBR:
        return DATASET_NAME_TO_ABBR[dataset_name]
    if dataset_name in DATASET_CLS_DEFS:
        return dataset_name
    raise KeyError(
        f"Unknown dataset {dataset_name!r}; known: "
        f"{sorted(DATASET_NAME_TO_ABBR)} or {sorted(DATASET_CLS_DEFS)}"
    )


# Spec §7.4: traffic definition set used for DoTA/DADA-style zero-shot eval.
TRAFFIC_DEFINITIONS = [
    "vehicle collision",
    "vehicle-pedestrian collision",
    "out-of-control vehicle",
    "vehicle driving against traffic",
]

_UNIVERSAL_CLS_DEFS: dict[str, list[str]] = {
    "Normal": [
        "Normal",
        "Normal user uploaded video",
        "Normal surveillance footage.",
        "Normal behavior, views or surveillance content captured by camera that "
        "not involve any unexpected or unusual events.",
    ],
    "Assault": [
        "Assault",
        "The man unilaterally attacked the victim, using fists or sticks",
        "Multiple people surround and assault one person with fists and cudgels.",
        "Continuous assault by one person on another",
    ],
    "AirAccident": [
        "Airplane crash scene",
        "Air accident",
        "The plane crashed from the sky, emitting black smoke",
        "The helicopter is hovering and falling to the ground, there may be an explosion",
    ],
    "AnimalAttackAnimal": [
        "An animal attacks another animal, often resulting in injury or death.",
        "Animals attack each other",
    ],
    "AnimalAttackHuman": [
        "An animal attacks a person, often resulting in injury or death.",
        "Animals (such as cows, bears, and deer) attack people and collide with property.",
        "Mad dog barks at people and bites them",
    ],
    "AnimalPredation": [
        "Predation",
        "Predatory scenes, often bloody and violent",
        "Animal hunting and consuming another for food, characterized by stalking, "
        "chasing, and killing prey.",
        "Animal hunting and killing another animal for food",
    ],
    "CarAccident": [
        "A collision between two or more vehicles, often resulting in injury or damage.",
        "Traffic accident scene, vehicle colliding with pedestrian",
        "The driving recorder recorded that two cars collided with each other",
        "Road accident scene",
    ],
    "Collapse": [
        "Collapse scene",
        "The sudden falling down of a structure, such as a building or bridge, "
        "creating a pile of rubble and dust clouds.",
        "A visual event where a structure, such as a building, falls down or caves in.",
    ],
    "CrowdViolence": [
        "A Crowd violence scene",
        "A chaotic situation in a group of people that turns violent, marked by "
        "pushing, punching, kicking.",
        "Violent behavior by a group of people, often leading to conflict or chaos.",
        "A scene of violent behavior involving a large group of people.",
    ],
    "Explosion": [
        "Explosion",
        "Explosion, often resulting in fire, smoke, and scattered debris.",
        "The scene of the explosion, with mushroom clouds and smoke after the explosion.",
    ],
    "FallDown": [
        "People fall down",
        "Someone losing balance and dropping to the ground, which can happen due to "
        "tripping, slipping, or being pushed.",
        "Falling while running",
        "People falling from a height",
    ],
    "Fighting": [
        "Fighting",
        "Violence",
        "Using violence to injure or kill someone, usually involving group fights.",
        "A group of people fighting and brawling, which can be seen in punches, kicks.",
        "In sports, players have conflicts and start fighting each other.",
        "A man knocked down another person to the ground.",
    ],
    "Fire": [
        "Fire disaster",
        "Fire accident. Burning properties and thick black smoke can be seen",
        "Flames burn in unexpected places",
    ],
    "FallIntoWater": [
        "Someone falling into water",
        "A scene where a person or object falls into a body of water.",
        "An object or person falling into water, often involving splash and "
        "subsequent disappearance of the object.",
    ],
    "MechanicalAccident": [
        "Accident due to machinery failures, often happened in factories, "
        "warehouses, or other industrial settings.",
        "An incident in a manufacturing plant that results in injuries, fatalities, "
        "or environmental damage, often due to machinery malfunctions or hazardous "
        "material leaks.",
        "Accident happened in a industrial setting.",
    ],
    "ObjectImpact": [
        "Object falls down",
        "An object strikes another, visible through dents, cracks, or broken pieces.",
        "An object falls and potentially hits a person",
    ],
    "Riot": [
        "Riot scene",
        "The chaotic riot scene. There are many people and special police officers "
        "who are suppressing it.",
        "large-scale, public riot, often involving breaking windows, setting fires, "
        "and clashing with law enforcement.",
        "Riot scene, armed police wearing helmets and holding shields forming a "
        "human wall, with smoke and flames in the background.",
        "Riot scene. The crowd marched with flags and slogans with police holding "
        "shields to form a human wall suppressing the riot. ",
    ],
    "Robbery": [
        "Robbery",
        "Robbing others property through violent means such as beating or holding a gun",
    ],
    "Shooting": [
        "Shooting",
        "Firing a weapon, usually involving muzzle flash and the trajectory of the bullets",
        "The act of firing a firearm, often with gun flame and people lying down.",
        "A person points a gun at another person and shoots, and the muzzle emits "
        "flames and smoke.",
    ],
    "TrainAccident": [
        "Train accident",
        "A collision or derailment involving a train",
        "An accident involving trains, which can result in derailments or collisions.",
        "The train collided with vehicles at the gate",
    ],
    "WarScene": [
        "War scene",
        "War scene, often involving gunfire, explosions, cannon, tanks, and blood.",
        "War scene, with fully armed soldiers carrying out tasks",
        "A combat scenario during armed conflict, featuring explosions, gunfire, "
        "smoke, and the movement of military vehicles.",
        "A scene depicting warfare, involving battles, combat, and destruction.",
    ],
}

# PreVAD v6 taxonomy — the 36 names of the *live* ``DEFAULT_CLASSES`` in
# ``LaGoVAD-PreVAD/src/datasets/PreVAD.py`` (space-separated), NOT the
# CamelCase list commented out above it. The release's own train/test CSVs use
# exactly these names in ``class_name``; 35 of the 36 occur in the data
# (``Fire-related Accident`` is a superclass with no "Others" rows), so
# ``defs.json`` carries 35 while this lookup covers all 36 and can never
# under-cover. Six names are simultaneously superclasses and classes
# (``Vehicle Accident``, ``Violence``, ``Robbery``, ``Production Accident``,
# ``Fire-related Accident``, ``Animal-related Violence``, ``Daily Accident``):
# those rows are the taxonomy's "Others" buckets, so their definitions are
# deliberately broad.
_PREVAD_CLS_DEFS: dict[str, list[str]] = {
    "Normal": _UNIVERSAL_CLS_DEFS["Normal"],
    # --- 1. Vehicle Accident ------------------------------------------------
    "Vehicle Accident": [
        "A vehicle accident",
        "A moving vehicle crashes, overturns or loses control, leaving wreckage "
        "and debris on the scene.",
        "An accident involving a vehicle of any kind — a boat, a motorcycle, a "
        "bus or a truck — that ends in a collision or a rollover.",
    ],
    "Air Accident": _UNIVERSAL_CLS_DEFS["AirAccident"],
    "Train Accident": _UNIVERSAL_CLS_DEFS["TrainAccident"],
    "Car Accident": _UNIVERSAL_CLS_DEFS["CarAccident"],
    # --- 2. Violence --------------------------------------------------------
    "Violence": _UNIVERSAL_CLS_DEFS["Fighting"],
    "Vandalism": [
        "Vandalism",
        "Vandalism, smashing store doors, breaking windows.",
        "Deliberate destruction of property — kicking in doors, overturning "
        "shelves, damaging parked vehicles.",
    ],
    "Crowd Violence": _UNIVERSAL_CLS_DEFS["CrowdViolence"],
    "Riot": _UNIVERSAL_CLS_DEFS["Riot"],
    "Assault": _UNIVERSAL_CLS_DEFS["Assault"],
    "Range Shooting": [
        "Shooting in a range",
        "A shooting scene in a range, often involving multiple people and firearms.",
        "Firing guns at a shooting range, where people aim at targets from a set "
        "distance, often seen bullet holes in targets.",
        "The act of shooting at a target from a distance",
    ],
    "Shooting Accident": [
        "Shooting accident",
        "A firearm discharges unintentionally, injuring the shooter or a "
        "bystander, with visible muzzle flash and people reacting in alarm.",
        "Someone is shot, the muzzle emits flames and smoke, and people scatter "
        "or fall to the ground.",
        "An unintended gunshot during handling, cleaning or celebration with a "
        "firearm.",
    ],
    "War": _UNIVERSAL_CLS_DEFS["WarScene"],
    # --- 3. Robbery ---------------------------------------------------------
    "Robbery": _UNIVERSAL_CLS_DEFS["Robbery"],
    "Carjacking": [
        "Carjacking",
        "A vehicle is taken by force from its driver, often at gunpoint or by "
        "dragging the driver out of the seat.",
        "Someone approaches a stopped car, threatens the occupant and drives the "
        "vehicle away.",
    ],
    "Mugging": [
        "Mugging",
        "A person is robbed in the street, their bag or phone snatched, often "
        "after being pushed or threatened.",
        "A street robbery of a passer-by by one or more people, involving "
        "grabbing, shoving or a weapon.",
    ],
    "Store Robbery": [
        "Store robbery",
        "Someone robs a shop or a convenience store, threatening the clerk over "
        "the counter and taking money from the register.",
        "An armed robbery inside a store, with the staff raising their hands and "
        "the cash drawer being emptied.",
    ],
    # --- 4. Production Accident --------------------------------------------
    "Production Accident": [
        "A production accident",
        "An accident on a work site — a factory, a warehouse, a construction "
        "site — that injures a worker or destroys equipment.",
        "An industrial mishap during work, involving heavy loads, machinery or "
        "unstable structures.",
    ],
    "Mechanical Accident": _UNIVERSAL_CLS_DEFS["MechanicalAccident"],
    "Object Impact": _UNIVERSAL_CLS_DEFS["ObjectImpact"],
    "Collapse": _UNIVERSAL_CLS_DEFS["Collapse"],
    "Fall from Height": [
        "A person falls from a height",
        "Someone falls from a roof, a ladder, scaffolding or a balcony and hits "
        "the ground below.",
        "A worker loses footing at elevation and drops, often with the structure "
        "or platform giving way.",
    ],
    # --- 5. Fire-related Accident ------------------------------------------
    "Fire-related Accident": [
        "A fire-related accident",
        "An accident involving fire, heat or smoke — flames spreading, thick "
        "smoke filling the scene, or a sudden burst of heat.",
    ],
    "Fume": [
        "Smoke and fumes",
        "Thick smoke or gas billows out of a vehicle, a machine or a building, "
        "with no open flame yet visible.",
        "Dense white or black fumes pour from a hood, a vent or an engine "
        "compartment and people back away.",
    ],
    "Fire": _UNIVERSAL_CLS_DEFS["Fire"],
    "Explosion": _UNIVERSAL_CLS_DEFS["Explosion"],
    # --- 6. Animal-related Violence -----------------------------------------
    "Animal-related Violence": [
        "Animal-related violence",
        "A violent encounter involving an animal, ending in injury to the animal "
        "or to a person.",
        "An animal behaves aggressively — charging, biting or trampling.",
    ],
    "Predation": _UNIVERSAL_CLS_DEFS["AnimalPredation"],
    "Animal Attack Animal": _UNIVERSAL_CLS_DEFS["AnimalAttackAnimal"],
    "Animal Attack Human": _UNIVERSAL_CLS_DEFS["AnimalAttackHuman"],
    # --- 7. Daily Accident --------------------------------------------------
    "Daily Accident": [
        "An everyday accident",
        "A small mishap in ordinary life — someone slips, trips, drops something "
        "or bumps into an obstacle.",
        "An unexpected everyday accident at home, in a shop or on the street.",
    ],
    "Sport Fail": [
        "A sporting failure",
        "An athlete crashes, misses or wipes out during a sport — falling off a "
        "bike, missing a landing, colliding with another player.",
        "A failed attempt during a sporting activity that ends in a fall or a "
        "collision.",
    ],
    "Stunt Fail": [
        "A failed stunt",
        "A stunt goes wrong — the jump falls short, the trick is missed, and the "
        "performer crashes hard.",
        "Someone attempting a daring trick loses control and hits the ground, a "
        "wall or an obstacle.",
    ],
    "Fall into Water": _UNIVERSAL_CLS_DEFS["FallIntoWater"],
    "Fall to the Ground": _UNIVERSAL_CLS_DEFS["FallDown"],
    "Drop Something": [
        "Someone drops something",
        "An object slips out of someone's hands and falls to the floor, often "
        "breaking or spilling.",
        "A carried item is dropped and hits the ground, scattering its contents.",
    ],
}

_UCF_CLS_DEFS: dict[str, list[str]] = {
    "Normal": _UNIVERSAL_CLS_DEFS["Normal"],
    "Abuse": [
        "Intentional beating or abuse of animals like dogs.",
        "Abuse, ill-treatment of pets like dogs or cats.",
        "Beating or kicking pets, torture of animals",
    ],
    "Arrest": [
        "Police arresting suspects, which may involve pressing them to the ground, "
        "controlling hands or aiming with guns."
    ],
    "Arson": [
        "The deliberate setting of a fire by someone, usually characterized by "
        "flames, smoke, puring gasoline."
    ],
    "Assault": _UNIVERSAL_CLS_DEFS["Assault"],
    "Burglary": [
        "Burglary, usually characterized by crossing the cashier, breaking doors "
        "and windows, and carry things."
    ],
    "Explosion": _UNIVERSAL_CLS_DEFS["Explosion"],
    "Fighting": _UNIVERSAL_CLS_DEFS["Fighting"],
    "RoadAccidents": _UNIVERSAL_CLS_DEFS["CarAccident"],
    "Robbery": [
        "Robbing others property through violent means such as beating or holding a gun"
    ],
    "Shooting": _UNIVERSAL_CLS_DEFS["Shooting"],
    "Shoplifting": [
        "Shoplifting, sneak things into bags, clothes or under skirts in stores."
    ],
    "Stealing": ["Stealing property from cars or stealing motorcycles and batteries."],
    "Vandalism": ["Damaging vehicles, overturning shelves, or smashing store door."],
}

_MSAD_CLS_DEFS: dict[str, list[str]] = {
    "Normal": _UNIVERSAL_CLS_DEFS["Normal"],
    "Assault": _UNIVERSAL_CLS_DEFS["Assault"],
    "Fighting": _UNIVERSAL_CLS_DEFS["Fighting"],
    "People_falling": _UNIVERSAL_CLS_DEFS["FallDown"],
    "Robbery": _UNIVERSAL_CLS_DEFS["Robbery"],
    "Shooting": _UNIVERSAL_CLS_DEFS["Shooting"],
    "Traffic_accident": _UNIVERSAL_CLS_DEFS["CarAccident"],
    "Vandalism": [
        "Vandalism",
        "Vandalism, smashing store door, breaking windows.",
    ],
    "Explosion": _UNIVERSAL_CLS_DEFS["Explosion"],
    "Fire": _UNIVERSAL_CLS_DEFS["Fire"],
    "Object_falling": [
        "Object Falling",
        "Something like trees or buildings collapsed due to strong winds, "
        "earthquakes, or impacts.",
    ],
    "Water_incident": [
        "flood scene, with vehicles or furniture submerged",
        "a furious storm",
        "The indoor corridor is filled with water",
    ],
}

_DOTA_CLS_DEFS: dict[str, list[str]] = {
    "Normal": _UNIVERSAL_CLS_DEFS["Normal"],
    "CarAccident": _UNIVERSAL_CLS_DEFS["CarAccident"],
}

_TAD_CLS_DEFS: dict[str, list[str]] = {
    "Normal": _UNIVERSAL_CLS_DEFS["Normal"],
    "Car Accident": _UNIVERSAL_CLS_DEFS["CarAccident"],
}

DATASET_CLS_DEFS: dict[str, dict[str, list[str]]] = {
    "prevad": _PREVAD_CLS_DEFS,
    "ucf": _UCF_CLS_DEFS,
    "msad": _MSAD_CLS_DEFS,
    "dota": _DOTA_CLS_DEFS,
    "dada": _DOTA_CLS_DEFS,  # same traffic definition set as DoTA (spec §7.5)
    "tad": _TAD_CLS_DEFS,
}


class DatasetSpecVerbalizer:
    """Sample one definition string per class name for the active dataset."""

    def __init__(
        self, dataset: str = "prevad", rng: random.Random | None = None
    ) -> None:
        # definition sampling is training augmentation, not security-sensitive
        self._rng = rng if rng is not None else random.Random()  # nosec B311
        self.curr_dataset_name = ""
        self.cls2text: dict[str, list[str]] = {}
        self.set_dataset(dataset)

    def set_dataset(self, dataset_name: str) -> None:
        if dataset_name == self.curr_dataset_name:
            return
        if dataset_name not in DATASET_CLS_DEFS:
            raise KeyError(
                f"No definitions for dataset {dataset_name!r}; "
                f"available: {sorted(DATASET_CLS_DEFS)}"
            )
        self.curr_dataset_name = dataset_name
        self.cls2text = DATASET_CLS_DEFS[dataset_name]

    def __call__(self, inputs: str | list[str]) -> str | list[str]:
        if isinstance(inputs, str):
            return self._rng.choice(self.cls2text[inputs])
        if len(inputs) == 2 and SPECIAL_ABNORMAL_CLS in inputs:
            # unlabeled-abnormal marker: verbalize every known class instead
            return [self._rng.choice(defs) for defs in self.cls2text.values()]
        return [self._rng.choice(self.cls2text[name]) for name in inputs]


def verbalize_class_name(verbalizer: DatasetSpecVerbalizer, name: str) -> str:
    """One sampled definition for ``name``, index-aligned with its class list.

    Classes without definitions (e.g. the ``Abnormal`` fallback) verbalize to
    themselves — unlike ``verbalizer([...])``, this never changes list length.
    """
    if name in verbalizer.cls2text:
        return str(verbalizer(name))
    return name


__all__ = [
    "DATASET_CLS_DEFS",
    "DATASET_NAME_TO_ABBR",
    "NORMAL_QUERY_CAPTION",
    "SPECIAL_ABNORMAL_CLS",
    "TRAFFIC_DEFINITIONS",
    "DatasetSpecVerbalizer",
    "dataset_abbr",
    "verbalize_class_name",
]
