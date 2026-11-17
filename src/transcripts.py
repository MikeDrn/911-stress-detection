"""Scripted 911 call transcripts - the *content* training data.

No public dataset of real covert 911 calls exists (privacy), so this module
writes realistic caller-side transcripts from scenario *families*. Three classes:

genuine_emergency  the caller openly reports an emergency. Stress is EXPECTED here
                   ("someone is dead in front of me") and must NOT raise a threat alert.
non_emergency      information requests, complaints, honest misdials, pocket dials.
covert             the caller is under threat and cannot speak freely: orders a
                   pizza on 911, answers the dispatcher's yes/no questions in code
                   ("yes, with pepperoni"), pretends to call a relative, whispers.

Three design choices keep the task honest (lesson from the synthetic-audio model):

1. Families are held out: train / val / test use *different* scenario families and
   phrasings, so test accuracy measures generalisation, not memorised templates.
2. Hard negatives on purpose: genuine emergencies that mention food ("he's choking
   on pizza"), honest misdials to a pizza shop, covert calls that never mention food.
   A keyword rule ("pizza => threat") must fail - the sanity report checks it.
3. Speech-to-text noise: lower case, no punctuation, fillers (uh, um), dropped and
   misheard words - what Whisper output looks like on a real phone line.

The transcripts are authored, not recorded: present them as such.
"""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path

LABELS = ("genuine_emergency", "non_emergency", "covert")

SLOTS: dict[str, list[str]] = {
    "addr": [
        "1420 maple street",
        "88 oak avenue",
        "2310 river road",
        "17 hillcrest drive",
        "455 park lane apartment 3",
        "92 cedar court",
        "3001 lakeview boulevard",
        "61 elm street",
        "740 sunset avenue",
        "12 harbor way unit 5",
        "205 pine street",
        "1550 west main street",
    ],
    "road": [
        "highway 9",
        "the main bridge",
        "route 66",
        "fifth avenue",
        "the ring road",
        "the coastal highway",
        "exit 14",
        "the roundabout near the mall",
    ],
    "place": [
        "kitchen",
        "garage",
        "basement",
        "living room",
        "bedroom",
        "apartment",
        "building",
        "office",
    ],
    "room": ["bathroom", "closet", "bedroom", "attic", "basement"],
    "food": [
        "a large pizza",
        "two medium pizzas",
        "a family size pizza",
        "a pizza",
        "an order of chinese food",
        "some sushi",
        "a burger meal",
        "a sandwich",
    ],
    "topping": [
        "pepperoni",
        "extra cheese",
        "mushrooms",
        "olives",
        "sausage",
        "no onions",
        "chicken",
        "peppers",
    ],
    "name": [
        "mike",
        "sarah",
        "john",
        "lina",
        "david",
        "maria",
        "tony",
        "rachel",
        "omar",
        "emma",
        "chris",
        "nadia",
    ],
    "office": [
        "dmv",
        "city hall",
        "post office",
        "passport office",
        "police station",
        "town library",
        "tax office",
    ],
    "business": [
        "pizza place",
        "pizzeria",
        "chinese restaurant",
        "taxi company",
        "pharmacy",
        "dentist",
        "hair salon",
        "bank",
    ],
    "item": ["wallet", "phone", "keys", "bag", "bike", "passport"],
    "pet": ["dog", "cat", "parrot", "puppy"],
    "time": ["tonight", "right now", "in twenty minutes", "as soon as possible", "today"],
    "color": ["blue", "red", "black", "white", "gray"],
}

# family -> (label, list of templates). Each template is a list of sentence options
# that are concatenated (some optional with "?" prefix).
FAMILIES: dict[str, tuple[str, list[list[str]]]] = {
    # ------------------------------------------------------------ genuine emergencies
    "collapse": (
        "genuine_emergency",
        [
            [
                "my {rel} just collapsed",
                "my {rel} fell down and isn't moving",
                "oh god my {rel} collapsed on the floor",
            ],
            [
                "{pron} is not breathing",
                "{pron} won't wake up",
                "{pron} is turning blue",
                "i can't feel a pulse",
            ],
            [
                "?please send an ambulance to {addr}",
                "?we're at {addr} please hurry",
                "?please hurry please",
            ],
        ],
    ),
    "chest_pain": (
        "genuine_emergency",
        [
            [
                "my {rel} is having really bad chest pains",
                "{rel} says {pron} can't breathe and {pron} chest hurts",
                "i think my {rel} is having a heart attack",
            ],
            [
                "{pron} is sweating and pale",
                "{pron} is holding {poss} arm",
                "{pron} is really dizzy",
            ],
            ["?the address is {addr}", "?we need an ambulance at {addr}"],
        ],
    ),
    "overdose": (
        "genuine_emergency",
        [
            [
                "i found my {rel} unconscious",
                "my {rel} took a whole bottle of pills",
                "i think my {rel} overdosed",
            ],
            ["{pron} is barely breathing", "there are pills everywhere", "{pron} won't respond"],
            ["?please come to {addr}", "?hurry we're at {addr}"],
        ],
    ),
    "fire": (
        "genuine_emergency",
        [
            [
                "there's a fire in my {place}",
                "my neighbor's house is on fire",
                "the whole {place} is full of smoke",
            ],
            [
                "the flames are getting bigger",
                "i can't get to the door",
                "there are kids inside",
                "?we got out but my {rel} is still inside",
            ],
            ["?it's {addr}", "?send the fire department to {addr}"],
        ],
    ),
    "car_crash": (
        "genuine_emergency",
        [
            [
                "there's been a bad car accident on {road}",
                "a car just hit a pedestrian on {road}",
                "two cars crashed on {road}",
            ],
            [
                "someone is trapped inside",
                "there's a lot of blood",
                "one of them isn't moving",
                "the car is smoking",
            ],
            ["?please send help", "?we need ambulances"],
        ],
    ),
    "witnessed_death": (
        "genuine_emergency",
        [
            [
                "someone is dead in front of me",
                "i think my {rel} is dead",
                "i just found a body in the {place}",
            ],
            ["{pron} is cold and not moving", "there's no pulse", "i don't know what to do"],
            ["?i'm at {addr}", "?please send someone to {addr}"],
        ],
    ),
    "open_break_in": (
        "genuine_emergency",
        [
            [
                "someone is breaking into my house right now",
                "i can hear someone breaking the back door",
                "there's a man trying to get in through my window",
            ],
            ["i'm hiding in the {room}", "i locked myself in the {room}", "i'm alone with my kids"],
            ["?i'm at {addr} please send police", "?the address is {addr}"],
        ],
    ),
    "assault_witness": (
        "genuine_emergency",
        [
            [
                "a man is attacking a woman outside",
                "there's a fight and someone pulled a knife",
                "someone just got stabbed outside the {business}",
            ],
            ["he's still hitting her", "there's blood on the ground", "people are screaming"],
            ["?it's near {addr}", "?please send police now"],
        ],
    ),
    "missing_child": (
        "genuine_emergency",
        [
            [
                "my {child} is missing",
                "i can't find my {child} anywhere",
                "my {child} was in the park a minute ago and now {pron} is gone",
            ],
            [
                "{pron} is only five years old",
                "i've looked everywhere",
                "{pron} was wearing a {color} jacket",
            ],
            ["?we were near {addr}", "?please help me find {obj}"],
        ],
    ),
    "choking_food": (
        "genuine_emergency",
        [  # hard negative: food words, real emergency
            [
                "we were eating {food} and my {child} started choking",
                "my {rel} is choking on a piece of pizza",
                "my {child} swallowed something while eating",
            ],
            ["{pron} can't breathe", "{pron} is turning red", "i tried to hit {poss} back"],
            ["?please send an ambulance to {addr}", "?what do i do"],
        ],
    ),
    "seizure": (
        "genuine_emergency",
        [
            ["my {rel} is having a seizure", "my {child} is shaking and won't stop"],
            ["{pron} hit {poss} head", "it's been going on for five minutes"],
            ["?we're at {addr}", "?please send someone"],
        ],
    ),
    "drowning": (
        "genuine_emergency",
        [
            [
                "a kid fell into the pool",
                "someone is drowning in the lake",
                "my {child} went under the water",
            ],
            ["we pulled {obj} out but {pron} isn't breathing", "{pron} is not coming up"],
            ["?we're at {addr}", "?hurry please"],
        ],
    ),
    "gas_leak": (
        "genuine_emergency",
        [
            ["i smell gas everywhere in my {place}", "there's a gas leak in our building"],
            ["my {rel} fainted", "everyone is getting dizzy", "we're getting out now"],
            ["?it's {addr}", "?send the fire department"],
        ],
    ),
    "shots_fired": (
        "genuine_emergency",
        [
            ["i just heard gunshots outside", "someone is shooting near the {business}"],
            ["people are running", "i think someone got hit", "i'm hiding behind a car"],
            ["?it's near {road}", "?please send police"],
        ],
    ),
    # ------------------------------------------------------------ non emergencies
    "info_question": (
        "non_emergency",
        [
            ["hi i have a question", "hello sorry to bother you", "yes hello"],
            [
                "what time does the {office} open",
                "how do i renew my driver's license",
                "do you know if the {office} is open on saturday",
                "where do i pay a parking ticket",
            ],
            ["?thank you", "?okay thanks bye"],
        ],
    ),
    "noise_complaint": (
        "non_emergency",
        [
            [
                "my neighbors are playing really loud music",
                "there's a party next door and it's two in the morning",
                "the people upstairs won't stop making noise",
            ],
            ["i can't sleep", "this happens every weekend", "can someone tell them to stop"],
            ["?i live at {addr}", "?it's not an emergency i just don't know who to call"],
        ],
    ),
    "lost_item": (
        "non_emergency",
        [
            [
                "i lost my {item}",
                "i think someone took my {item} from my car",
                "i left my {item} on the bus",
            ],
            ["how do i report it", "do i need to file a report", "is there a lost and found"],
            ["?thanks", "?okay i'll call tomorrow"],
        ],
    ),
    "lost_pet": (
        "non_emergency",
        [
            ["my {pet} ran away", "i lost my {pet} this morning", "my {pet} escaped from the yard"],
            ["{pron} is {color} and small", "can you tell animal control", "who should i call"],
            ["?i'm at {addr}", "?thank you"],
        ],
    ),
    "honest_misdial_food": (
        "non_emergency",
        [  # hard negative: food words, no threat
            [
                "oh sorry wrong number",
                "oops i'm so sorry i dialed the wrong number",
                "sorry i meant to call the {business}",
            ],
            ["i was trying to order {food}", "i hit the wrong button", "my mistake"],
            ["everything is fine sorry", "sorry to bother you bye", "have a good night"],
        ],
    ),
    "pocket_dial": (
        "non_emergency",
        [
            [
                "sorry my kid was playing with the phone",
                "oh i didn't mean to call",
                "sorry my phone called by itself",
            ],
            ["everything is okay here", "there's no emergency", "sorry about that"],
            ["?bye", "?have a good day"],
        ],
    ),
    "power_outage": (
        "non_emergency",
        [
            [
                "the power is out in our whole street",
                "there's a streetlight broken on {road}",
                "there's a big pothole on {road}",
            ],
            ["who do i report it to", "is it being fixed", "how long will it take"],
            ["?thanks", "?okay"],
        ],
    ),
    "report_number": (
        "non_emergency",
        [
            [
                "i need the number of my police report",
                "i want to check on a report i made last week",
                "someone scratched my car in the parking lot yesterday",
            ],
            ["what information do you need", "can i do it online", "it's not urgent"],
            ["?thank you", "?bye"],
        ],
    ),
    "parking": (
        "non_emergency",
        [
            ["someone parked in front of my driveway", "there's a car blocking my garage"],
            ["it's been there all day", "can someone tow it", "i need to leave for work"],
            ["?the address is {addr}", "?thanks"],
        ],
    ),
    # ------------------------------------------------------------ covert (under threat)
    "pizza_order": (
        "covert",
        [
            [
                "hi i'd like to order {food}",
                "yes hello i want to order {food} for delivery",
                "can i get {food} please",
            ],
            ["with {topping}", "yes with {topping}", "{topping} please"],
            ["deliver it to {addr}", "the address is {addr}", "it's for {addr}"],
            ["?as fast as you can please", "?yes please hurry", "?yes that's right"],
        ],
    ),
    "yes_no_code": (
        "covert",
        [  # answering the dispatcher's covert questions
            ["yes", "yeah", "yes that's right"],
            ["yes with {topping}", "no just one", "yes he's still here", "yes there are two"],
            ["no i can't", "yes the {color} one", "no not right now"],
            ["?{addr}", "?please hurry", "?yes"],
        ],
    ),
    "pretend_relative": (
        "covert",
        [
            ["hi mom", "hey {name} it's me", "hi sis how are you"],
            ["yeah i'm fine everything is fine", "no no everything's okay", "i'm good i'm good"],
            [
                "can you come over to {addr}",
                "could you stop by {addr} {time}",
                "bring {name} with you okay",
            ],
            ["?i can't really talk right now", "?just come quick okay", "?love you bye"],
        ],
    ),
    "fake_appointment": (
        "covert",
        [
            [
                "hi i'm calling about my appointment",
                "yes i wanted to confirm the repair visit",
                "i need someone to come check the cable",
            ],
            ["can someone come to {addr} {time}", "it has to be {time}", "i'm at {addr}"],
            ["yes the door is unlocked", "please come as soon as possible", "yes just knock"],
        ],
    ),
    "whisper_threat": (
        "covert",
        [
            ["i can't talk", "he's here", "please don't hang up"],
            ["he has a gun", "he's in the other room", "he'll hear me"],
            ["?{addr}", "?please come quietly"],
        ],
    ),
    "stay_on_line": (
        "covert",
        [
            ["oh sorry i think i have the wrong number", "oh hi sorry"],
            [
                "actually can you stay on the line",
                "no wait don't hang up",
                "can you just keep talking to me",
            ],
            ["i'll just talk about the weather", "it's a nice day isn't it", "yes yes i'm fine"],
            ["?{addr}", "?yes please send someone"],
        ],
    ),
    "taxi_order": (
        "covert",
        [
            ["hi i need a taxi", "can you send a cab to {addr}", "i'd like to book a ride"],
            ["yes to {addr}", "for two people", "{time} please"],
            ["yes just one passenger is with me", "no he won't let me leave", "please be quick"],
        ],
    ),
    "food_order_other": (
        "covert",
        [  # covert without pizza
            ["hello i want to order {food}", "can i place an order for {food}"],
            ["yes for {addr}", "deliver to {addr} please", "yes the usual"],
            ["no that's all", "yes he's hungry too", "please hurry i'm really hungry"],
        ],
    ),
    "fine_but_address": (
        "covert",
        [
            ["everything is fine", "no there's no problem here", "i'm okay"],
            [
                "but can you send someone to {addr}",
                "i just need someone at {addr}",
                "could a car drive by {addr}",
            ],
            ["yes", "no i can't explain", "please"],
        ],
    ),
}

# Families held out for evaluation (unseen phrasing types at test time).
TEST_FAMILIES = {
    "seizure",
    "shots_fired",
    "report_number",
    "pocket_dial",
    "food_order_other",
    "fine_but_address",
    "drowning",
}
VAL_FAMILIES = {"gas_leak", "parking", "taxi_order"}

FILLERS = ["uh", "um", "like", "you know", "okay", "uh huh", "hmm"]
MISHEARD = {
    "to": "two",
    "there": "their",
    "for": "four",
    "right": "write",
    "hear": "here",
    "no": "know",
    "won't": "want",
    "can't": "can",
    "please": "pleas",
}

# Person slots are gender-consistent with the pronouns used in the same transcript.
PRONOUNS = {
    "male": {"pron": "he", "poss": "his", "obj": "him"},
    "female": {"pron": "she", "poss": "her", "obj": "her"},
}
PEOPLE = {
    "male": {
        "rel": [
            "husband",
            "father",
            "son",
            "brother",
            "grandfather",
            "uncle",
            "roommate",
            "neighbor",
            "friend",
            "boyfriend",
        ],
        "child": ["son", "little boy", "baby", "kid"],
    },
    "female": {
        "rel": [
            "wife",
            "mother",
            "daughter",
            "sister",
            "grandmother",
            "aunt",
            "roommate",
            "neighbor",
            "friend",
            "girlfriend",
        ],
        "child": ["daughter", "little girl", "baby", "kid"],
    },
}


@dataclass
class Transcript:
    text: str
    label: str
    family: str
    split: str


def _fill(sentence: str, rng: random.Random, gender: str, chosen: dict[str, str]) -> str:
    def repl(m: re.Match) -> str:
        key = m.group(1)
        if key in PRONOUNS[gender]:
            return PRONOUNS[gender][key]
        if key in PEOPLE[gender]:  # the same relative throughout one call
            chosen.setdefault(key, rng.choice(PEOPLE[gender][key]))
            return chosen[key]
        return rng.choice(SLOTS[key])

    return re.sub(r"\{(\w+)\}", repl, sentence)


def asr_noise(text: str, rng: random.Random, strength: float = 1.0) -> str:
    """Make clean text look like phone-line speech-to-text output."""
    words = re.sub(r"[^\w\s']", " ", text.lower()).split()
    out = []
    for w in words:
        r = rng.random()
        if r < 0.03 * strength:
            continue  # dropped word
        if r < 0.07 * strength and w in MISHEARD:
            w = MISHEARD[w]
        out.append(w)
        if rng.random() < 0.06 * strength:
            out.append(rng.choice(FILLERS))
    if rng.random() < 0.3 * strength:
        out.insert(0, rng.choice(["hello", "hi", "uh hello", "yes hello"]))
    return " ".join(out)


def render(family: str, rng: random.Random, noise: float = 1.0) -> str:
    _, parts = FAMILIES[family]
    gender = rng.choice(sorted(PRONOUNS))
    chosen: dict[str, str] = {}
    sentences = []
    for options in parts:
        choice = rng.choice(options)
        if choice.startswith("?"):
            if rng.random() < 0.5:
                continue
            choice = choice[1:]
        sentences.append(_fill(choice, rng, gender, chosen))
    return asr_noise(". ".join(sentences), rng, noise)


def family_split(family: str) -> str:
    return "test" if family in TEST_FAMILIES else "val" if family in VAL_FAMILIES else "train"


def generate(per_family: int = 60, seed: int = 911) -> list[Transcript]:
    rng = random.Random(seed)
    rows, seen = [], set()
    for family in sorted(FAMILIES):
        label = FAMILIES[family][0]
        made, attempts = 0, 0
        while made < per_family and attempts < per_family * 20:
            attempts += 1
            text = render(family, rng)
            if text in seen:
                continue
            seen.add(text)
            rows.append(Transcript(text, label, family, family_split(family)))
            made += 1
    return rows


def write_csv(rows: list[Transcript], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "label", "family", "split"])
        for r in rows:
            w.writerow([r.text, r.label, r.family, r.split])


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    from src.config import PROJECT_ROOT, load_params

    p = load_params()["text"]
    rows = generate(int(p["per_family"]), int(p["seed"]))
    out = PROJECT_ROOT / p["transcripts"]
    write_csv(rows, out)
    from collections import Counter

    counts = Counter((r.split, r.label) for r in rows)
    print(f"[transcripts] {len(rows)} transcripts from {len(FAMILIES)} families -> {out}")
    for (split, label), n in sorted(counts.items()):
        print(f"               {split:5s} {label:18s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
