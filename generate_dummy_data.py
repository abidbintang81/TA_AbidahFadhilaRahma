"""
generate_dummy_data.py
======================
Membuat DATASET DUMMY (bukan data asli) dengan format yang sama seperti hasil
scraping Apify untuk IMDb dan Letterboxd.

PENANDA DUMMY (agar mudah dikenali dan diganti):
  * Nama file diawali "DUMMY_".
  * IMDb: kolom `id` diawali "DUMMY-rw", `authorUserId` diawali "DUMMY-ur",
    semua URL mengarah ke https://example.com/DUMMY/...
  * Letterboxd: `username` diawali "dummy_", `reviewer` diawali "DUMMY ",
    `reviewurl` mengarah ke https://example.com/DUMMY/...
  * Teks ulasan disusun otomatis dari bank kalimat (sintetis), bukan kutipan
    ulasan pengguna sungguhan.

Data asli cukup diletakkan di data/raw/ lalu ubah SATU baris di config.py
(USE_DUMMY = False). Lihat README.md.

Jalankan:  python generate_dummy_data.py
"""
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
N_IMDB = 700
N_LBX = 360
OUT_DIR = Path(__file__).resolve().parent / "data" / "dummy"

# ---------------------------------------------------------------------------
# Bank kalimat per aspek. Setiap aspek punya kalimat positif (pos) & negatif (neg).
# ---------------------------------------------------------------------------
ASPECTS = {
    "acting": {
        "pos": [
            "Inde Navarrette gives a fearless performance as Nikki and completely owns every scene.",
            "The lead actress is phenomenal, switching from sweet to terrifying in a single look.",
            "Navarrette deserves award attention for this role, her performance is unhinged in the best way.",
            "Michael Johnston plays Bear with an awkward charm that makes the character believable.",
            "The acting carries the whole film, especially the way Nikki smiles while something is clearly wrong.",
            "Her facial expressions alone are scarier than most monsters in modern horror.",
            "The chemistry between the two leads feels real, which makes the descent even more painful.",
            "Every actor commits fully, but the actress playing Nikki is the reason to watch.",
            "What a breakout performance, the cast is small but the acting is outstanding.",
            "Navarrette balances vulnerability and menace so well that you never know which Nikki you will get.",
        ],
        "neg": [
            "The performances are uneven and Bear feels wooden for most of the runtime.",
            "Even a strong lead performance cannot save the weak supporting cast.",
            "Some of the acting from the side characters feels like a school play.",
            "Johnston never sells the panic his character should feel.",
        ],
    },
    "gore": {
        "pos": [
            "The gore is brutal and the body horror is used sparingly but effectively.",
            "There are a few bloody scenes that made the whole audience gasp.",
            "The practical effects look disgusting in the best possible way.",
            "When the violence arrives it is shocking, gruesome and very graphic.",
            "One kitchen scene is so gory that I had to look away from the screen.",
            "The blood and body horror feel earned instead of cheap.",
            "Fans of gruesome practical effects will be very happy with this one.",
        ],
        "neg": [
            "The gore feels gratuitous and replaces actual scares with shock value.",
            "Too much blood, too little story, the violence becomes numbing.",
            "The graphic violence is disturbing for no real reason and left a bad taste.",
            "Some of the bloody effects look cheap and fake.",
        ],
    },
    "scares": {
        "pos": [
            "The jump scares actually work because the tension builds slowly before each one.",
            "The sound design is incredible, every creak and whisper makes your skin crawl.",
            "I jumped out of my seat at least three times, the scares are genuinely effective.",
            "The atmosphere is tense and unsettling from the first minute to the last.",
            "The score and the silence are used perfectly to create dread.",
            "It is genuinely creepy and keeps you anxious the whole time.",
            "The tension is almost unbearable during the night scenes in the house.",
        ],
        "neg": [
            "Most jump scares are just loud noises with nothing behind them.",
            "The sound mixing is obnoxiously loud and the scares become predictable.",
            "It is not scary at all, the tension disappears after the first act.",
            "Cheap jump scares and loud sound effects replace real atmosphere.",
        ],
    },
    "premise": {
        "pos": [
            "The premise is simple but clever: a wish that makes your crush love you comes at a terrible price.",
            "The One Wish Willow concept is original and the story uses it in creative ways.",
            "I love how the story takes a silly magical wish and turns it into a nightmare.",
            "The concept of being careful what you wish for is executed really well.",
            "It is a fresh and original horror story, something we rarely get nowadays.",
            "The script builds its own rules for the wish and sticks to them.",
            "The story is unpredictable, I never knew what would happen next.",
        ],
        "neg": [
            "The premise is interesting but the story runs out of ideas halfway through.",
            "The wish concept is never explored deeply and the plot feels thin.",
            "The story is predictable once you understand the wish mechanic.",
            "A great concept wasted on a repetitive plot.",
        ],
    },
    "characters": {
        "pos": [
            "Bear is a frustrating character on purpose, and the writing knows it.",
            "The characters make questionable choices, but those choices fit who they are.",
            "The writing gives Bear just enough flaws to make the situation believable.",
            "Even the side characters have small moments that feel human.",
        ],
        "neg": [
            "Bear makes the dumbest decisions I have ever seen in a horror movie.",
            "No sane person would stay in that relationship, the character logic falls apart.",
            "The side characters are paper thin and only exist to move the plot forward.",
            "I could not believe the main character would ignore so many red flags.",
            "The writing forces characters to act stupid so the plot can continue.",
            "His behavior stretches plausibility way beyond what the story can support.",
            "It is hard to care about a protagonist who never takes responsibility.",
        ],
    },
    "humor": {
        "pos": [
            "It is surprisingly funny, I laughed as much as I screamed.",
            "The dark humor lands perfectly and makes the horror hit even harder.",
            "Some moments are so absurd that the whole theater burst out laughing.",
            "The comedy and horror blend together in a very fun way.",
            "The awkward humor is hilarious and very self aware.",
            "Funniest horror movie of the year, the customer service call is iconic.",
        ],
        "neg": [
            "The jokes undercut the tension and the tone feels confused.",
            "I was never sure if I should laugh or be scared, the humor did not work for me.",
            "The comedic moments feel out of place in such a dark story.",
        ],
    },
    "ending": {
        "pos": [
            "The ending is devious and absolutely perfect.",
            "That final scene is one of the best horror endings in years.",
            "The third act goes completely insane and I loved every second.",
            "The pacing is tight and the finale pays everything off.",
        ],
        "neg": [
            "The ending is rushed and leaves too many questions unanswered.",
            "The third act drags and the finale feels like a letdown.",
            "The pacing is slow in the middle and the movie is at least twenty minutes too long.",
            "It builds and builds but the ending has no real payoff.",
            "The final act spins in circles and goes nowhere.",
        ],
    },
    "director": {
        "pos": [
            "Curry Barker makes an impressive feature debut after his YouTube shorts.",
            "For such a low budget, the direction and cinematography look amazing.",
            "Barker clearly understands horror, his directing is confident and stylish.",
            "Proof that a small indie production can beat big studio horror.",
            "The camera work and lighting create a very distinct visual style.",
            "I followed the director on YouTube and this debut exceeded my expectations.",
        ],
        "neg": [
            "The direction feels like a stretched out YouTube short.",
            "You can tell it is a first feature, the direction is messy at times.",
            "The low budget shows in the cinematography and editing.",
        ],
    },
    "themes": {
        "pos": [
            "Underneath the horror it is a sharp commentary on consent and entitlement.",
            "The film deconstructs the nice guy trope in a smart and uncomfortable way.",
            "It is a metaphor for toxic relationships and controlling love.",
            "The movie says a lot about obsession, loneliness and wanting love without risk.",
            "The way it treats agency and consent is surprisingly thoughtful.",
            "It works as a dark allegory about possessive love and self esteem.",
        ],
        "neg": [
            "The themes about consent are handled clumsily and feel underdeveloped.",
            "It wants to say something about toxic love but never commits to it.",
            "The social commentary is too on the nose.",
        ],
    },
    "theater": {
        "pos": [
            "Seeing it with a packed theater crowd made the experience even better.",
            "The audience was screaming and laughing together, a great cinema experience.",
            "Watch it in a full theater, the crowd reactions are part of the fun.",
            "I went with friends and we talked about it for hours afterwards.",
            "Best theater experience I have had in a long time.",
        ],
        "neg": [
            "The crowd in my theater laughed at everything, which ruined the mood.",
            "I saw it in an empty cinema and it felt much less effective.",
            "The hype from the audience made it feel overrated.",
        ],
    },
}

# Aspek digabung menjadi 6 kelompok laten (topik "sebenarnya" pada data dummy)
GROUPS = {
    "acting": ["acting"],
    "horror": ["gore", "scares"],
    "story": ["premise", "ending"],
    "characters": ["characters", "themes"],
    "humor": ["humor"],
    "production": ["director", "theater"],
}
# Preferensi kelompok aspek per platform (bobot pemilihan)
PLATFORM_WEIGHTS = {
    "imdb": {"acting": 1.5, "horror": 1.5, "story": 1.3, "characters": 1.0, "humor": 0.6, "production": 1.1},
    "letterboxd": {"acting": 1.2, "horror": 0.9, "story": 0.8, "characters": 1.5, "humor": 1.7, "production": 0.5},
}
# Probabilitas kalimat positif per aspek (memengaruhi rating)
POS_PROB = {"acting": 0.92, "gore": 0.7, "scares": 0.72, "premise": 0.78, "characters": 0.3,
            "humor": 0.82, "ending": 0.45, "director": 0.85, "themes": 0.8, "theater": 0.8}

OPENERS = [
    "Just got back from the theater.", "Went in blind and came out shaken.",
    "I did not expect much from this one.", "This was my most anticipated horror of the year.",
    "Okay, I need to talk about this.", "Watched it last night.", "Second viewing.",
    "Not sure how I feel yet.", "Finally saw it.", "Heard a lot of hype about this one.",
]
CLOSERS = [
    "Overall a solid horror movie.", "Definitely worth a watch.", "Would recommend to horror fans.",
    "Not for everyone, but I enjoyed it.", "I wanted to like it more.", "Overrated in my opinion.",
    "Can't stop thinking about it.", "A must see.", "Mixed feelings overall.", "Skip it.",
]
LBX_QUIPS = [
    "be careful what you wish for lol", "not me laughing during the scariest scene",
    "nikki deserved better", "bear you absolute coward", "the willow did nothing wrong",
    "girlboss horror", "rewatched immediately", "my heart rate is still up",
    "watched this with my crush, bad idea", "customer service scene should win an oscar",
]
# Frasa benda per aspek untuk kalimat campuran (membuat kosakata tumpang tindih antartopik)
ASPECT_NOUNS = {
    "acting": ["acting", "lead performance", "cast", "Nikki", "the actress", "performances"],
    "gore": ["gore", "blood", "body horror", "violence", "practical effects"],
    "scares": ["jump scares", "sound design", "tension", "atmosphere", "scares"],
    "premise": ["premise", "story", "concept", "wish idea", "plot"],
    "characters": ["characters", "Bear", "writing", "dialogue", "side characters"],
    "humor": ["humor", "comedy", "jokes", "dark comedy", "tone"],
    "ending": ["ending", "third act", "pacing", "finale", "runtime"],
    "director": ["direction", "cinematography", "budget", "camera work", "Curry Barker"],
    "themes": ["themes", "message", "commentary", "metaphor", "relationship"],
    "theater": ["theater crowd", "audience", "cinema experience", "screening", "crowd"],
}
POS_ADJ = ["great", "amazing", "solid", "excellent", "effective", "strong", "fantastic", "really good", "memorable"]
NEG_ADJ = ["weak", "messy", "forgettable", "disappointing", "uneven", "flat", "annoying", "underwhelming"]
MIX_TEMPLATES = [
    "The {n1} was {a1} and the {n2} was {a2}.",
    "{A1} {n1}, {a2} {n2}, but the {n3} felt {a3}.",
    "I thought the {n1} was {a1}, although the {n2} was {a2}.",
    "Between the {n1} and the {n2}, there is a lot to talk about here.",
    "The {n1} and {n2} work together better than expected.",
    "If you care about {n1} more than {n2}, you will probably enjoy it.",
    "The {n1} is {a1}; the {n2}, not so much.",
    "What stayed with me was the {n1}, even more than the {n2}.",
]
# Kalimat konteks pribadi (kosakata acak lintas topik, seperti ulasan nyata)
CTX_WHEN = ["on opening weekend", "last night", "on a Tuesday", "after work", "on my birthday", "on a rainy Sunday",
            "at a late show", "during a matinee", "on a date night", "after finals week"]
CTX_WHO = ["my sister", "my boyfriend", "my girlfriend", "my roommate", "my dad", "two coworkers", "my best friend",
           "my cousin", "a group of friends", "my wife", "my husband", "nobody"]
CTX_WHERE = ["a small local cinema", "an AMC", "the IMAX screen", "a drive in", "home on my laptop",
             "a nearly empty theater", "a sold out screening", "a festival screening", "the mall cinema"]
CTX_EXTRA = ["The trailer gave away way too much.", "The popcorn was overpriced but whatever.",
             "My expectations were sky high after all the TikTok buzz.", "I almost walked out at one point.",
             "My hands were sweating the entire time.", "The guy next to me kept checking his phone.",
             "I usually hate this genre but gave it a shot.", "Reminded me a bit of early Sam Raimi stuff.",
             "My partner covered their eyes half the time.", "I read the reddit thread afterwards.",
             "The poster does not do it justice.", "Bought a ticket on a whim.", "It is going on my top ten list.",
             "Not sure my mom would approve.", "Halloween season pick for sure.", "The rain outside added to the mood.",
             "Spent the drive home arguing about it.", "My little brother loved it more than me."]


def context_sentence(rng):
    if rng.random() < 0.5:
        return (f"I saw it {CTX_WHEN[int(rng.integers(0, len(CTX_WHEN)))]} with "
                f"{CTX_WHO[int(rng.integers(0, len(CTX_WHO)))]} at {CTX_WHERE[int(rng.integers(0, len(CTX_WHERE)))]}.")
    return CTX_EXTRA[int(rng.integers(0, len(CTX_EXTRA)))]


GENERIC = [
    "Honestly I have mixed feelings about this one.", "It was fine, nothing special but not bad either.",
    "My friend recommended it and I see why people talk about it.", "Not what I expected from the trailer.",
    "Pretty good for a Friday night.", "I think people are overhyping it a little.",
    "There is a lot going on and some of it works.", "Hard to put into words, you just have to experience it.",
    "Gave it a chance because of the reviews here.", "I liked some parts and hated others.",
    "This is the kind of horror I want more of.", "Better than most of what came out this year.",
    "Kept me entertained from start to finish.", "I need a few days to process what I just saw.",
    "Wow. Just wow.", "Ten out of ten would be scared again.", "Way better than I thought it would be.",
]
NON_ENGLISH = [
    "Película muy buena, la actuación es increíble y el final me dejó sin palabras.",
    "Filmnya seram banget, aktingnya bagus dan endingnya bikin kaget.",
    "Filme muito assustador, a atriz principal está incrível do começo ao fim.",
    "Un film d'horreur très original, l'actrice principale est formidable.",
    "Ein sehr gruseliger Film mit einer großartigen Hauptdarstellerin.",
    "Film horornya unik, ceritanya tentang permintaan yang jadi kutukan.",
]


def pick_aspects(rng, platform, k):
    w = PLATFORM_WEIGHTS[platform]
    names = list(w)
    p = np.array([w[n] for n in names], dtype=float)
    p /= p.sum()
    return list(rng.choice(names, size=k, replace=False, p=p))


# Variasi leksikal: sinonim acak + kata penguat, agar kosakata tidak terlalu seragam
SYNONYMS = {
    "performance": ["performance", "portrayal", "turn", "work"], "phenomenal": ["phenomenal", "incredible", "stellar", "superb"],
    "terrifying": ["terrifying", "chilling", "horrifying", "frightening"], "brutal": ["brutal", "savage", "relentless", "vicious"],
    "gore": ["gore", "carnage", "splatter"], "bloody": ["bloody", "gory", "messy", "grisly"],
    "scene": ["scene", "sequence", "moment", "bit"], "scenes": ["scenes", "sequences", "moments", "parts"],
    "story": ["story", "narrative", "plot", "storyline"], "premise": ["premise", "setup", "idea", "hook"],
    "original": ["original", "fresh", "unique", "novel"], "clever": ["clever", "smart", "inventive", "witty"],
    "ending": ["ending", "conclusion", "climax", "last stretch"], "funny": ["funny", "hilarious", "amusing", "comical"],
    "creepy": ["creepy", "eerie", "unnerving", "spooky"], "tension": ["tension", "suspense", "dread", "unease"],
    "character": ["character", "protagonist", "guy", "lead"], "dumbest": ["dumbest", "stupidest", "worst", "silliest"],
    "decisions": ["decisions", "choices", "moves", "calls"], "direction": ["direction", "filmmaking", "craft", "vision"],
    "theater": ["theater", "cinema", "auditorium", "theatre"], "audience": ["audience", "crowd", "people", "viewers"],
    "great": ["great", "awesome", "excellent", "terrific"], "weak": ["weak", "poor", "lackluster", "thin"],
    "slow": ["slow", "sluggish", "dragging", "plodding"], "loud": ["loud", "deafening", "noisy", "blaring"],
    "relationship": ["relationship", "romance", "dynamic", "bond"], "love": ["love", "affection", "devotion", "romance"],
    "acting": ["acting", "performances", "casting", "acting work"], "effects": ["effects", "makeup", "prosthetics", "FX"],
}
INTENSIFIERS = ["honestly", "basically", "literally", "absolutely", "kinda", "super", "genuinely", "totally",
                "definitely", "seriously", "lowkey", "personally", "surprisingly", "pretty"]


def vary(sentence, rng):
    words = sentence.split(" ")
    out = []
    for w in words:
        core = w.strip(".,;:!?").lower()
        if core in SYNONYMS and rng.random() < 0.6:
            rep = SYNONYMS[core][int(rng.integers(0, len(SYNONYMS[core])))]
            if w[0].isupper():
                rep = rep.capitalize()
            w = w.replace(w.strip(".,;:!?"), rep)
        out.append(w)
    if rng.random() < 0.35 and len(out) > 3:
        pos = int(rng.integers(1, len(out) - 1))
        out.insert(pos, INTENSIFIERS[int(rng.integers(0, len(INTENSIFIERS)))])
    return " ".join(out)


def html_noise(text, rng):
    """Menambah tag/entitas HTML seperti data mentah IMDb (<br/>, &#39;, &quot;)."""
    text = text.replace("'", "&#39;") if rng.random() < 0.8 else text
    return text


def build_review(rng, platform):
    if rng.random() < 0.18:   # ulasan umum/campuran (calon outlier BERTopic)
        n = int(rng.integers(2, 5)) if platform == "imdb" else int(rng.integers(1, 3))
        idx = rng.choice(len(GENERIC), size=n, replace=False)
        text = " ".join([GENERIC[i] for i in idx] + [context_sentence(rng) for _ in range(int(rng.integers(0, 3)))])
        return html_noise(text, rng), float(np.clip(round(rng.normal(7, 1.8)), 1, 10)), "generic"
    if platform == "imdb":
        k = rng.choice([1, 2, 3], p=[0.25, 0.45, 0.30])
        n_sent = int(rng.integers(3, 9))
    else:
        k = rng.choice([1, 2], p=[0.6, 0.4])
        n_sent = int(rng.integers(1, 5))
    groups = pick_aspects(rng, platform, k)
    aspects = [m for g in groups for m in GROUPS[g]]
    primary_group = GROUPS[groups[0]]
    primary = primary_group[0]
    sentences, polarity = [], []
    for i in range(n_sent):
        u = rng.random()
        if i > 0 and u < 0.16:                      # kalimat umum
            sentences.append(GENERIC[int(rng.integers(0, len(GENERIC)))])
            continue
        if i > 0 and u < 0.36:                      # kalimat konteks pribadi
            sentences.append(context_sentence(rng))
            continue
        if i > 0 and u < 0.62:                      # kalimat campuran lintas aspek
            names = list(ASPECT_NOUNS)
            picks = [aspects[int(rng.integers(0, len(aspects)))]] + list(rng.choice(names, size=2, replace=False))
            ns = [ASPECT_NOUNS[a][int(rng.integers(0, len(ASPECT_NOUNS[a])))] for a in picks]
            pol = [rng.random() < POS_PROB[a] for a in picks]
            adj = [(POS_ADJ if q else NEG_ADJ)[int(rng.integers(0, 8))] for q in pol]
            tpl = MIX_TEMPLATES[int(rng.integers(0, len(MIX_TEMPLATES)))]
            sent = tpl.format(n1=ns[0], n2=ns[1], n3=ns[2], a1=adj[0], a2=adj[1], a3=adj[2], A1=adj[0].capitalize())
            sentences.append(sent[0].upper() + sent[1:])
            polarity.append(1 if pol[0] else -1)
            continue
        if i == 0 or rng.random() < 0.45:
            a = primary_group[int(rng.integers(0, len(primary_group)))]
        else:
            a = aspects[int(rng.integers(0, len(aspects)))]
        pos = rng.random() < POS_PROB[a]
        bank = ASPECTS[a]["pos" if pos else "neg"]
        sentences.append(bank[int(rng.integers(0, len(bank)))])
        polarity.append(1 if pos else -1)
    sentences = [vary(x, rng) for x in dict.fromkeys(sentences)]  # hapus kalimat ganda + variasi
    score = np.mean(polarity) if polarity else 0.0
    if platform == "imdb":
        if rng.random() < 0.5:
            sentences.insert(0, OPENERS[int(rng.integers(0, len(OPENERS)))])
        if rng.random() < 0.5:
            sentences.append(CLOSERS[int(rng.integers(0, len(CLOSERS)))])
        sep = "<br/><br/>" if rng.random() < 0.5 else " "
        text = sep.join(sentences)
    else:
        if rng.random() < 0.35:
            sentences.append(LBX_QUIPS[int(rng.integers(0, len(LBX_QUIPS)))])
        text = " ".join(sentences)
        if rng.random() < 0.3:
            text = text.lower()
    # rating 1-10 dari polaritas + noise
    r10 = 6.5 + 3.0 * score + rng.normal(0, 1.2)
    r10 = float(np.clip(round(r10), 1, 10))
    return html_noise(text, rng), r10, primary


def random_date(rng, start=date(2025, 9, 1), end=date(2026, 9, 26)):
    return start + timedelta(days=int(rng.integers(0, (end - start).days)))


def make_imdb(rng):
    rows = []
    n_noise_lang, n_empty, n_dup = 9, 4, 12
    n_main = N_IMDB - n_noise_lang - n_empty - n_dup
    for i in range(n_main):
        text, r10, primary = build_review(rng, "imdb")
        rows.append(dict(_primary=primary, text=text, authorRating=int(r10)))
    for i in range(n_noise_lang):
        rows.append(dict(_primary="non_english", text=NON_ENGLISH[i % len(NON_ENGLISH)],
                         authorRating=int(rng.integers(6, 11))))
    for i in range(n_empty):
        rows.append(dict(_primary="empty", text="" if i % 2 == 0 else "   ", authorRating=int(rng.integers(1, 11))))
    for i in range(n_dup):
        src = rows[int(rng.integers(0, n_main))]
        rows.append(dict(src))
    rng.shuffle(rows)
    out = []
    for i, r in enumerate(rows, start=1):
        rid = f"DUMMY-rw{100000 + i:07d}"
        uid = f"DUMMY-ur{500000 + int(rng.integers(0, 99999)):07d}"
        up = int(rng.integers(0, 900))
        out.append({
            "authorRating": r["authorRating"],
            "authorUrl": f"https://example.com/DUMMY/imdb/user/{uid}/",
            "authorUserId": uid,
            "helpfulnessDownVotes": int(rng.integers(0, max(1, up // 2 + 1))),
            "helpfulnessUpVotes": up,
            "id": rid,
            "reportingLink": f"https://example.com/DUMMY/imdb/report/{rid}",
            "reviewPageUrl": f"https://example.com/DUMMY/imdb/review/{rid}/",
            "spoiler": bool(rng.random() < 0.12),
            "submissionDate": random_date(rng).isoformat(),
            "summary": f"[DUMMY] review {i}",
            "text": r["text"],
        })
    return pd.DataFrame(out)


def make_letterboxd(rng):
    rows = []
    n_noise_lang, n_empty, n_dup = 6, 3, 6
    n_main = N_LBX - n_noise_lang - n_empty - n_dup
    for i in range(n_main):
        text, r10, primary = build_review(rng, "letterboxd")
        r5 = max(0.5, min(5.0, round(r10 / 2 * 2) / 2 + rng.choice([-0.5, 0, 0, 0.5])))
        rows.append(dict(reviewtext=text, rating=r5))
    for i in range(n_noise_lang):
        rows.append(dict(reviewtext=NON_ENGLISH[(i + 2) % len(NON_ENGLISH)], rating=float(rng.choice([3.5, 4, 4.5]))))
    for i in range(n_empty):
        rows.append(dict(reviewtext="", rating=float(rng.choice([2, 3, 4]))))
    for i in range(n_dup):
        rows.append(dict(rows[int(rng.integers(0, n_main))]))
    rng.shuffle(rows)
    out = []
    for i, r in enumerate(rows, start=1):
        uname = f"dummy_user_{i:04d}"
        out.append({
            "filmtitle": "Obsession",
            "filmyear": 2025,
            "reviewer": f"DUMMY Reviewer {i:04d}",
            "username": uname,
            "rating": float(r["rating"]),
            "reviewtext": r["reviewtext"],
            "isspoiler": bool(rng.random() < 0.08),
            "likes": int(rng.integers(0, 2500)),
            "reviewdate": random_date(rng).isoformat(),
            "reviewurl": f"https://example.com/DUMMY/letterboxd/{uname}/film/obsession-2025/",
        })
    return pd.DataFrame(out)


def main():
    rng = np.random.default_rng(SEED)
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    imdb = make_imdb(rng)
    lbx = make_letterboxd(rng)
    imdb.to_csv(OUT_DIR / "DUMMY_imdb_obsession_reviews.csv", index=False)
    lbx.to_csv(OUT_DIR / "DUMMY_letterboxd_obsession_reviews.csv", index=False)
    (OUT_DIR / "BACA_SAYA_DUMMY.txt").write_text(
        "Folder ini berisi DATA DUMMY (sintetis) untuk menguji program.\n"
        "Bukan data ulasan asli. Ganti dengan CSV Apify asli di data/raw/ dan set\n"
        "USE_DUMMY = False di config.py.\n", encoding="utf-8")
    print(f"IMDb dummy      : {len(imdb)} baris -> {OUT_DIR/'DUMMY_imdb_obsession_reviews.csv'}")
    print(f"Letterboxd dummy: {len(lbx)} baris -> {OUT_DIR/'DUMMY_letterboxd_obsession_reviews.csv'}")


if __name__ == "__main__":
    main()
