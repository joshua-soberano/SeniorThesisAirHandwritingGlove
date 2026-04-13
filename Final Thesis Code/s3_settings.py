"""
m5_common.py — Shared functions for m5_train.py and m5_eval.py
--------------------------------------------------------------
Contains: constants, data loading, model definition, loss,
          dataset/collate, decode+evaluate, save functions.
Import this module from m5_train.py and m5_eval.py.
"""

import os
import sys
import copy
import math
import random
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

# decoder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m5_decoder import BeamDecoder, LanguageModel

# =============================================================================
# CONSTANTS
# =============================================================================

RANDOM_SEED         = 42
TRAIN_RATIO         = 0.70
VAL_RATIO           = 0.15
TEST_RATIO          = 0.15
EARLY_STOP_PATIENCE = 10
INPUT_DIM           = 9
HIDDEN_SIZE         = 128
N_LAYERS            = 2
DROPOUT             = 0.3

# boundary state indices — must match m5_decoder.py
START      = 0
INSIDE     = 1
END        = 2
TRANSITION = 3
N_BOUNDARY = 4

# synthetic word builder
MIN_TRANSITION_INTERVALS = 1
MAX_TRANSITION_INTERVALS = 5

# 500 common English words — filtered at runtime to only use available letters
COMMON_WORDS = [
    "the","be","to","of","and","a","in","that","have","it",
    "for","not","on","with","he","as","you","do","at","this",
    "but","his","by","from","they","we","say","her","she","or",
    "an","will","my","one","all","would","there","their","what",
    "so","up","out","if","about","who","get","which","go","me",
    "when","make","can","like","time","no","just","him","know",
    "take","people","into","year","your","good","some","could",
    "them","see","other","than","then","now","look","only","come",
    "its","over","think","also","back","after","use","two","how",
    "our","work","first","well","way","even","new","want","because",
    "any","these","give","day","most","us","great","between","need",
    "large","often","hand","high","place","hold","turn","here","why",
    "help","ask","men","read","land","different","home","move","try",
    "kind","hand","picture","again","change","off","play","spell","air",
    "away","animal","house","point","page","letter","mother","answer",
    "found","study","still","learn","should","world","those","never",
    "next","below","add","food","plant","last","school","father","keep",
    "tree","never","start","city","earth","eye","light","thought","head",
    "under","story","saw","left","dont","few","while","along","might",
    "close","something","seem","side","been","open","begin","life","always",
    "those","both","paper","together","got","group","often","run","important",
    "until","children","side","feet","car","mile","night","walk","white",
    "sea","began","grow","took","river","four","carry","state","once",
    "book","hear","stop","without","second","late","miss","idea","enough",
    "eat","face","watch","far","indian","real","almost","let","above",
    "girl","sometimes","mountain","cut","young","talk","soon","list","song",
    "being","leave","family","body","music","color","stand","sun","questions",
    "fish","area","mark","horse","birds","problem","complete","room","knew",
    "since","ever","piece","told","usually","friends","easy","heard","order",
    "red","door","sure","become","top","ship","across","today","during",
    "short","better","best","however","low","hours","black","products","happened",
    "whole","measure","remember","early","waves","reached","listen","wind","rock",
    "space","covered","fast","several","hold","himself","toward","five","step",
    "morning","passed","vowel","true","hundred","against","pattern","numeral",
    "table","north","slowly","money","map","farm","pulled","draw","voice",
    "power","town","fine","drive","led","cry","dark","machine","note","waited",
    "plan","figure","star","box","noun","field","rest","able","pound","done",
    "beauty","drive","stood","contain","front","teach","final","gave","green",
    "quick","develop","ocean","warm","free","minute","strong","special","mind",
    "behind","clear","tail","produce","fact","street","inch","lot","nothing",
    "course","stay","wheel","full","force","blue","object","decide","surface",
    "deep","moon","island","foot","yet","busy","test","record","boat","common",
    "gold","possible","plane","age","dry","wonder","laugh","thousand","ago",
    "ran","check","game","shape","miss","brought","heat","snow","bed","bring",
    "sit","perhaps","fill","east","weight","language","among","present","heavy",
    "leader","dog","race","south","west","lay","pass","exact","remain","dress",
    "cat","ring","fall","floor","valley","cent","natural","log","intend","case",
    "middle","kill","son","lake","moment","scale","loud","spring","observe",
    "child","straight","consonant","nation","dictionary","milk","speed","method",
    "organ","pay","age","section","dress","cloud","surprise","quiet","stone",
    "tiny","climb","cool","design","poor","lot","experiment","bottom","key",
    "iron","single","stick","flat","twenty","skin","smile","crease","hole",
    "trade","melody","trip","office","receive","row","mouth","exact","symbol",
    "die","least","trouble","shout","except","wrote","seed","tone","join",
    "suggest","clean","break","lady","yard","rise","bad","blow","oil","blood",
    "touch","grew","cent","mix","team","wire","cost","lost","brown","wear",
    "garden","equal","sent","choose","fell","fit","flow","fair","bank","collect",
    "save","control","decimal","gentle","woman","captain","practice","separate",
    "difficult","doctor","please","protect","noon","crop","modern","element",
    "hit","student","corner","party","supply","bone","rail","imagine","provide",
    "agree","thus","capital","chair","danger","fruit","rich","thick","soldier",
    "process","operate","guess","necessary","sharp","wing","create","neighbor",
    "wash","bat","rather","crowd","corn","compare","poem","string","bell","depend",
    "meat","rub","tube","famous","dollar","stream","fear","sight","thin","triangle",
    "planet","hurry","chief","colony","clock","mine","tie","enter","major","fresh",
    "search","send","yellow","gun","allow","print","dead","spot","desert","suit",
    "current","lift","rose","continue","block","chart","hat","sell","success",
    "company","subtract","event","particular","deal","swim","term","opposite",
    "wife","shoe","shoulder","spread","arrange","camp","invent","cotton","born",
    "determine","quart","nine","truck","noise","level","chance","gather","shop",
    "stretch","throw","shine","property","column","molecule","select","wrong",
    "gray","repeat","require","broad","prepare","salt","nose","plural","anger",
    "claim","syllable","felt","sudden","happen","lead","able","plain","paragraph",
    # batch 2 — common English words
    "act","aid","aim","art","bay","bit","bow","bus","buy","cap",
    "cup","cut","ear","egg","end","era","fee","fly","gap","gel",
    "gut","hub","ice","jam","jar","jaw","jet","joy","kid","kit",
    "lab","lap","law","leg","lid","lip","map","mat","mud","net",
    "nod","nut","oak","odd","opt","orb","ore","owe","own","pan",
    "pat","pen","pie","pig","pin","pit","pod","pop","pot","pro",
    "pub","pun","put","raw","ray","rod","rug","rum","rut","sap",
    "sat","saw","set","sew","shy","sin","sip","sir","ski","sky",
    "sly","sob","sod","sow","spa","spy","sub","sue","sum","sun",
    "tab","tan","tap","tar","tax","tea","tip","toe","toy","tug",
    "urn","van","vat","vet","via","vow","wax","web","wed","wit",
    "woe","wok","yam","yap","yew","zap","zen","zoo","arc","ash",
    "awe","axe","aye","ban","bar","bay","beg","bin","bog","bud",
    "bug","bun","cob","cod","cog","cop","cot","cub","cud","dab",
    "dag","dam","dip","dob","dug","dun","eel","elm","emu","eve",
    "ewe","fad","fag","fan","fax","fib","fig","fin","foe","fog",
    "fop","fug","fun","fur","gag","gap","gel","gem","gin","gnu",
    "gob","god","got","gum","gust","gym","had","hag","ham","hap",
    "hay","hem","hen","hew","hex","hid","hip","his","hob","hog",
    "hop","hot","how","hub","hug","hum","hun","hut","icy","ill",
    "imp","inn","ion","ire","ivy","jab","jag","jig","job","jog",
    "jot","jow","joy","jug","jut","keg","kin","lap","lax","lea",
    "lee","let","lob","lop","lot","low","lug","nun","nip","nit",
    "nob","nod","nor","now","oar","off","oil","old","one","ooze",
    "orb","out","owl","ox","pad","pap","paw","pea","peg","pig",
    "pit","ply","pod","poi","polo","pond","pony","pooh","pose","pose",
    # batch 3 — medium length common words
    "able","arch","army","aunt","bake","ball","barn","base","bath","beam",
    "bean","bear","beat","beef","beer","belt","bend","bike","bill","bind",
    "bird","bite","bold","bolt","bomb","bond","book","boom","boot","bore",
    "bowl","bull","burn","cage","cake","calm","came","cane","card","care",
    "cart","cash","cast","cave","cell","chef","chin","chip","chop","cite",
    "clap","clay","clip","club","clue","coal","coat","code","coil","coin",
    "cold","cope","core","cork","crew","cure","curl","dare","dark","dart",
    "dash","data","date","dawn","deal","dean","deck","deed","deer","deny",
    "desk","dial","diet","dine","dire","disk","dive","dock","dome","door",
    "dose","dove","draw","drip","drop","drum","dual","dune","dusk","dust",
    "each","earn","ease","edge","edit","emit","envy","epic","even","evil",
    "exam","exit","fade","fame","farm","fast","fate","fawn","feed","feel",
    "fend","fern","file","film","find","fine","fire","firm","fish","fist",
    "flag","flaw","flea","flew","flex","flip","flog","foam","fold","fond",
    "font","ford","fore","fork","form","fort","foul","four","fowl","fray",
    "fuel","fume","fund","fuse","gale","gall","gaze","gear","gene","gild",
    "gill","gist","glee","glen","glow","glue","goal","goat","gore","gown",
    "grab","gram","grin","grip","grit","gust","hack","hail","hall","halt",
    "hang","hard","hare","harm","harp","hate","haul","hawk","haze","heal",
    "heap","hear","heel","helm","help","hemp","herb","herd","hero","hide",
    "hill","hint","hire","hoax","hold","home","honk","hood","hoop","hope",
    "horn","host","howl","hull","hump","hunt","hurl","hymn","idea","idle",
    "imam","inch","itch","item","jade","jail","jest","jibe","jolt","junk",
    "just","keen","kept","kern","knot","lace","laid","lake","lame","lard",
    "lark","lash","last","late","laud","lawn","lean","leap","lend","lent",
    "levy","lick","lift","lime","lint","lion","list","live","load","loan",
    "loft","lone","long","loom","loot","lore","lout","love","lull","lump",
    "lure","lurk","lust","mace","made","mail","main","mane","mare","mark",
    "mars","mash","mast","mate","maul","maze","meal","mean","meet","melt",
    "memo","mend","mere","mesh","mild","mill","mint","mire","mist","moan",
    "moat","mode","monk","mood","moor","mope","more","most","moth","move",
    "much","mule","muse","must","nail","name","near","neat","need","nest",
    "news","next","nigh","node","none","norm","note","numb","obey","once",
    "open","oral","oven","over","pace","page","paid","pain","pair","pale",
    "pall","palm","park","part","past","path","pave","peak","peel","peer",
    "pelt","perk","pest","pick","pile","pine","pipe","plea","plot","plow",
    "ploy","plum","plus","poke","poll","pool","pore","port","pose","post",
    "pour","pray","prep","prey","prod","prop","prow","pull","pump","pure",
    "push","rack","rage","raid","rail","rain","rake","ramp","rank","rant",
    "raze","read","real","reap","reef","reel","rein","rely","rent","rile",
    "riot","risk","rite","roam","roar","robe","role","roof","root","rope",
    "rote","rout","rule","rush","rust","safe","sage","sake","sale","sand",
    "sane","sang","sash","scan","scar","seal","seam","sear","sect","self",
    "sell","semi","shed","shin","ship","shop","shot","show","shut","sick",
    "sift","sign","silk","sill","sing","sink","site","size","skew","slab",
    "slap","slew","slim","slip","slot","slow","slug","slum","snap","soak",
    "soar","sock","soft","soil","sole","some","soot","sort","soul","soup",
    "sour","span","spar","spec","spin","spit","stab","stag","stem","step",
    "stew","stir","stop","stub","stun","such","sulk","sunk","sure","surf",
    "swam","swap","swat","sway","tack","tale","tame","tang","task","taut",
    "tell","tent","text","than","that","them","thud","tick","tide","tied",
    "tier","tile","time","tint","tire","toil","told","toll","tomb","tore",
    "torn","toss","tour","town","tram","trap","tray","trim","trio","trod",
    "tuck","tuft","tune","turf","turn","tusk","type","ugly","unit","upon",
    "urge","used","vale","vary","vast","veil","vein","very","vest","view",
    "vine","void","volt","vote","wade","wage","wail","wait","wake","wall",
    "wane","ward","ware","warn","warp","wart","wave","weak","weal","wean",
    "weld","well","wend","went","whet","whip","wide","wile","will","wilt",
    "wind","wine","wink","wipe","wise","wish","wolf","word","wore","worm",
    "wort","wrap","wren","writ","yell","yoga","yore","your","zeal","zero",
    "zone","zoom","zest","also","area","army","away","back","been","body",
    "call","came","cant","care","city","come","copy","dont","down","draw",
    "drop","each","else","even","ever","eyes","face","felt","find","fine",
    "fire","five","form","four","from","full","give","gone","good","hand",
    "hard","have","head","hear","help","here","high","hill","home","hope",
    "hour","huge","hurt","into","just","keep","kind","knew","know","land",
    "last","late","left","less","life","like","line","live","long","look",
    "made","make","many","mean","meet","mind","mine","more","move","much",
    "must","near","neck","next","nice","nine","none","note","once","only",
    "open","over","part","past","pick","plan","play","plus","pull","push",
    "read","rest","ride","road","rock","room","rose","runs","said","same",
    "seem","seen","self","send","shed","show","side","sing","size","skip",
    "slow","snow","some","song","soon","sort","soul","spot","step","stop",
    "such","sure","take","talk","tall","task","tell","tend","test","than",
    "them","then","they","thin","this","thou","thus","till","time","told",
    "toll","took","tops","torn","tree","true","turn","type","used","very",
    "view","wait","walk","want","warm","ways","week","when","whom","wide",
    "wife","wild","wipe","wish","with","word","work","worn","your","zero",
    # batch 4 — longer and less common but valid words
    "about","above","abuse","actor","acute","admit","adopt","adult","after","again",
    "agent","agree","ahead","alarm","album","alert","align","alike","alive","alley",
    "allow","alone","along","aloud","alter","angel","anger","angle","ankle","annex",
    "apart","apple","apply","arena","argue","arise","armor","aroma","array","aside",
    "asset","atlas","attic","audio","audit","avoid","await","awake","award","aware",
    "badly","basic","basis","batch","begin","being","below","bench","bible","black",
    "blade","blame","bland","blank","blast","blaze","bleed","blend","bless","blind",
    "block","blond","bloom","blown","board","boast","bonus","boost","bound","brain",
    "brand","brave","bread","break","breed","brick","bride","brief","bring","broad",
    "broke","brook","broom","brown","brush","budge","build","built","burst","buyer",
    "cabin","cable","cargo","carry","catch","cause","cease","chain","chair","chalk",
    "chaos","chase","cheap","cheat","check","cheek","cheer","chess","chest","chief",
    "child","choir","chord","civic","civil","claim","clash","class","clean","clear",
    "clerk","click","cliff","climb","cling","clock","close","cloth","cloud","coach",
    "coast","color","comic","comma","coral","corps","could","count","court","cover",
    "crack","craft","crane","crash","crazy","cream","creek","crime","cross","crowd",
    "crown","crush","curve","cycle","daily","dance","debut","delay","delta","dense",
    "depth","devil","disco","dizzy","dodge","doubt","dough","draft","drain","drama",
    "drank","drawn","dread","dream","dried","drink","drive","drove","dying","eager",
    "early","earth","eight","elite","email","empty","enemy","enjoy","enter","equal",
    "error","essay","event","every","exact","exist","extra","faint","faith","false",
    "fancy","fatal","fault","feast","fence","fever","fewer","field","fifth","fifty",
    "fight","filed","final","first","fixed","flame","flash","fleet","flesh","flies",
    "flood","floor","flour","fluid","flush","focus","force","forge","forth","forum",
    "found","frame","frank","fraud","freed","front","froze","funds","funny","gains",
    "giant","given","glass","gleam","globe","gloom","glory","glove","going","grace",
    "grade","grand","grant","graph","grasp","grass","grave","great","greed","grief",
    "groan","groin","gross","group","grove","grown","guard","guide","guild","guile",
    "guise","gulch","habit","happy","heart","heavy","hence","herbs","hinge","horse",
    "hotel","house","human","humor","hurry","ideal","image","imply","imply","index",
    "inner","input","irony","issue","joint","joker","judge","juice","juicy","juror",
    "knife","knock","known","label","lance","large","laser","later","laugh","layer",
    "learn","lease","leave","legal","lemon","level","light","limit","linen","liver",
    "local","lodge","logic","loose","lover","lower","lucky","lunch","lying","magic",
    "major","maker","manor","march","match","mayor","media","merit","metal","might",
    "minor","minus","mixed","model","money","month","moral","motor","mount","mouse",
    "mouth","movie","mural","music","naive","nerve","never","night","noble","noise",
    "north","noted","novel","nurse","nylon","occur","ocean","offer","often","olive",
    "onset","order","other","ought","outer","owner","oxide","ozone","paint","panel",
    "panic","paper","party","patch","pause","peace","pearl","penal","penny","phase",
    "phone","photo","piano","pilot","pitch","pixel","place","plaid","plain","plant",
    "plate","plaza","plead","pluck","plumb","plume","plunge","point","polar","posed",
    "power","press","price","pride","prime","print","prior","prize","probe","prone",
    "proof","prose","proud","prove","proxy","psalm","pulse","pupil","queen","query",
    "quest","queue","quick","quiet","quite","quota","quote","radar","radio","raise",
    "rally","range","rapid","ratio","reach","ready","realm","rebel","refer","reign",
    "relay","renew","reply","resin","rider","ridge","rifle","right","rigid","risky",
    "rival","river","robot","rocky","roman","round","route","royal","ruler","rumor",
    "rural","sadly","saint","salad","sauce","scale","scene","score","scout","seize",
    "sense","serve","seven","sever","shade","shaft","shake","shall","shame","shape",
    "share","shark","sharp","sheer","sheet","shelf","shell","shift","shine","shirt",
    "shock","shore","short","shrug","sight","since","sixth","sixty","skill","skull",
    "slain","sleep","slide","slope","small","smart","smell","smile","smoke","solid",
    "solve","sorry","south","space","spare","spark","speak","speed","spend","spill",
    "spoke","spook","sport","spray","squad","staff","stage","stain","stake","stale",
    "stall","stamp","stand","stark","start","state","stays","steam","steel","steep",
    "steer","stern","stick","stiff","still","stomp","stone","stood","store","storm",
    "story","stove","strap","straw","strip","stuck","study","style","sugar","suite",
    "super","surge","swear","sweep","sweet","swept","swift","sword","swore","swung",
    "table","taste","teach","tears","teeth","thank","their","there","these","thick",
    "thing","think","third","those","three","threw","throw","thumb","tiger","tight",
    "timer","tired","title","today","token","total","touch","tough","towel","tower",
    "toxic","trade","trail","train","trait","tramp","trend","trial","tribe","trick",
    "tried","troop","truck","truly","trump","trunk","trust","truth","tumor","tuner",
    "twice","twist","tyrant","under","unify","union","until","upper","upset","urban",
    "usage","usual","utter","valid","value","valve","video","vigor","viral","virus",
    "visit","vista","vital","vivid","vocal","voice","voter","waste","watch","water",
    "wheel","where","which","while","white","whole","whose","widow","witch","witty",
    "woman","women","world","worry","worse","worst","worth","would","wound","write",
    "wrong","wrote","yacht","yield","young","youth","zebra","zoned",
]


# =============================================================================
# PARSE SINGLE-LETTER SEQUENCE FILES
# =============================================================================

def parse_sequences_file(path: str) -> list:
    strokes   = []
    in_stroke = False
    rows      = []
    letter    = os.path.basename(path).split('_')[0].upper()

    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('%'):
                if 'Stroke:' in line:
                    if in_stroke and rows:
                        s = _build_stroke(rows, letter)
                        if s: strokes.append(s)
                    rows      = []
                    in_stroke = True
                elif 'Letter:' in line:
                    try: letter = line.split(':')[1].strip().upper()
                    except: pass
                continue
            if line.startswith('IntervalStart'): continue
            if in_stroke and line.strip():
                rows.append(line.strip())

    if in_stroke and rows:
        s = _build_stroke(rows, letter)
        if s: strokes.append(s)
    return strokes


def _build_stroke(rows: list, letter: str) -> dict:
    X_list = []
    for row in rows:
        parts = row.split(',')
        if len(parts) < 11: continue
        try:
            probs = [float(parts[i]) for i in range(2, 11)]
        except (ValueError, IndexError):
            continue
        X_list.append(probs)
    if not X_list: return None
    return {'X': np.array(X_list, dtype=np.float32), 'letter': letter}


def load_all_strokes(directory: str) -> dict:
    files = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith('_sequences.txt')
    ])
    if not files:
        print(f"No _sequences.txt files found in: {directory}")
        sys.exit(1)

    strokes_by_letter = {}
    total = 0
    for fpath in files:
        strokes = parse_sequences_file(fpath)
        for s in strokes:
            l = s['letter']
            strokes_by_letter.setdefault(l, []).append(s)
            total += 1
        print(f"  {os.path.basename(fpath)}: {len(strokes)} stroke(s)")

    print(f"\nTotal strokes: {total}")
    print(f"Letters available: {sorted(strokes_by_letter.keys())}")
    return strokes_by_letter


# =============================================================================
# BOUNDARY STATE LABELLING
# =============================================================================

def label_boundary_states(K: int) -> np.ndarray:
    """
    Assign 4-class boundary states to K intervals of one letter stroke.
    First 10% → START, middle 80% → INSIDE, last 10% → END.
    Minimum 1 interval per class guaranteed.
    """
    b = np.full(K, INSIDE, dtype=np.int64)
    n_start = max(1, int(round(0.10 * K)))
    n_end   = max(1, int(round(0.10 * K)))
    # make sure start + end don't exceed K
    if n_start + n_end >= K:
        n_start = 1
        n_end   = 1
    b[:n_start]  = START
    b[K-n_end:]  = END
    return b


# =============================================================================
# SYNTHETIC WORD SEQUENCE BUILDER
# =============================================================================

def get_transition_intervals(strokes_by_letter: dict,
                             n: int,
                             rng: random.Random) -> np.ndarray:
    """
    Sample n real 100ms intervals from random letter strokes to use as
    transition intervals between letters. Each interval is one row (9,)
    from a real recorded stroke — captures real deceleration/repositioning
    motion rather than a synthetic REST vector.
    """
    all_strokes = [s for strokes in strokes_by_letter.values() for s in strokes]
    rows = []
    for _ in range(n):
        stroke = rng.choice(all_strokes)
        K      = stroke['X'].shape[0]
        row_idx = rng.randint(0, K - 1)
        rows.append(stroke['X'][row_idx])
    return np.stack(rows, axis=0).astype(np.float32)   # (n, 9)


def build_word_sequences(strokes_by_letter: dict,
                         word_list: list,
                         seed: int = RANDOM_SEED,
                         num_repeats: int = 1) -> list:
    """
    For each word in word_list build a synthetic continuous sequence by:
      1. Sampling one stroke per letter from strokes_by_letter
      2. Assigning START/INSIDE/END boundary states to each stroke
      3. Inserting 1-5 real 100ms intervals (sampled from existing strokes)
         between letters as TRANSITION
    Returns list of sample dicts.
    """
    rng     = random.Random(seed)
    samples = []

    buildable = [w for w in word_list
                 if all(l.upper() in strokes_by_letter for l in w)
                 and len(w) >= 2]

    for word in buildable:
        letters = list(word.upper())
        for rep in range(num_repeats):
            X_parts  = []
            b_parts  = []
            offsets  = []
            pos      = 0

            for i, ltr in enumerate(letters):
                stroke = rng.choice(strokes_by_letter[ltr])
                K      = stroke['X'].shape[0]

                X_parts.append(stroke['X'])
                b_parts.append(label_boundary_states(K))
                pos += K
                offsets.append(pos - 1)

                if i < len(letters) - 1:
                    n_trans  = rng.randint(MIN_TRANSITION_INTERVALS,
                                           MAX_TRANSITION_INTERVALS)
                    trans_X  = get_transition_intervals(strokes_by_letter, n_trans, rng)
                    trans_b  = np.full(n_trans, TRANSITION, dtype=np.int64)
                    X_parts.append(trans_X)
                    b_parts.append(trans_b)
                    pos += n_trans

            X_concat = np.concatenate(X_parts, axis=0).astype(np.float32)
            b_concat = np.concatenate(b_parts, axis=0).astype(np.int64)

            samples.append({
                'X'       : X_concat,
                'boundary': b_concat,
                'word'    : word.upper(),
                'letters' : letters,
                'offsets' : offsets,
            })

    rng.shuffle(samples)
    print(f"Built {len(samples)} word sequences "
          f"({len(buildable)} unique words × {num_repeats} repeat(s)).")
    return samples


# =============================================================================
# DATASET & COLLATE
# =============================================================================

def collate_fn(batch):
    Xs, bs, letters_list, offsets_list, lengths = zip(*batch)
    max_len = max(lengths);  B = len(Xs)
    X_pad   = torch.zeros(B, max_len, INPUT_DIM)
    b_pad   = torch.full((B, max_len), TRANSITION, dtype=torch.long)  # pad with TRANSITION

    for i, (x, b_) in enumerate(zip(Xs, bs)):
        k = x.shape[0]
        X_pad[i, :k, :] = x
        b_pad[i, :k]    = b_

    return X_pad, b_pad, list(letters_list), list(offsets_list), \
           torch.tensor(lengths, dtype=torch.long)


# =============================================================================
# MODEL
# =============================================================================

class WordGRU(nn.Module):
    """
    Unidirectional GRU for continuous letter sequence recognition.

    Input  : (B, K, 9) direction probability sequences
    Output :
        letter_logits   : (B, K, n_letters)   raw letter scores at every step
        boundary_logits : (B, K, 4)           raw boundary scores at every step

    At inference: apply softmax to each to get (K, 26) + (K, 4) = (K, 30)
    """

    def __init__(self, n_letters: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = INPUT_DIM,
            hidden_size = HIDDEN_SIZE,
            num_layers  = N_LAYERS,
            dropout     = DROPOUT,
            batch_first = True,
        )
        self.letter_head   = nn.Linear(HIDDEN_SIZE, n_letters)
        self.boundary_head = nn.Linear(HIDDEN_SIZE, N_BOUNDARY)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)  # (B, T, H)

        letter_logits   = self.letter_head(out)    # (B, T, n_letters)
        boundary_logits = self.boundary_head(out)  # (B, T, 4)
        return letter_logits, boundary_logits



# =============================================================================
# CTC MODEL
# =============================================================================

BLANK_IDX = 0   # CTC blank token is index 0; letter indices are shifted by +1

class WordGRU_CTC(nn.Module):
    """
    Unidirectional GRU trained with CTC loss.
    Output is (B, T, n_letters+1) — index 0 is blank, 1..n_letters are letters.
    No boundary head — segmentation is implicit via CTC blank token.
    At inference: log_softmax over output, feed to CTC greedy/beam decoder.
    """

    def __init__(self, n_letters: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = INPUT_DIM,
            hidden_size = HIDDEN_SIZE,
            num_layers  = N_LAYERS,
            dropout     = DROPOUT,
            batch_first = True,
        )
        # output size = n_letters + 1 blank token
        self.output_head = nn.Linear(HIDDEN_SIZE, n_letters + 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)   # (B, T, H)
        return self.output_head(out)                           # (B, T, n_letters+1)


def compute_ctc_loss(logits, letters_list, lengths, device):
    """
    logits   : (B, T, n_letters+1) raw logits — blank=0, letters=1..n
    letters_list : list of (L,) long tensors — letter indices 0-based
                   these get shifted to 1-based to leave 0 for blank
    lengths  : (B,) int tensor — real sequence lengths
    Returns scalar CTC loss.
    """
    # CTC expects (T, B, C) log_probs
    log_probs    = torch.log_softmax(logits, dim=-1)  # (B, T, C)
    log_probs    = log_probs.permute(1, 0, 2)         # (T, B, C)

    # shift letter targets from 0-based to 1-based (blank=0)
    targets      = torch.cat([l.to(device) + 1 for l in letters_list])
    target_lens  = torch.tensor([len(l) for l in letters_list],
                                 dtype=torch.long, device=device)

    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, reduction='mean', zero_infinity=True)
    return ctc_loss(log_probs, targets, lengths.to(device), target_lens)


def ctc_greedy_decode(logits: np.ndarray, letter_classes: list) -> str:
    """
    CTC greedy decode: argmax at each timestep, collapse blanks and repeats.
    logits : (T, n_letters+1) — blank=0, letters=1..n
    Returns decoded string.
    """
    indices = np.argmax(logits, axis=-1)   # (T,)
    decoded = []
    prev    = -1
    for idx in indices:
        if idx != prev:
            if idx != BLANK_IDX:
                decoded.append(letter_classes[idx - 1])  # shift back to 0-based
            prev = idx
    return ''.join(decoded)


def train_model_ctc(model, train_loader, val_loader,
                    n_epochs, lr, device, letter_classes):
    """
    Train WordGRU_CTC with CTCLoss.
    No boundary loss, no lambda — CTC handles segmentation implicitly.
    Returns best model by val loss.
    """
    def _cer(t, p):
        m, n = len(t), len(p)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1): dp[i][0] = i
        for j in range(n+1): dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                dp[i][j] = dp[i-1][j-1] if t[i-1]==p[j-1] \
                            else 1+min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])
        return dp[m][n] / max(m, 1)

    optimizer     = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    best_weights  = copy.deepcopy(model.state_dict())
    patience_ctr  = 0

    model.to(device)

    print(f"\n  Training GRU-CTC | epochs={n_epochs} lr={lr}")
    print(f"  {'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  "
          f"{'Val Word Acc':>14}  {'Val CER':>10}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*14}  {'-'*10}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        tr_loss = 0.0

        for X_pad, b_pad, letters_list, offsets_list, lengths in train_loader:
            X_pad   = X_pad.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            logits = model(X_pad, lengths)
            loss   = compute_ctc_loss(logits, letters_list, lengths, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss += loss.item() * len(lengths)

        tr_loss /= len(train_loader.dataset)

        model.eval()
        val_loss  = 0.0
        val_hits  = 0
        val_total = 0
        val_cer   = 0.0

        with torch.no_grad():
            for X_pad, b_pad, letters_list, offsets_list, lengths in val_loader:
                X_pad   = X_pad.to(device)
                lengths = lengths.to(device)
                logits  = model(X_pad, lengths)
                loss    = compute_ctc_loss(logits, letters_list, lengths, device)
                val_loss += loss.item() * len(lengths)

                for i in range(X_pad.shape[0]):
                    logits_i = logits[i, :lengths[i]].cpu().numpy()
                    decoded  = ctc_greedy_decode(logits_i, letter_classes)
                    true_letters = [letter_classes[j] for j in
                                    letters_list[i].cpu().numpy()]
                    true_word = ''.join(true_letters)
                    val_hits += int(decoded == true_word)
                    val_cer  += _cer(true_word, decoded)
                    val_total += 1

        val_loss /= len(val_loader.dataset)
        val_acc   = val_hits / val_total if val_total > 0 else 0.0
        val_cer_m = val_cer  / val_total if val_total > 0 else 0.0

        print(f"  {epoch:>6}  {tr_loss:>12.4f}  {val_loss:>10.4f}  "
              f"{val_acc:>14.4f}  {val_cer_m:>10.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = copy.deepcopy(model.state_dict())
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_weights)
    print(f"  Best val loss: {best_val_loss:.4f}")
    return model


def ctc_letter_beam_decode(logits: np.ndarray, letter_classes: list,
                           lm=None, beam_width: int = 5,
                           lm_weight: float = 0.3) -> str:
    """
    Letter-level CTC beam search.

    Instead of running the beam at every timestep, runs greedy CTC first
    to identify emission points — timesteps where the argmax transitions
    from blank to a non-blank letter. At each emission point the top-K
    letter probabilities are used to branch the beam.

    This is much faster than timestep-level beam search and more stable
    because branching only happens at genuine letter boundaries, not at
    every interval of a 150-200 timestep sequence.

    logits     : (T, n_letters+1) — blank=0, letters=1..n
    lm         : language model with log_score(letter, context) interface
    beam_width : number of hypotheses to maintain
    lm_weight  : weight of LM score relative to acoustic score
    """
    # softmax for probability extraction
    exp_l = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=-1, keepdims=True)   # (T, n_letters+1)

    indices = np.argmax(logits, axis=-1)   # (T,) argmax path

    # identify emission timesteps — where argmax transitions to a letter
    emission_probs = []   # list of (n_letters,) prob vectors at each emission
    prev = -1
    for t, idx in enumerate(indices):
        if idx != prev:
            if idx != BLANK_IDX:
                # top of a letter peak — record full letter prob distribution
                emission_probs.append(probs[t, 1:])   # drop blank, keep letters
            prev = idx

    if not emission_probs:
        return ''

    # letter-level beam search over emission points
    # each hypothesis: (word_str, log_acoustic_score, lm_context)
    lm_order = getattr(lm, 'lm_order', 1) if lm is not None else 1
    init_ctx = ('_',) * max(0, lm_order - 1)
    beam = [('', 0.0, init_ctx)]   # (word, score, context)

    for ep in emission_probs:
        # ep: (n_letters,) probability vector

        # get top beam_width letters by acoustic probability
        top_k = min(beam_width, len(letter_classes))
        top_indices = np.argsort(ep)[::-1][:top_k]

        new_beam = []
        for word, score, ctx in beam:
            for li in top_indices:
                letter  = letter_classes[li]
                log_p   = math.log(max(float(ep[li]), 1e-10))

                # LM score for this letter given context
                lm_sc = 0.0
                if lm is not None:
                    lm_sc = lm_weight * lm.log_score(letter, ctx)

                new_word  = word + letter
                new_score = score + log_p + lm_sc
                new_ctx   = (ctx + (letter,))[-(lm_order-1):] \
                            if lm_order > 1 else ()
                new_beam.append((new_word, new_score, new_ctx))

        # prune to beam_width by score
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_width]

    # return top hypothesis
    return beam[0][0] if beam else ''


# keep old name as alias for backward compatibility
def ctc_beam_decode(log_probs: np.ndarray, letter_classes: list,
                    lm, beam_width: int = 10,
                    lm_weight: float = 0.3) -> str:
    """Alias for ctc_letter_beam_decode — converts log_probs to logits."""
    return ctc_letter_beam_decode(log_probs, letter_classes,
                                  lm, beam_width, lm_weight)


def decode_and_evaluate_ctc(model, test_samples, letter_classes,
                             word_list, device,
                             avg_letter_intervals=20.0,
                             length_tolerance=2,
                             use_beam=False,
                             beam_width=10,
                             lm=None,
                             lm_weight=0.3):
    """
    Evaluate a CTC model on test_samples.
    Two-pass approach:
      Pass 1 — greedy decode all samples, collect raw decoded strings
      Build confusion matrix from raw decodes vs true words
      Pass 2 — run word correction using confusion matrix as secondary signal
    """
    import time
    model.eval()
    t_start = time.time()

    if lm is None:
        from m5_decoder import LanguageModel as LM
        lm = LM.uniform(letter_classes)

    # --- Pass 1: decode all samples ---
    print(f"\n  Pass 1: decoding {len(test_samples)} samples...")
    raw_decodes = []
    est_lens    = []
    with torch.no_grad():
        for s in test_samples:
            X       = torch.tensor(s['X'][np.newaxis],
                                   dtype=torch.float32).to(device)
            lengths = torch.tensor([s['X'].shape[0]],
                                   dtype=torch.long).to(device)
            logits    = model(X, lengths)
            logits_np = logits[0].cpu().numpy()

            if use_beam:
                log_probs = logits_np - \
                    np.log(np.exp(logits_np).sum(axis=-1, keepdims=True) + 1e-10)
                decoded = ctc_beam_decode(log_probs, letter_classes,
                                          lm, beam_width, lm_weight)
            else:
                decoded = ctc_greedy_decode(logits_np, letter_classes)

            raw_decodes.append(decoded)
            est_lens.append(max(1, round(X.shape[1] / avg_letter_intervals)))

    # --- Build confusion matrix from pass 1 ---
    raw_results = [{'true_word': s['word'], 'decoded_raw': d}
                   for s, d in zip(test_samples, raw_decodes)]
    confusion = build_confusion_matrix(raw_results, letter_classes)
    print(f"  Confusion matrix built from {len(test_samples)} samples.")

    # print top confusions for info
    print(f"  Top letter confusions (true→decoded, excluding correct):")
    conf_pairs = []
    for i, true_c in enumerate(letter_classes):
        for j, pred_c in enumerate(letter_classes):
            if i != j and confusion[i][j] > 0.05:
                conf_pairs.append((confusion[i][j], true_c, pred_c))
    for prob, true_c, pred_c in sorted(conf_pairs, reverse=True)[:10]:
        print(f"    {true_c}→{pred_c}: {prob:.2f}")

    # --- Pass 2: word correction with confusion matrix ---
    print(f"\n  Pass 2: word correction...")
    decoder_name = f'CTC beam (width={beam_width})' if use_beam else 'CTC greedy'
    print(f"  Decoder: {decoder_name}")
    print(f"\n  {'Sample':>8}  {'True':>12}  {'Raw':>14}  "
          f"{'Corrected':>12}  {'OK':>4}  {'CER':>6}  "
          f"{'Word Acc':>10}  {'Elapsed':>8}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*14}  "
          f"{'-'*12}  {'-'*4}  {'-'*6}  "
          f"{'-'*10}  {'-'*8}")

    results   = []
    word_hits = 0
    cer_total = 0.0

    for idx, (s, decoded, est_len) in enumerate(
            zip(test_samples, raw_decodes, est_lens), 1):
        true_word = s['word']
        corrected = word_correct(decoded, word_list, length_tolerance,
                                 est_len=est_len,
                                 confusion=confusion,
                                 letter_classes=letter_classes)

        c_raw = cer(true_word, decoded)
        c_cor = cer(true_word, corrected)
        hit   = int(corrected == true_word)
        word_hits += hit
        cer_total += c_cor

        results.append({
            'true_word'        : true_word,
            'decoded_raw'      : decoded,
            'decoded_corrected': corrected,
            'correct_raw'      : int(decoded == true_word),
            'correct_corrected': hit,
            'cer_raw'          : round(c_raw, 4),
            'cer_corrected'    : round(c_cor, 4),
            'decoded_collapse' : decoded,
            'correct_collapse' : int(decoded == true_word),
            'cer_collapse'     : round(c_raw, 4),
        })

        elapsed  = time.time() - t_start
        curr_acc = word_hits / idx
        print(f"  {idx:>8}  {true_word:>12}  {decoded:>14}  "
              f"{corrected:>12}  "
              f"{'YES' if hit else 'no':>4}  {c_cor:>6.3f}  "
              f"{curr_acc:>10.4f}  {elapsed:>7.1f}s")

    n          = len(test_samples)
    word_acc   = word_hits / n
    mean_cer   = cer_total / n
    letter_acc = float('nan')
    return results, word_acc, mean_cer, letter_acc

def compute_loss(letter_logits, boundary_logits,
                 letters_list, offsets_list, b_pad,
                 lengths, letter_crit, boundary_crit, lam, device):
    """
    letter_logits   : (B, T, n_letters)
    boundary_logits : (B, T, 4)
    letter loss     : CrossEntropy at each true END (offset) position
    boundary loss   : CrossEntropy over all real timesteps
    """
    B, T, _ = letter_logits.shape

    # letter loss at END positions
    letter_preds_all = [];  letter_true_all = []
    for i in range(B):
        offsets = offsets_list[i].to(device).clamp(max=lengths[i]-1)
        letters = letters_list[i].to(device)
        idx     = offsets.unsqueeze(1).expand(-1, letter_logits.shape[2])
        preds   = letter_logits[i].gather(0, idx)
        letter_preds_all.append(preds)
        letter_true_all.append(letters)

    l_loss = letter_crit(
        torch.cat(letter_preds_all, dim=0),
        torch.cat(letter_true_all,  dim=0)
    )

    # boundary loss over all real timesteps (4-class CrossEntropy)
    t_idx     = torch.arange(T, device=device).unsqueeze(0)
    real_mask = t_idx < lengths.unsqueeze(1).to(device)   # (B, T)
    b_loss    = boundary_crit(
        boundary_logits[real_mask],      # (N_real, 4)
        b_pad.to(device)[real_mask]      # (N_real,)  int labels
    )

    return l_loss + lam * b_loss, l_loss.item(), b_loss.item()


# =============================================================================
# TRAINING
# =============================================================================

def make_loader(samples, batch_size=16, shuffle=True):
    class DS(torch.utils.data.Dataset):
        def __init__(self, s): self.s = s
        def __len__(self): return len(self.s)
        def __getitem__(self, i):
            s = self.s[i]
            return (torch.tensor(s['X'],        dtype=torch.float32),
                    torch.tensor(s['boundary'], dtype=torch.long),
                    torch.tensor(s['letters_enc'], dtype=torch.long),
                    torch.tensor(s['offsets'],  dtype=torch.long),
                    s['X'].shape[0])
    return DataLoader(DS(samples), batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate_fn)


def train_model(model, train_loader, val_loader,
                n_epochs, lr, lam, device):
    optimizer     = torch.optim.Adam(model.parameters(), lr=lr)
    letter_crit   = nn.CrossEntropyLoss()

    # class weights to counteract INSIDE domination
    # END gets highest weight — most important and most underrepresented per stroke
    # INSIDE gets lowest weight — dominates naturally and needs no encouragement
    boundary_weights = torch.tensor(
        [3.0,   # START
         0.5,   # INSIDE
         5.0,   # END
         2.0],  # TRANSITION
        dtype=torch.float32
    ).to(device)
    boundary_crit = nn.CrossEntropyLoss(weight=boundary_weights)
    best_val_loss = float('inf')
    best_weights  = copy.deepcopy(model.state_dict())
    patience_ctr  = 0

    model.to(device)

    print(f"\n  Training GRU | epochs={n_epochs} lr={lr} lambda={lam}")
    print(f"  {'Epoch':>6}  {'Train Loss':>12}  {'L Loss':>8}  "
          f"{'B Loss':>8}  {'Val Loss':>10}  {'Val Ltr Acc':>12}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*12}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        tr_loss = tr_l = tr_b = 0.0

        for X_pad, b_pad, letters_list, offsets_list, lengths in train_loader:
            X_pad   = X_pad.to(device);  lengths = lengths.to(device)
            letters_list = [l.to(device) for l in letters_list]
            offsets_list = [o.to(device) for o in offsets_list]

            optimizer.zero_grad()
            ll, bl = model(X_pad, lengths)
            loss, lv, bv = compute_loss(ll, bl, letters_list, offsets_list,
                                        b_pad, lengths, letter_crit,
                                        boundary_crit, lam, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            n = len(lengths)
            tr_loss += loss.item()*n;  tr_l += lv*n;  tr_b += bv*n

        n_tr    = len(train_loader.dataset)
        tr_loss /= n_tr;  tr_l /= n_tr;  tr_b /= n_tr

        model.eval()
        val_loss = val_c = val_t = 0

        with torch.no_grad():
            for X_pad, b_pad, letters_list, offsets_list, lengths in val_loader:
                X_pad   = X_pad.to(device);  lengths = lengths.to(device)
                ld = [l.to(device) for l in letters_list]
                od = [o.to(device) for o in offsets_list]
                ll, bl = model(X_pad, lengths)
                loss, _, _ = compute_loss(ll, bl, ld, od, b_pad, lengths,
                                          letter_crit, boundary_crit, lam, device)
                val_loss += loss.item() * len(lengths)

                for i in range(X_pad.shape[0]):
                    offsets = od[i].clamp(max=lengths[i]-1)
                    letters = ld[i]
                    idx     = offsets.unsqueeze(1).expand(-1, ll.shape[2])
                    preds   = ll[i].gather(0, idx).argmax(dim=1)
                    val_c  += (preds == letters).sum().item()
                    val_t  += len(letters)

        val_loss /= len(val_loader.dataset)
        val_acc   = val_c / val_t if val_t > 0 else 0.0

        print(f"  {epoch:>6}  {tr_loss:>12.4f}  {tr_l:>8.4f}  "
              f"{tr_b:>8.4f}  {val_loss:>10.4f}  {val_acc:>12.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = copy.deepcopy(model.state_dict())
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_weights)
    print(f"  Best val loss: {best_val_loss:.4f}")
    return model


# =============================================================================
# DECODE + EVALUATE
# =============================================================================

def greedy_decode(combined: np.ndarray, letter_classes: list,
                  end_threshold: float = 0.25) -> str:
    """
    Fast greedy decoder — no beam search.

    Logic:
    - Suppress emissions for first WARMUP timesteps
    - Track when P(END) rises above end_threshold for at least MIN_WINDOW_LEN
      consecutive timesteps — confirms a real END event (not noise)
    - Within the confirmed window find the timestep with highest letter
      confidence and emit that letter on the falling edge
    - Hard cooldown of MIN_GAP timesteps after each emission
    - End-of-sequence: if still in confirmed window, emit at best letter timestep
    """
    n_letters       = len(letter_classes)
    END_IDX         = 2
    MIN_GAP         = 8        # hard cooldown after emission (400ms)
    WARMUP          = 10       # suppress emissions for first N timesteps
    MIN_WINDOW_LEN  = 2        # P(END) must stay above threshold this many
                               # consecutive timesteps to confirm a real END event
    word            = []
    emit_times      = []   # timestep of each emission — used for collapse
    emit_probs      = []   # full letter prob vector at each emission — used for soft correction

    in_end_window   = False
    consecutive     = 0        # consecutive timesteps above threshold
    peak_p_end      = 0.0
    peak_end_t      = 0
    best_ltr_conf   = 0.0
    best_ltr_t      = 0
    last_emit_t     = -MIN_GAP

    for t in range(combined.shape[0]):
        lp    = combined[t, :n_letters]
        bp    = combined[t, n_letters:]
        p_end = float(bp[END_IDX])

        # warmup suppression
        if t < WARMUP:
            continue

        # cooldown — cannot open new window until MIN_GAP after last emit
        if (t - last_emit_t) < MIN_GAP:
            in_end_window = False
            consecutive   = 0
            peak_p_end    = 0.0
            best_ltr_conf = 0.0
            continue

        if p_end >= end_threshold:
            consecutive += 1

            # track best within the rising region regardless of confirmation
            if p_end > peak_p_end:
                peak_p_end = p_end
                peak_end_t = t
            ltr_conf = float(lp.max())
            if ltr_conf > best_ltr_conf:
                best_ltr_conf = ltr_conf
                best_ltr_t    = t

            # confirm window only after MIN_WINDOW_LEN consecutive timesteps
            if consecutive >= MIN_WINDOW_LEN:
                in_end_window = True

        else:
            # falling edge
            if in_end_window:
                # confirmed window just closed — emit
                letter = letter_classes[int(combined[best_ltr_t, :n_letters].argmax())]
                word.append(letter)
                emit_times.append(best_ltr_t)
                emit_probs.append(combined[best_ltr_t, :n_letters].copy())
                last_emit_t = best_ltr_t

            # reset window state
            in_end_window = False
            consecutive   = 0
            peak_p_end    = 0.0
            best_ltr_conf = 0.0

    # end of sequence — emit if confirmed window still open
    if in_end_window and (best_ltr_t - last_emit_t) >= MIN_GAP:
        letter = letter_classes[int(combined[best_ltr_t, :n_letters].argmax())]
        word.append(letter)
        emit_times.append(best_ltr_t)
        emit_probs.append(combined[best_ltr_t, :n_letters].copy())

    return ''.join(word), emit_times, emit_probs


def collapse_duplicates(decoded: str, emit_times: list,
                        emit_probs: list,
                        min_gap: int = 8):
    """
    Remove duplicate adjacent letters whose emission timesteps are too
    close together to represent a genuine repeated letter.
    Returns (collapsed_str, collapsed_emit_times, collapsed_emit_probs).
    Falls back to simple adjacent dedup if emit_times is mismatched.
    """
    if not decoded:
        return decoded, emit_times, emit_probs

    if len(emit_times) != len(decoded):
        # fallback — simple adjacent dedup, drop probs too
        result = [decoded[0]]
        for ch in decoded[1:]:
            if ch != result[-1]:
                result.append(ch)
        return ''.join(result), [], []

    result    = [decoded[0]]
    result_t  = [emit_times[0]]
    result_p  = [emit_probs[0]] if emit_probs else []

    for i, (ch, t) in enumerate(zip(decoded[1:], emit_times[1:]), start=1):
        if ch == result[-1] and (t - result_t[-1]) < min_gap:
            continue
        result.append(ch)
        result_t.append(t)
        if emit_probs:
            result_p.append(emit_probs[i])

    return ''.join(result), result_t, result_p


def build_confusion_matrix(results: list, letter_classes: list) -> np.ndarray:
    """
    Build a letter confusion matrix from decoded results.
    Aligns decoded_raw against true_word using edit distance traceback
    to identify which letters are commonly substituted for which.

    Returns confusion[i][j] = P(decoded as letter_classes[j] | true is letter_classes[i])
    Rows normalised to sum to 1.
    """
    n      = len(letter_classes)
    ltr_idx = {c: i for i, c in enumerate(letter_classes)}
    counts  = np.zeros((n, n), dtype=np.float32)

    for r in results:
        true = r['true_word']
        pred = r['decoded_raw']
        if not pred:
            continue
        m_len, n_len = len(true), len(pred)
        dp = [[0]*(n_len+1) for _ in range(m_len+1)]
        for i in range(m_len+1): dp[i][0] = i
        for j in range(n_len+1): dp[0][j] = j
        for i in range(1, m_len+1):
            for j in range(1, n_len+1):
                if true[i-1] == pred[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        # traceback
        i, j = m_len, n_len
        while i > 0 and j > 0:
            if true[i-1] == pred[j-1]:
                ti = ltr_idx.get(true[i-1])
                if ti is not None: counts[ti][ti] += 1
                i -= 1; j -= 1
            elif dp[i][j] == dp[i-1][j-1] + 1:
                ti = ltr_idx.get(true[i-1])
                pi = ltr_idx.get(pred[j-1])
                if ti is not None and pi is not None:
                    counts[ti][pi] += 1
                i -= 1; j -= 1
            elif dp[i][j] == dp[i-1][j] + 1:
                i -= 1
            else:
                j -= 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return counts / row_sums


def word_correct(decoded: str, word_list: list,
                 length_tolerance: int = 2,
                 est_len: int = None,
                 emit_probs: list = None,
                 confusion: np.ndarray = None,
                 letter_classes: list = None,
                 confusion_weight: float = 0.15) -> str:
    """
    Find the closest word in word_list using CER as the primary metric.

    CER = edit_distance(decoded, candidate) / len(candidate)

    Substitution cost is softened by two optional secondary signals,
    both with much smaller weight than the CER normalisation:

    1. confusion matrix — commonly confused letter pairs cost less
       e.g. if H is often decoded as N, H→N substitution costs 0.85 not 1.0
       weighted by confusion_weight (default 0.15)

    2. emit_probs — if provided, overrides confusion with direct model probs

    est_len adds a small soft length penalty if provided.
    """
    if not decoded:
        return decoded
    if not word_list:
        return decoded

    ltr_idx = {c: i for i, c in enumerate(letter_classes)}               if confusion is not None and letter_classes else {}

    def sub_cost(i: int, true_char: str, candidate_char: str) -> float:
        # emit_probs takes precedence if available
        if emit_probs and i < len(emit_probs):
            idx = ord(candidate_char) - ord('A')
            if 0 <= idx < len(emit_probs[i]):
                return max(0.0, 1.0 - float(emit_probs[i][idx]))
        # confusion matrix nudge — secondary signal only
        if confusion is not None and ltr_idx:
            ti = ltr_idx.get(true_char)
            ci = ltr_idx.get(candidate_char)
            if ti is not None and ci is not None:
                return max(0.0, 1.0 - confusion_weight * float(confusion[ti][ci]))
        return 1.0

    best_word  = decoded
    best_score = float('inf')

    for w in word_list:
        m, n = len(decoded), len(w)
        dp   = [[0.0]*(n+1) for _ in range(m+1)]
        for i in range(m+1): dp[i][0] = float(i)
        for j in range(n+1): dp[0][j] = float(j)
        for i in range(1, m+1):
            for j in range(1, n+1):
                if decoded[i-1] == w[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j-1] + sub_cost(i-1, decoded[i-1], w[j-1]),
                        dp[i-1][j]   + 1.0,
                        dp[i][j-1]   + 1.0,
                    )

        score = dp[m][n] / max(len(w), 1)
        if est_len is not None:
            score += 0.1 * abs(len(w) - est_len) / max(est_len, 1)

        if score < best_score or (score == best_score and len(w) < len(best_word)):
            best_score = score
            best_word  = w

    return best_word


def cer(true_word: str, pred_word: str) -> float:
    """Character Error Rate = edit distance / len(true_word)."""
    m, n = len(true_word), len(pred_word)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1] if true_word[i-1] == pred_word[j-1] \
                       else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n] / max(m, 1)


def decode_and_evaluate(model, test_samples, letter_classes,
                        lm, device, word_list,
                        avg_letter_intervals=20.0,
                        use_beam=False,
                        beam_width=2, lookahead=0,
                        alpha=1.0, beta=0.5, gamma=0.3,
                        end_threshold=0.25,
                        length_tolerance=2,
                        max_beam_samples=20):
    """
    For each test sample:
      1. Run model forward to get (K, 30) combined probs
      2. Feed into BeamDecoder timestep by timestep
      3. Compare decoded word to true word
    Returns (results_list, word_acc, mean_cer, letter_acc)
    """
    import time
    model.eval()
    results    = []
    word_hits  = 0
    cer_total  = 0.0
    ltr_hits   = ltr_total = 0
    n_total    = len(test_samples)
    t_start    = time.time()

    # limit beam search to avoid freezing — greedy runs on all samples
    if use_beam and max_beam_samples > 0:
        beam_samples   = test_samples[:max_beam_samples]
        greedy_samples = test_samples[max_beam_samples:]
        print(f"\n  Beam search on first {len(beam_samples)} samples, "
              f"greedy on remaining {len(greedy_samples)} samples.")
    else:
        beam_samples   = test_samples if use_beam else []
        greedy_samples = [] if use_beam else test_samples

    print(f"\n  Decoding {n_total} test samples...")

    # --- diagnostic on first sample ---
    s0     = test_samples[0]
    X0     = torch.tensor(s0['X'][np.newaxis], dtype=torch.float32).to(device)
    l0     = torch.tensor([s0['X'].shape[0]], dtype=torch.long).to(device)
    ll0, bl0 = model(X0, l0)
    lp0  = torch.softmax(ll0[0], dim=-1).detach().cpu().numpy()
    bp0  = torch.softmax(bl0[0], dim=-1).detach().cpu().numpy()
    p_end_vals = bp0[:, 2]
    print(f"\n  [Diagnostic — first test sample: '{s0['word']}']")
    print(f"  Sequence length  : {s0['X'].shape[0]} intervals")
    print(f"  P(END) min       : {p_end_vals.min():.4f}")
    print(f"  P(END) max       : {p_end_vals.max():.4f}")
    print(f"  P(END) mean      : {p_end_vals.mean():.4f}")
    print(f"  P(END) > 0.25    : {(p_end_vals > 0.25).sum()} timesteps")
    print(f"  P(END) > 0.15    : {(p_end_vals > 0.15).sum()} timesteps")
    print(f"  P(END) > 0.10    : {(p_end_vals > 0.10).sum()} timesteps")
    print(f"  Boundary argmax counts:")
    for i, name in enumerate(['START','INSIDE','END','TRANSITION']):
        count = int((bp0.argmax(axis=1) == i).sum())
        print(f"    {name:>12}: {count} timesteps")
    print(f"  True word offsets (END positions): {s0['offsets']}")
    print(f"  P(END) at true END positions:")
    for off in s0['offsets']:
        off = min(off, len(p_end_vals)-1)
        print(f"    t={off:>4}  P(END)={p_end_vals[off]:.4f}")
    print()
    print(f"\n  {'Sample':>8}  {'True':>10}  {'Raw':>10}  "
          f"{'Collapsed':>10}  {'Corrected':>10}  {'OK':>4}  "
          f"{'CER':>6}  {'Word Acc':>10}  {'Elapsed':>8}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  "
          f"{'-'*10}  {'-'*10}  {'-'*4}  "
          f"{'-'*6}  {'-'*10}  {'-'*8}")

    with torch.no_grad():
        # combine: beam samples first, then greedy samples
        all_decode_samples = [(s, True)  for s in beam_samples] + \
                             [(s, False) for s in greedy_samples]

        for idx, (s, run_beam) in enumerate(all_decode_samples, 1):
            true_word = s['word']
            X         = torch.tensor(s['X'][np.newaxis], dtype=torch.float32).to(device)
            lengths   = torch.tensor([s['X'].shape[0]], dtype=torch.long).to(device)

            ll, bl    = model(X, lengths)
            lp = torch.softmax(ll[0], dim=-1).cpu().numpy()
            bp = torch.softmax(bl[0], dim=-1).cpu().numpy()
            combined = np.concatenate([lp, bp], axis=-1)

            # letter accuracy at true END positions
            offsets = s['offsets']
            letters = s['letters_enc']
            for off, ltr in zip(offsets, letters):
                off = min(off, lp.shape[0]-1)
                pred_ltr = int(lp[off].argmax())
                if pred_ltr == ltr: ltr_hits += 1
                ltr_total += 1

            # decode
            if run_beam:
                is_first = (idx == 1)
                if is_first:
                    print(f"\n  [Beam debug for sample 1: '{true_word}']")
                decoder = BeamDecoder(
                    letter_classes = letter_classes,
                    lm             = lm,
                    beam_width     = beam_width,
                    lookahead      = lookahead,
                    alpha          = alpha,
                    beta           = beta,
                    gamma          = gamma,
                    end_threshold  = end_threshold,
                    debug          = is_first,
                )
                for t in range(combined.shape[0]):
                    decoder.step(combined[t])
                    if decoder.should_terminate:
                        break
                decoded   = decoder.finalise()
                collapsed = decoded
                est_len   = max(1, round(combined.shape[0] / avg_letter_intervals))
                corrected = word_correct(collapsed, word_list, length_tolerance,
                                         est_len=est_len)
            else:
                decoded, emit_times, emit_probs = greedy_decode(
                    combined, letter_classes, end_threshold)
                collapsed, _, col_probs = collapse_duplicates(
                    decoded, emit_times, emit_probs)
                est_len   = max(1, round(combined.shape[0] / avg_letter_intervals))
                corrected = word_correct(collapsed, word_list, length_tolerance,
                                         est_len=est_len, emit_probs=col_probs)

            # score all three versions
            c_raw  = cer(true_word, decoded)
            c_col  = cer(true_word, collapsed)
            c_cor  = cer(true_word, corrected)

            hit_raw = int(decoded   == true_word)
            hit_col = int(collapsed == true_word)
            hit_cor = int(corrected == true_word)

            word_hits += hit_cor   # primary metric uses corrected
            cer_total += c_cor

            results.append({
                'true_word'      : true_word,
                'decoded_raw'    : decoded,
                'decoded_collapse': collapsed,
                'decoded_corrected': corrected,
                'correct_raw'    : hit_raw,
                'correct_collapse': hit_col,
                'correct_corrected': hit_cor,
                'cer_raw'        : round(c_raw, 4),
                'cer_collapse'   : round(c_col, 4),
                'cer_corrected'  : round(c_cor, 4),
            })

            # print progress — show all three
            elapsed  = time.time() - t_start
            curr_acc = word_hits / idx
            print(f"  {idx:>8}  {true_word:>10}  {decoded:>10}  "
                  f"{collapsed:>10}  {corrected:>10}  "
                  f"{'YES' if hit_cor else 'no':>4}  {c_cor:>6.3f}  "
                  f"{curr_acc:>10.4f}  {elapsed:>7.1f}s")

    n          = len(test_samples)
    word_acc   = word_hits / n if n > 0 else 0.0
    mean_cer   = cer_total / n if n > 0 else 0.0
    letter_acc = ltr_hits / ltr_total if ltr_total > 0 else 0.0
    return results, word_acc, mean_cer, letter_acc


# =============================================================================
# SAVE
# =============================================================================

def save_results(results, word_acc, mean_cer, letter_acc,
                 letter_classes, out_dir, ts, run_config: dict):
    # csv — one row per sample
    df  = pd.DataFrame(results)
    csv_out = os.path.join(out_dir, f'm5_results_{ts}.csv')
    df.to_csv(csv_out, index=False)
    print(f"\nResults CSV  : {csv_out}")

    # txt — human readable summary
    txt_out = os.path.join(out_dir, f'm5_results_{ts}.txt')
    n       = len(results)
    with open(txt_out, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Air Writing — m5_train_eval Results\n")
        f.write(f"Timestamp    : {ts}\n")
        f.write(f"Test samples : {n}\n")
        f.write(f"Letters      : {letter_classes}\n")
        f.write("=" * 60 + "\n\n")

        f.write("RUN CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset directory  : {run_config.get('data_dir', 'unknown')}\n")
        f.write(f"Learning rate      : {run_config.get('lr', 'unknown')}\n")
        f.write(f"Epochs             : {run_config.get('n_epochs', 'unknown')}\n")
        f.write(f"Lambda (boundary)  : {run_config.get('lam', 'unknown')}\n")
        f.write(f"Boundary weights   : START={run_config.get('w_start', 'unknown')}  "
                f"INSIDE={run_config.get('w_inside', 'unknown')}  "
                f"END={run_config.get('w_end', 'unknown')}  "
                f"TRANSITION={run_config.get('w_trans', 'unknown')}\n")
        f.write(f"Decoder            : {run_config.get('decoder', 'unknown')}\n")
        if run_config.get('decoder') == 'greedy':
            f.write(f"END threshold      : {run_config.get('end_threshold', 'unknown')}\n")
            f.write(f"MIN_GAP            : 8 intervals (400ms)\n")
            f.write(f"Warmup             : 10 intervals (500ms)\n")
            f.write(f"Min window length  : 2 consecutive intervals\n")
            f.write(f"Length tolerance   : ±{run_config.get('length_tolerance', 2)} letters\n")
        else:
            f.write(f"Beam width         : {run_config.get('beam_width', 'unknown')}\n")
            f.write(f"Lookahead          : {run_config.get('lookahead', 'unknown')}\n")
            f.write(f"Alpha              : {run_config.get('alpha', 'unknown')}\n")
            f.write(f"Beta               : {run_config.get('beta', 'unknown')}\n")
            f.write(f"Gamma              : {run_config.get('gamma', 'unknown')}\n")
            f.write(f"Language model     : {run_config.get('lm_path', 'uniform')}\n")
        f.write(f"Avg letter duration: {run_config.get('avg_letter_intervals', 0):.1f} intervals "
                f"({run_config.get('avg_letter_intervals', 0) * 50:.0f}ms) "
                f"[median stroke duration excl. transitions]\n")
        f.write("\n")

        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Letter accuracy (at true END positions) : {letter_acc:.4f}\n\n")
        raw_acc = sum(r['correct_raw']        for r in results) / max(n, 1)
        col_acc = sum(r['correct_collapse']   for r in results) / max(n, 1)
        cor_acc = sum(r['correct_corrected']  for r in results) / max(n, 1)
        raw_cer = sum(r['cer_raw']            for r in results) / max(n, 1)
        col_cer = sum(r['cer_collapse']       for r in results) / max(n, 1)
        cor_cer = sum(r['cer_corrected']      for r in results) / max(n, 1)
        f.write(f"{'Stage':30}  {'Word Acc':>10}  {'Mean CER':>10}\n")
        f.write(f"{'-'*30}  {'-'*10}  {'-'*10}\n")
        f.write(f"{'Raw greedy':30}  {raw_acc:>10.4f}  {raw_cer:>10.4f}\n")
        f.write(f"{'After collapse':30}  {col_acc:>10.4f}  {col_cer:>10.4f}\n")
        f.write(f"{'After word correction':30}  {cor_acc:>10.4f}  {cor_cer:>10.4f}\n\n")

        f.write("PREDICTIONS\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'#':>5}  {'True':>12}  {'Raw':>12}  {'Collapsed':>12}  "
                f"{'Corrected':>12}  {'OK':>5}  {'CER(cor)':>8}\n")
        f.write(f"{'--':>5}  {'-'*12}  {'-'*12}  {'-'*12}  "
                f"{'-'*12}  {'-'*5}  {'-'*8}\n")
        for i, r in enumerate(results, 1):
            f.write(f"{i:>5}  {r['true_word']:>12}  {r['decoded_raw']:>12}  "
                    f"{r['decoded_collapse']:>12}  {r['decoded_corrected']:>12}  "
                    f"{'YES' if r['correct_corrected'] else 'no':>5}  "
                    f"{r['cer_corrected']:>8.3f}\n")

        f.write("\n" + "=" * 60 + "\n")
        correct = sum(r['correct_corrected'] for r in results)
        f.write(f"Correct (after word correction): {correct} / {n}\n")
        f.write("=" * 60 + "\n")

    print(f"Results TXT  : {txt_out}")


def save_checkpoint(model, letter_classes, out_dir, ts):
    pt_out  = os.path.join(out_dir, f'm5_gru_{ts}.pt')
    npy_out = os.path.join(out_dir, f'm5_gru_{ts}_classes.npy')
    torch.save(model.state_dict(), pt_out)
    np.save(npy_out, np.array(letter_classes))
    print(f"Model        : {pt_out}")
    print(f"Classes      : {npy_out}")

