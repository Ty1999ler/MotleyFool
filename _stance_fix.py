p = "foolwatch/stance.py"
s = open(p, encoding="utf-8").read()

old = '''BUY_QUESTION = (
    r"\bis\s+.{0,60}?\ba\s+buy\b",'''
new = '''BUY_QUESTION = (
    r"\bis\s+.{0,60}?\ba\s+buy\b",
    # "Why Is CAVA Stock Crashing, and Is It a Buying Opportunity?" leans buy,
    # but without this it fell through to the crash wording and scored bearish.
    r"\bbuying\s+opportunit(?:y|ies)\b",'''
assert old in s, "BUY_QUESTION anchor"; s = s.replace(old, new, 1)

old = '''COMPARISON_RE = re.compile('''
new = '''# "Why Corning Plunged Today" explains a move that already happened; it is news,
# not a bearish call. Without this, a downward move hit the warning vocabulary
# first while an upward one ("Why X Soared Today") correctly fell to news — an
# asymmetry that filed about a quarter of all warnings under the wrong label,
# and stocks that have just plunged often rebound, so it also skewed the track
# record against the Fool's cautious calls.
MOVE_EXPLAINER_RE = re.compile(
    r"^why\s+.{0,90}?\b(?:plunge[sd]?|plunging|sink(?:s|ing)?|sank|sunk|"
    r"tumble[sd]?|tumbling|crash(?:es|ed|ing)?|fell|falls|falling|"
    r"drop(?:s|ped|ping)?|slid(?:e|es|ing)?|slump(?:s|ed|ing)?|"
    r"skyrocket(?:s|ed|ing)?|soar(?:s|ed|ing)?|jump(?:s|ed|ing)?|"
    r"surge[sd]?|surging|rall(?:y|ies|ied|ying)|pop(?:s|ped|ping)?|"
    r"climb(?:s|ed|ing)?|spike[sd]?|spiking|rose|rising|gain(?:s|ed|ing)?)\b",
    re.I,
)
# ...unless it looks forward: "Why a Stock Market Crash Could Be Coming" is a
# genuine warning, not an explainer.
FORWARD_RE = re.compile(r"\b(?:could|may|might|will|would|about to|coming|set to)\b",
                        re.I)

COMPARISON_RE = re.compile('''
assert old in s, "COMPARISON anchor"; s = s.replace(old, new, 1)

old = '''    if sell_q:
        return ("sell_lean", -1)'''
new = '''    if sell_q:
        return ("sell_lean", -1)
    # Explicit buy/sell wording above still wins; only a bare move explainer
    # is demoted to news.
    if MOVE_EXPLAINER_RE.search(t) and not FORWARD_RE.search(t):
        return ("news", 0)'''
assert old in s, "sell_q anchor"; s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("stance.py updated")
