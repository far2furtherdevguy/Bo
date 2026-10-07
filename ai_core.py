import base64
import json
import os
import random
import re
import sqlite3
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Result:
    text: str
    conf: float
    source: str
    meta: Optional[Dict] = None


def _is_word(token: str) -> bool:
    return bool(token) and (token[0].isalnum() or token[0] == "_")


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# ==================== HYPERDIMENSIONAL VECTORS ====================
class HDV:
    """Hyperdimensional computing for semantic memory"""

    SCALE = 10  # association strengths are stored as ints (strength * SCALE)

    def __init__(self, dim=2048):
        self.dim = dim
        self.symbols: Dict[str, np.ndarray] = {}
        self.assocs: Dict[str, np.ndarray] = {}
        self._matrix = None
        self._matrix_keys: List[str] = []

    def _vec(self, token: str) -> np.ndarray:
        v = self.symbols.get(token)
        if v is None:
            # FIX: stable seed (python's hash() is randomized per process) and a
            # private RNG (the old code reseeded numpy's GLOBAL RNG on every call)
            rng = np.random.RandomState(zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF)
            v = rng.choice(np.array([-1, 1], dtype=np.int8), self.dim)
            self.symbols[token] = v
            self._matrix = None
        return v

    def _symbol_matrix(self):
        if self._matrix is None or len(self._matrix_keys) != len(self.symbols):
            self._matrix_keys = list(self.symbols.keys())
            if self._matrix_keys:
                self._matrix = np.stack(
                    [self.symbols[k] for k in self._matrix_keys]
                ).astype(np.float32)
            else:
                self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        return self._matrix_keys, self._matrix

    def bind(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        return v1 * v2

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        return np.sum(vectors, axis=0, dtype=np.int32)

    def similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1.astype(np.int32), v2.astype(np.int32))) / self.dim

    def _ctx_vec(self, tokens: List[str]) -> np.ndarray:
        """Bipolar (+1/-1) vector for a context (majority of bundled tokens)."""
        s = self.bundle([self._vec(t) for t in tokens])
        return np.where(s >= 0, 1, -1).astype(np.int8)

    def learn(self, context: List[str], target: str, strength=1.0):
        if not context or not target:
            return
        ctx = context[-5:]
        key = " ".join(ctx)
        binding = self._ctx_vec(ctx).astype(np.int32) * self._vec(target).astype(np.int32)
        delta = binding * int(round(strength * self.SCALE))
        old = self.assocs.get(key)
        total = delta if old is None else old.astype(np.int32) + delta
        self.assocs[key] = np.clip(total, -120, 120).astype(np.int8)

    def predict(self, tokens: List[str], k=5) -> List[Tuple[str, float]]:
        key = None
        for n in range(min(5, len(tokens)), 0, -1):
            cand = " ".join(tokens[-n:])
            if cand in self.assocs:
                key = cand
                break
        if key is None:
            return []
        # FIX: the old code compared the stored binding directly with symbol
        # vectors, which is just noise. Unbind with the context first.
        probe = self.assocs[key].astype(np.float32) * self._ctx_vec(key.split()).astype(np.float32)
        keys, mat = self._symbol_matrix()
        if not keys:
            return []
        sims = (mat @ probe) / (self.dim * self.SCALE)
        order = np.argsort(-sims)[: k + 10]
        out = []
        for i in order:
            name = keys[i]
            if sims[i] > 0.1 and not name.startswith("__"):
                out.append((name, float(sims[i])))
            if len(out) >= k:
                break
        return out

    def get_related(self, token: str, k=10) -> List[Tuple[str, float]]:
        # FIX: random hypervectors are never similar to each other, so the old
        # version always returned []. Use learned associations instead.
        return self.predict([token], k=k)

    def compose(self, words: List[str], roles: List[str]) -> np.ndarray:
        if len(words) != len(roles):
            roles = ["content"] * len(words)
        vec = np.zeros(self.dim, dtype=np.int32)
        for word, role in zip(words, roles):
            vec += self._vec(f"__role_{role}").astype(np.int32) * self._vec(word).astype(np.int32)
        return vec

    # FIX: pickle.loads on a user-supplied string is remote code execution. Use JSON.
    def to_dict(self) -> Dict:
        return {
            "dim": self.dim,
            "symbols": list(self.symbols.keys()),
            "assocs": {
                k: base64.b64encode(v.tobytes()).decode() for k, v in self.assocs.items()
            },
        }

    def from_dict(self, data: Dict):
        if data.get("dim") != self.dim:
            raise ValueError("dimension mismatch")
        self.symbols = {}
        self._matrix = None
        for name in data.get("symbols", []):
            self._vec(name)
        self.assocs = {}
        for k, b in data.get("assocs", {}).items():
            arr = np.frombuffer(base64.b64decode(b), dtype=np.int8)
            if arr.shape[0] == self.dim:
                self.assocs[k] = arr.copy()

    def save(self) -> str:
        return base64.b64encode(json.dumps(self.to_dict()).encode()).decode()

    def load(self, encoded: str):
        self.from_dict(json.loads(base64.b64decode(encoded)))


# ==================== LEARNING CLASSIFIER SYSTEM ====================
class LCS:
    """Genetic algorithm for IF-THEN rules"""

    def __init__(self, max_rules=200):
        self.rules: List[Dict] = []
        self.max_rules = max_rules
        self.mutation_rate = 0.1

    def learn(self, context: List[str], action: str, reward=1.0):
        condition = " ".join(context[-4:])
        for rule in self.rules:
            if rule["if"] == condition:
                if rule["then"] == action:
                    rule["f"] = min(1.0, rule["f"] + 0.15 * reward)
                else:
                    rule["f"] = max(0.0, rule["f"] - 0.15 * reward)
                    if rule["f"] < 0.3:
                        rule["then"] = action
                        rule["f"] = 0.5
                rule["uses"] += 1
                return

        self.rules.append({"if": condition, "then": action, "f": 0.5,
                           "uses": 1, "created": len(self.rules)})

        if len(self.rules) > self.max_rules:
            self.rules.sort(key=lambda r: r["f"] * np.log(r["uses"] + 1), reverse=True)
            self.rules = self.rules[: self.max_rules]
            for rule in self.rules[-10:]:
                if random.random() < self.mutation_rate:
                    words = rule["if"].split()
                    if words:
                        idx = random.randint(0, len(words) - 1)
                        words[idx] = random.choice(words)
                        rule["if"] = " ".join(words)

    def predict(self, tokens: List[str]) -> List[Tuple[str, float]]:
        matches = []
        for n in range(min(4, len(tokens)), 0, -1):
            suffix = tokens[-n:]
            matches = [r for r in self.rules if r["if"].split()[-n:] == suffix]
            if matches:
                break
        matches.sort(key=lambda x: x["f"], reverse=True)
        return [(r["then"], r["f"]) for r in matches[:5]]

    def get_rules(self, min_fitness=0.3) -> List[Dict]:
        return [r for r in self.rules if r["f"] > min_fitness]


# ==================== CELLULAR AUTOMATA ====================
class CA:
    """1D cellular automaton for pattern generation"""

    def __init__(self, size=20):
        self.size = size
        self.state = np.random.randint(0, 2, size).astype(np.int8)
        self.rule = self._random_rule()
        self.generation = 0
        self.stability = 0

    def _random_rule(self) -> np.ndarray:
        return np.random.randint(0, 2, 8).astype(np.int8)

    def step(self, n=1):
        for _ in range(n):
            left = np.roll(self.state, 1)
            right = np.roll(self.state, -1)
            idx = left * 4 + self.state * 2 + right
            new_state = self.rule[idx]
            if np.array_equal(new_state, self.state):
                self.stability += 1
            else:
                self.stability = 0
            self.state = new_state
            self.generation += 1

    def features(self) -> np.ndarray:
        return self.state.astype(np.float32)

    def entropy(self) -> float:
        _, counts = np.unique(self.state, return_counts=True)
        probs = counts / len(self.state)
        return float(-np.sum(probs * np.log2(probs + 1e-10)))

    def evolve_if_stable(self, threshold=10):
        if self.stability > threshold:
            self.rule = self._random_rule()
            self.stability = 0
            return True
        return False


# ==================== PRODUCTION RULES (REASONING) ====================
class ProductionSystem:
    """Forward-chaining inference engine"""

    def __init__(self):
        self.rules: List[Dict] = []
        self.facts = set()
        self.inferred: List[Dict] = []

    def add_rule(self, conditions: List[str], conclusion: str, conf=0.9):
        self.rules.append({"if": {c.lower() for c in conditions},
                           "then": conclusion.lower(), "conf": conf})

    def add_fact(self, fact: str):
        self.facts.add(fact.lower())

    def infer(self, max_iterations=10) -> List[str]:
        new_facts = []
        for _ in range(max_iterations):
            changed = False
            for rule in self.rules:
                if rule["if"].issubset(self.facts) and rule["then"] not in self.facts:
                    self.facts.add(rule["then"])
                    new_facts.append(rule["then"])
                    self.inferred.append({
                        "rule": f"{' & '.join(sorted(rule['if']))} -> {rule['then']}",
                        "conf": rule["conf"],
                    })
                    changed = True
            if not changed:
                break
        return new_facts

    def infer_from(self, extra_facts: List[str], max_iterations=10) -> set:
        """Non-mutating inference: what follows from known facts + extra_facts."""
        start = set(self.facts) | {f.lower() for f in extra_facts}
        facts = set(start)
        for _ in range(max_iterations):
            changed = False
            for rule in self.rules:
                if rule["if"].issubset(facts) and rule["then"] not in facts:
                    facts.add(rule["then"])
                    changed = True
            if not changed:
                break
        return facts - start

    def query(self, fact: str) -> Tuple[bool, Optional[str]]:
        fact = fact.lower()
        if fact in self.facts:
            return True, "known"
        self.infer()
        if fact in self.facts:
            return True, "inferred"
        return False, None

    def explain(self, fact: str) -> List[str]:
        return [i["rule"] for i in self.inferred if fact.lower() in i["rule"]]


# ==================== ADVANCED NLP ====================
class NLPProcessor:
    """Rule-based NLP with statistical components"""

    def __init__(self):
        self.word_freq = Counter()
        self.bigrams = defaultdict(Counter)
        self.trigrams = defaultdict(Counter)
        self.stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'shall', 'can', 'to', 'of',
            'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'up',
            'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then'}

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower())

    QUESTION_FILLER = {'what', 'whats', 's', 't', 'm', 'your', 'you', 'me', 'my', 'i', 'it',
                       'how', 'tell', 'about', 'please', 'who', 'where', 'when', 'why'}

    def content_words(self, text: str) -> set:
        return {t for t in self.tokenize(text)
                if _is_word(t) and len(t) > 1 and t not in self.stop_words
                and t not in self.QUESTION_FILLER}

    def pos_tag(self, tokens: List[str]) -> List[Tuple[str, str]]:
        tags = []
        for token in tokens:
            if not _is_word(token):
                tags.append((token, 'PUNCT'))
            elif token in self.stop_words:
                tags.append((token, 'STOP'))
            elif token.isdigit():
                tags.append((token, 'NUM'))
            elif token.endswith(('ing', 'ed')):
                tags.append((token, 'VERB'))
            elif token.endswith('ly'):
                tags.append((token, 'ADV'))
            elif token.endswith(('ous', 'ive', 'al', 'able', 'ible', 'ful', 'less')):
                tags.append((token, 'ADJ'))
            else:
                tags.append((token, 'NOUN'))
        return tags

    def extract_entities(self, tokens: List[str]) -> Dict[str, List[str]]:
        """Extract simple named entities (expects ORIGINAL-case tokens)."""
        entities = {'PERSON': [], 'LOCATION': [], 'ORG': [], 'MISC': []}
        current = []
        for token in tokens:
            if token and token[0].isupper() and token.lower() not in self.stop_words:
                current.append(token)
            else:
                if current:
                    entities['MISC'].append(' '.join(current))
                current = []
        if current:
            entities['MISC'].append(' '.join(current))
        match = re.search(r'\b(?:mr|mrs|dr|prof)\.?\s+(\w+)', ' '.join(tokens), re.I)
        if match:
            entities['PERSON'].append(match.group(1))
        return entities

    def update_stats(self, tokens: List[str]):
        for token in tokens:
            self.word_freq[token] += 1
        for i in range(len(tokens) - 1):
            self.bigrams[tokens[i]][tokens[i + 1]] += 1
        for i in range(len(tokens) - 2):
            self.trigrams[(tokens[i], tokens[i + 1])][tokens[i + 2]] += 1

    def predict_next(self, tokens: List[str], k=3) -> List[Tuple[str, float]]:
        if len(tokens) >= 2:
            key = (tokens[-2], tokens[-1])
            if key in self.trigrams:
                total = sum(self.trigrams[key].values())
                return [(w, c / total) for w, c in self.trigrams[key].most_common(k)]
        if tokens:
            last = tokens[-1]
            if last in self.bigrams:
                total = sum(self.bigrams[last].values())
                return [(w, c / total) for w, c in self.bigrams[last].most_common(k)]
        total = sum(self.word_freq.values())
        if not total:
            return []
        return [(w, c / total) for w, c in self.word_freq.most_common(k)]

    def similarity(self, text1: str, text2: str) -> float:
        """Jaccard similarity over content words"""
        t1, t2 = self.content_words(text1), self.content_words(text2)
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)


# ==================== CREATIVE GENERATOR ====================
class CreativeGenerator:
    """Template + HDV + CA based text generation"""

    DEFAULT_WORDS = ['mystery', 'shadow', 'dream', 'echo', 'void']

    def __init__(self, hdv: HDV, ca: CA, nlp: NLPProcessor):
        self.hdv = hdv
        self.ca = ca
        self.nlp = nlp
        self.templates = [
            "The {adj} {noun} {verb} through the {noun2}.",
            "In a world of {noun}, {noun2} becomes {adj}.",
            "{Noun} whispers secrets to {noun2}.",
            "Every {noun} carries a {noun2} within.",
            "When {noun} meets {noun2}, {verb} begins.",
            "The {noun} of {noun2} is {adj} and {adj2}.",
            "Beneath the {noun}, {noun2} dreams of {noun3}.",
        ]

    def generate(self, seed_words: List[str], creativity=0.7) -> str:
        seeds = [w for w in seed_words if w.isalpha()]
        related = []
        for word in seeds:
            related.extend(w for w, _ in self.hdv.get_related(word, k=5) if w.isalpha())
        all_words = list(dict.fromkeys(seeds + related))
        for w in self.DEFAULT_WORDS:
            if len(all_words) >= 5:
                break
            if w not in all_words:
                all_words.append(w)

        self.ca.step(15)
        template = self.templates[int(np.sum(self.ca.features())) % len(self.templates)]

        words = all_words.copy()
        random.shuffle(words)
        result = template
        slots = ['adj', 'noun', 'verb', 'noun2', 'adj2', 'noun3', 'Noun']
        for i, slot in enumerate(slots):
            tag = '{' + slot + '}'
            if tag in result:
                word = words[i] if i < len(words) else random.choice(all_words)
                if slot == 'Noun':
                    word = word.capitalize()
                result = result.replace(tag, word, 1)
        return result

    def generate_poem(self, seed_words: List[str], lines=4) -> str:
        return '\n'.join(self.generate(seed_words, creativity=0.9) for _ in range(lines))


# ==================== ANALOGICAL REASONER ====================
class AnalogicalReasoner:
    """Solve A:B :: C:? problems"""

    def __init__(self, hdv: HDV):
        self.hdv = hdv
        self.analogies: List[Dict] = []

    def add_analogy(self, a: str, b: str, relation: str):
        va = self.hdv._vec(a).astype(np.int16)
        vb = self.hdv._vec(b).astype(np.int16)
        self.analogies.append({"pair": (a, b), "relation": relation, "transform": vb - va})

    def solve(self, a: str, b: str, c: str) -> Optional[str]:
        # FIX: random vectors have no semantics, so the pure vector method only
        # works if the answer pair was taught. Use the relation label first.
        relation = None
        transform = None
        for an in self.analogies:
            if an["pair"] == (a, b):
                relation, transform = an["relation"], an["transform"]
                break
            if an["pair"] == (b, a):
                relation, transform = an["relation"], -an["transform"]
                break
        if relation is None:
            return None

        for an in self.analogies:
            if an["relation"] == relation and an["pair"][0] == c and an["pair"] != (a, b):
                return an["pair"][1]

        target = self.hdv._vec(c).astype(np.int16) + transform
        best_word, best_sim = None, -1.0
        for word, vec in self.hdv.symbols.items():
            if word in (a, b, c) or word.startswith("__"):
                continue
            sim = float(np.dot(target.astype(np.int32), vec.astype(np.int32))) / self.hdv.dim
            if sim > best_sim:
                best_sim, best_word = sim, word
        return best_word if best_sim > 0.5 else None


# ==================== MAIN AI CLASS ====================
class SymbolicAI:
    def __init__(self, db_path: Optional[str] = None):
        self.hdv = HDV()
        self.lcs = LCS()
        self.ca = CA()
        self.ps = ProductionSystem()
        self.nlp = NLPProcessor()
        self.creative = CreativeGenerator(self.hdv, self.ca, self.nlp)
        self.analogies = AnalogicalReasoner(self.hdv)
        self.meta_w = {"hdv": 0.35, "lcs": 0.25, "ca": 0.1, "nlp": 0.2, "ps": 0.1}
        self.interactions = 0
        self.conversation_history: List[Dict] = []

        self.db = sqlite3.connect(db_path or os.environ.get("BO_DB_PATH", ":memory:"),
                                  check_same_thread=False)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY,
                topic TEXT,
                question TEXT,
                content TEXT,
                source TEXT,
                priority INTEGER DEFAULT 5,
                created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS training_log (
                id INTEGER PRIMARY KEY,
                input TEXT,
                output TEXT,
                source TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
        self.db.commit()
        self._init_defaults()

    # ---------- knowledge base ----------
    def add_knowledge(self, topic: str, content: str, priority: int = 5,
                      source: str = "default", question: Optional[str] = None) -> bool:
        """FIX: this method was missing -> AttributeError on startup."""
        topic = (topic or "general").lower()
        exists = self.db.execute(
            "SELECT 1 FROM knowledge WHERE topic=? AND content=? LIMIT 1",
            (topic, content)).fetchone()
        if exists:
            return False
        self.db.execute(
            "INSERT INTO knowledge (topic, question, content, source, priority) VALUES (?,?,?,?,?)",
            (topic, question, content, source, priority))
        self.db.commit()
        return True

    def _normalize(self, text: str) -> str:
        return " ".join(t for t in self.nlp.tokenize(text) if _is_word(t))

    def _init_defaults(self):
        greetings = [
            ("hello", "Hello! How can I help you today?"),
            ("hi", "Hi there! Nice to meet you."),
            ("hey", "Hey! What's on your mind?"),
            ("good morning", "Good morning! Hope you have a great day."),
            ("good afternoon", "Good afternoon! How are you doing?"),
            ("good evening", "Good evening! How can I assist you?"),
            ("how are you", "I'm doing well, thank you for asking! How about you?"),
            ("what's up", "Not much, just processing some thoughts. What's up with you?"),
            ("how's it going", "It's going great! Thanks for asking."),
            ("nice to meet you", "Nice to meet you too! I'm excited to learn from you."),
        ]
        for q, a in greetings:
            self.teach(q, a, source="default")

        facts = [
            ("name", "I am Symbolic AI, a learning system."),
            ("creator", "I was created by a curious developer."),
            ("purpose", "I learn from conversations and try to be helpful."),
        ]
        for topic, content in facts:
            self.add_knowledge(topic, content, priority=8)

        self.ps.add_rule(["human"], "mortal", 0.95)
        self.ps.add_rule(["alive"], "human", 0.7)
        self.ps.add_rule(["mortal"], "will_die", 0.9)

    def _update_meta_weights(self, source: str, success: bool):
        lr = 0.05
        if success:
            self.meta_w[source] = min(0.6, self.meta_w.get(source, 0.2) + lr)
        else:
            self.meta_w[source] = max(0.05, self.meta_w.get(source, 0.2) - lr)
        total = sum(self.meta_w.values())
        self.meta_w = {k: v / total for k, v in self.meta_w.items()}

    def _exact_answer(self, norm_question: str) -> Optional[str]:
        if not norm_question:
            return None
        row = self.db.execute(
            "SELECT content FROM knowledge WHERE question=? ORDER BY priority DESC, id DESC LIMIT 1",
            (norm_question,)).fetchone()
        return row[0] if row else None

    def _search_knowledge(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Token-overlap search (the old version LIKE-matched the whole sentence,
        so almost nothing ever matched)."""
        words = self.nlp.content_words(query)
        if not words:
            return []
        clauses, params = [], []
        for w in list(words)[:10]:
            pat = f"%{_like_escape(w)}%"
            clauses.append("topic LIKE ? ESCAPE '\\' OR question LIKE ? ESCAPE '\\' OR content LIKE ? ESCAPE '\\'")
            params += [pat, pat, pat]
        rows = self.db.execute(
            f"SELECT topic, question, content, priority FROM knowledge WHERE {' OR '.join(clauses)} LIMIT 100",
            params).fetchall()
        scored = []
        for topic, question, content, priority in rows:
            basis = question or f"{topic} {content}"
            cw = self.nlp.content_words(basis)
            if not cw:
                continue
            overlap = len(words & cw) / len(words | cw)
            scored.append((content, overlap + priority * 0.01))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ---------- inference ----------
    def think(self, query: str) -> Result:
        self.interactions += 1
        tokens = self.nlp.tokenize(query or "")
        words = [t for t in tokens if _is_word(t)]
        if not words:
            return Result("I didn't understand that.", 0.1, "error")

        self.nlp.update_stats(tokens)

        norm = " ".join(words)
        exact = self._exact_answer(norm)
        if exact:
            self._learn(query, exact, words)
            return Result(exact, 0.95, "memory", {"match": "exact"})

        candidates: Dict[str, float] = {}

        for text, score in self._search_knowledge(query):
            candidates[text] = candidates.get(text, 0) + score

        ps_result = self._try_production_system(words)
        if ps_result:
            candidates[ps_result] = candidates.get(ps_result, 0) + 0.8

        # Subsystems predict the FIRST WORD of a good reply (that's what _learn
        # trains them on), so use them to vote between full-sentence candidates.
        votes: Dict[str, float] = defaultdict(float)
        for tok, conf in self.hdv.predict(words, k=3):
            votes[tok] += conf * self.meta_w["hdv"]
        for tok, conf in self.lcs.predict(words):
            votes[tok] += conf * self.meta_w["lcs"]
        for tok, conf in self.nlp.predict_next(words, k=3):
            votes[tok] += conf * self.meta_w["nlp"]
        for text in list(candidates):
            first = next((t for t in self.nlp.tokenize(text) if _is_word(t)), None)
            if first:
                candidates[text] += votes.get(first, 0)

        best = max(candidates.items(), key=lambda x: x[1]) if candidates else None
        if best is None or best[1] < 0.15:
            creative = self.creative.generate(words[:3])
            return Result(creative, 0.3, "creative", {"fallback": True})

        self._learn(query, best[0], words)
        return Result(best[0], min(best[1], 1.0), "ensemble",
                      {"candidates": len(candidates), "tokens": len(tokens)})

    def _try_production_system(self, words: List[str]) -> Optional[str]:
        """Answer "what is X" via forward chaining (no longer pollutes the fact base)."""
        if "what" in words and "is" in words:
            idx = words.index("is")
            subject = next((w for w in words[idx + 1:] if w not in self.nlp.stop_words), None)
            if subject:
                derived = self.ps.infer_from([subject])
                if derived:
                    return f"{subject} implies: {', '.join(sorted(derived))}."
        return None

    def _learn(self, query: str, response: str, words: List[str]):
        response_tokens = [t for t in self.nlp.tokenize(response) if _is_word(t)]
        if response_tokens:
            for i in range(len(words)):
                self.hdv.learn(words[: i + 1], response_tokens[0], strength=0.5)
            self.lcs.learn(words, response_tokens[0], reward=0.5)

        if self.interactions % 20 == 0:
            self.ca.step(10)
            self.ca.evolve_if_stable()

        self.db.execute("INSERT INTO training_log (input, output, source) VALUES (?,?,?)",
                        (query, response, "conversation"))
        self.db.commit()

        self.conversation_history.append(
            {"user": query, "ai": response, "timestamp": self.interactions})
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-25:]

    def teach(self, question: str, answer: str, source="user") -> Dict:
        norm_q = self._normalize(question or "")
        a_tokens = [t for t in self.nlp.tokenize(answer or "") if _is_word(t)]
        q_tokens = norm_q.split()
        if not q_tokens or not (answer or "").strip():
            return {"status": "Error: need a question and an answer"}

        # latest answer for the same question wins
        self.db.execute("DELETE FROM knowledge WHERE question=?", (norm_q,))
        self.db.execute(
            "INSERT INTO knowledge (topic, question, content, source, priority) VALUES (?,?,?,?,?)",
            (q_tokens[0], norm_q, answer, source, 7 if source == "user" else 5))
        self.db.commit()

        if a_tokens:
            for i in range(len(q_tokens)):
                self.hdv.learn(q_tokens[: i + 1], a_tokens[0], strength=1.0)
            for i in range(len(a_tokens) - 1):
                self.hdv.learn([a_tokens[i]], a_tokens[i + 1], strength=0.8)
            self.lcs.learn(q_tokens, a_tokens[0], reward=1.0)

        self.nlp.update_stats(q_tokens)
        self.nlp.update_stats(a_tokens)

        return {
            "status": "Learned!",
            "hdv_symbols": len(self.hdv.symbols),
            "lcs_rules": len(self.lcs.rules),
            "db_entries": self.db.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0],
        }

    def train_from_text(self, text: str, source="file") -> Dict:
        lines = (text or "").split('\n')
        trained = 0
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tokens = [t for t in self.nlp.tokenize(line) if _is_word(t)]
            if len(tokens) >= 3:
                self.nlp.update_stats(tokens)
                for i in range(len(tokens) - 1):
                    self.hdv.learn([tokens[i]], tokens[i + 1], strength=0.3)
                trained += 1
        if source.lower().endswith('.rb') or 'ruby' in source.lower():
            self._learn_ruby_patterns(text)
        return {"lines_processed": len(lines), "patterns_learned": trained, "source": source}

    def _learn_ruby_patterns(self, code: str):
        ruby_patterns = [
            (["def"], "method"), (["class"], "Class"), (["end"], "block_end"),
            (["if"], "condition"), (["else"], "alternative"),
            (["puts"], "output"), (["require"], "import"),
        ]
        for context, target in ruby_patterns:
            self.hdv.learn(context, target, strength=0.5)
            self.lcs.learn(context, target, reward=0.5)

    # ---------- misc API ----------
    def generate_creative(self, seed: str, poem=False) -> str:
        tokens = self.nlp.tokenize(seed or "")
        return self.creative.generate_poem(tokens) if poem else self.creative.generate(tokens)

    def solve_analogy(self, a: str, b: str, c: str) -> Optional[str]:
        return self.analogies.solve(a, b, c)

    def add_analogy(self, a: str, b: str, relation: str):
        self.analogies.add_analogy(a, b, relation)

    def reason(self, facts: List[str]) -> List[str]:
        for fact in facts:
            self.ps.add_fact(fact)
        return self.ps.infer()

    def get_stats(self) -> Dict:
        return {
            "interactions": self.interactions,
            "hdv_symbols": len(self.hdv.symbols),
            "hdv_assocs": len(self.hdv.assocs),
            "lcs_rules": len(self.lcs.rules),
            "ca_generation": self.ca.generation,
            "ca_entropy": self.ca.entropy(),
            "ps_facts": len(self.ps.facts),
            "ps_rules": len(self.ps.rules),
            "nlp_vocab": len(self.nlp.word_freq),
            "db_entries": self.db.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0],
            "meta_weights": self.meta_w,
        }

    def get_rules(self) -> List[Dict]:
        return self.lcs.get_rules()

    def get_facts(self) -> List[str]:
        return sorted(self.ps.facts)

    def save_state(self) -> str:
        return self.hdv.save()

    def load_state(self, state: str):
        self.hdv.load(state)
