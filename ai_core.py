import numpy as np
import random
import sqlite3
import json
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
import pickle
import base64

@dataclass
class Result:
    text: str
    conf: float
    source: str
    meta: Dict = None

# ==================== HYPERDIMENSIONAL VECTORS ====================

class HDV:
    """Hyperdimensional computing for semantic memory"""
    def __init__(self, dim=10000):
        self.dim = dim
        self.symbols = {}
        self.assocs = {}
        self.phrase_cache = {}
    
    def _vec(self, token: str) -> np.ndarray:
        if token not in self.symbols:
            np.random.seed(hash(token) % 2**32)
            self.symbols[token] = np.random.choice([-1, 1], self.dim).astype(np.int8)
        return self.symbols[token]
    
    def bind(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        return v1 * v2
    
    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        return np.sum(vectors, axis=0)
    
    def similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.sum(v1 * v2)) / self.dim
    
    def learn(self, context: List[str], target: str, strength=1.0):
        if not context or not target:
            return
        
        ctx_key = " ".join(context[-5:])
        ctx_vec = self.bundle([self._vec(t) for t in context])
        target_vec = self._vec(target)
        
        # Role binding: context ⊗ target
        binding = ctx_vec * target_vec
        
        if ctx_key in self.assocs:
            self.assocs[ctx_key] = np.clip(
                self.assocs[ctx_key] + binding * strength, -10, 10
            ).astype(np.int8)
        else:
            self.assocs[ctx_key] = binding.astype(np.int8)
    
    def predict(self, tokens: List[str], k=5) -> List[Tuple[str, float]]:
        key = " ".join(tokens[-5:])
        if key not in self.assocs:
            # Try shorter contexts
            for i in range(len(tokens)-1, 0, -1):
                short_key = " ".join(tokens[-i:])
                if short_key in self.assocs:
                    key = short_key
                    break
            else:
                return []
        
        query_vec = self.assocs[key]
        scores = []
        
        for token, sym_vec in self.symbols.items():
            sim = self.similarity(query_vec, sym_vec)
            if sim > 0.05:
                scores.append((token, sim))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    
    def get_related(self, token: str, k=10) -> List[Tuple[str, float]]:
        if token not in self.symbols:
            return []
        
        base = self.symbols[token]
        scores = []
        for other, vec in self.symbols.items():
            if other != token:
                sim = self.similarity(base, vec)
                if sim > 0.1:
                    scores.append((other, sim))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    
    def compose(self, words: List[str], roles: List[str]) -> np.ndarray:
        """Compose sentence meaning with grammatical roles"""
        if len(words) != len(roles):
            roles = ["content"] * len(words)
        
        vec = np.zeros(self.dim, dtype=np.int8)
        for word, role in zip(words, roles):
            role_vec = self._vec(f"__role_{role}")
            word_vec = self._vec(word)
            vec += role_vec * word_vec
        return vec
    
    def save(self) -> str:
        """Serialize to base64 string"""
        data = {
            'symbols': {k: v.tolist() for k, v in self.symbols.items()},
            'assocs': {k: v.tolist() for k, v in self.assocs.items()}
        }
        return base64.b64encode(pickle.dumps(data)).decode()
    
    def load(self, encoded: str):
        """Deserialize from base64 string"""
        data = pickle.loads(base64.b64decode(encoded))
        self.symbols = {k: np.array(v, dtype=np.int8) for k, v in data['symbols'].items()}
        self.assocs = {k: np.array(v, dtype=np.int8) for k, v in data['assocs'].items()}

# ==================== LEARNING CLASSIFIER SYSTEM ====================

class LCS:
    """Genetic algorithm for IF-THEN rules"""
    def __init__(self, max_rules=200):
        self.rules = []
        self.max_rules = max_rules
        self.mutation_rate = 0.1
    
    def learn(self, context: List[str], action: str, reward=1.0):
        condition = " ".join(context[-4:])
        
        # Find matching rule
        for rule in self.rules:
            if rule["if"] == condition:
                rule["f"] = min(1.0, rule["f"] + 0.15 * reward)
                rule["then"] = action
                rule["uses"] += 1
                return
        
        # Create new rule
        self.rules.append({
            "if": condition,
            "then": action,
            "f": 0.5,
            "uses": 1,
            "created": len(self.rules)
        })
        
        # Evolution: keep best, mutate occasionally
        if len(self.rules) > self.max_rules:
            self.rules.sort(key=lambda r: r["f"] * np.log(r["uses"] + 1), reverse=True)
            self.rules = self.rules[:self.max_rules]
            
            # Mutate worst survivors
            for rule in self.rules[-10:]:
                if random.random() < self.mutation_rate:
                    words = rule["if"].split()
                    if words:
                        idx = random.randint(0, len(words)-1)
                        words[idx] = random.choice(words)  # Duplicate random word
                        rule["if"] = " ".join(words)
    
    def predict(self, tokens: List[str]) -> List[Tuple[str, float]]:
        condition = " ".join(tokens[-4:])
        matches = [r for r in self.rules if r["if"] == condition]
        
        if not matches:
            # Partial match
            for i in range(len(tokens)-1, 0, -1):
                partial = " ".join(tokens[-i:])
                matches = [r for r in self.rules if partial in r["if"]]
                if matches:
                    break
        
        return [(r["then"], r["f"]) for r in sorted(matches, key=lambda x: x["f"], reverse=True)[:5]]
    
    def get_rules(self, min_fitness=0.3) -> List[Dict]:
        return [r for r in self.rules if r["f"] > min_fitness]

# ==================== CELLULAR AUTOMATA ====================

class CA:
    """1D cellular automaton for pattern generation"""
    def __init__(self, size=20):
        self.size = size
        self.state = np.random.randint(0, 2, size, dtype=np.int8)
        self.rule = self._random_rule()
        self.generation = 0
        self.stability = 0
    
    def _random_rule(self) -> Dict[int, int]:
        return {i: random.randint(0, 1) for i in range(8)}
    
    def step(self, n=1):
        for _ in range(n):
            new_state = np.zeros(self.size, dtype=np.int8)
            for i in range(self.size):
                left = self.state[(i-1) % self.size]
                center = self.state[i]
                right = self.state[(i+1) % self.size]
                idx = (left << 2) | (center << 1) | right
                new_state[i] = self.rule.get(idx, 0)
            
            if np.array_equal(new_state, self.state):
                self.stability += 1
            else:
                self.stability = 0
            
            self.state = new_state
            self.generation += 1
    
    def features(self) -> np.ndarray:
        return self.state.astype(np.float32)
    
    def entropy(self) -> float:
        """Measure of pattern complexity"""
        unique, counts = np.unique(self.state, return_counts=True)
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
        self.rules = []  # List of {"if": [facts], "then": fact, "conf": float}
        self.facts = set()
        self.inferred = []
    
    def add_rule(self, conditions: List[str], conclusion: str, conf=0.9):
        self.rules.append({
            "if": set(conditions),
            "then": conclusion,
            "conf": conf
        })
    
    def add_fact(self, fact: str):
        self.facts.add(fact.lower())
    
    def infer(self, max_iterations=10) -> List[str]:
        """Forward chaining"""
        new_facts = []
        
        for _ in range(max_iterations):
            changed = False
            for rule in self.rules:
                if rule["if"].issubset(self.facts) and rule["then"] not in self.facts:
                    self.facts.add(rule["then"])
                    new_facts.append(rule["then"])
                    self.inferred.append({
                        "rule": f"{' & '.join(rule['if'])} → {rule['then']}",
                        "conf": rule["conf"]
                    })
                    changed = True
            if not changed:
                break
        
        return new_facts
    
    def query(self, fact: str) -> Tuple[bool, Optional[str]]:
        """Check if fact is known or can be inferred"""
        fact = fact.lower()
        if fact in self.facts:
            return True, "known"
        
        # Try to infer
        self.infer()
        if fact in self.facts:
            return True, "inferred"
        
        return False, None
    
    def explain(self, fact: str) -> List[str]:
        """Explain how a fact was derived"""
        explanations = []
        for inference in self.inferred:
            if fact in inference["rule"]:
                explanations.append(inference["rule"])
        return explanations

# ==================== ADVANCED NLP ====================

class NLPProcessor:
    """Rule-based NLP with statistical components"""
    def __init__(self):
        self.word_freq = Counter()
        self.bigrams = defaultdict(Counter)
        self.trigrams = defaultdict(Counter)
        self.pos_patterns = {}
        self.stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 
                          'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                          'could', 'should', 'may', 'might', 'shall', 'can', 'to', 'of',
                          'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                          'through', 'during', 'before', 'after', 'above', 'below', 'up',
                          'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then'}
    
    def tokenize(self, text: str) -> List[str]:
        """Simple but effective tokenization"""
        text = text.lower()
        text = re.sub(r'([.!?,:;])', r' \1 ', text)
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        return [t for t in tokens if t.strip()]
    
    def pos_tag(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Simple rule-based POS tagging"""
        tags = []
        for i, token in enumerate(tokens):
            if token in self.stop_words:
                tags.append((token, 'STOP'))
            elif token.endswith('ing'):
                tags.append((token, 'VERB'))
            elif token.endswith('ed'):
                tags.append((token, 'VERB'))
            elif token.endswith('ly'):
                tags.append((token, 'ADV'))
            elif token.endswith(('ous', 'ive', 'al', 'able', 'ible', 'ful', 'less')):
                tags.append((token, 'ADJ'))
            elif token.endswith(('tion', 'sion', 'ness', 'ment', 'ity', 'er', 'or')):
                tags.append((token, 'NOUN'))
            elif token[0].isupper():
                tags.append((token, 'PROPN'))
            elif token.isdigit():
                tags.append((token, 'NUM'))
            elif token in ['.', '!', '?']:
                tags.append((token, 'PUNCT'))
            elif i < len(tokens) - 1 and tokens[i+1] in ['is', 'are', 'was', 'were']:
                tags.append((token, 'NOUN'))
            else:
                tags.append((token, 'NOUN'))  # Default
        return tags
    
    def extract_entities(self, tokens: List[str]) -> Dict[str, List[str]]:
        """Extract simple named entities"""
        entities = {'PERSON': [], 'LOCATION': [], 'ORG': [], 'MISC': []}
        
        # Capitalized sequences
        current = []
        for token in tokens:
            if token[0].isupper() and token not in self.stop_words:
                current.append(token)
            else:
                if len(current) >= 1:
                    entities['MISC'].append(' '.join(current))
                    current = []
        if current:
            entities['MISC'].append(' '.join(current))
        
        # Known patterns
        text = ' '.join(tokens)
        if re.search(r'\b(?:mr|mrs|dr|prof)\.?\s+\w+', text, re.I):
            match = re.search(r'\b(?:mr|mrs|dr|prof)\.?\s+(\w+)', text, re.I)
            if match:
                entities['PERSON'].append(match.group(1))
        
        return entities
    
    def update_stats(self, tokens: List[str]):
        """Update n-gram statistics"""
        for token in tokens:
            self.word_freq[token] += 1
        
        for i in range(len(tokens) - 1):
            self.bigrams[tokens[i]][tokens[i+1]] += 1
        
        for i in range(len(tokens) - 2):
            self.trigrams[(tokens[i], tokens[i+1])][tokens[i+2]] += 1
    
    def predict_next(self, tokens: List[str], k=3) -> List[Tuple[str, float]]:
        """N-gram based prediction"""
        if len(tokens) >= 2:
            key = (tokens[-2], tokens[-1])
            if key in self.trigrams:
                total = sum(self.trigrams[key].values())
                return [(w, c/total) for w, c in self.trigrams[key].most_common(k)]
        
        if tokens:
            last = tokens[-1]
            if last in self.bigrams:
                total = sum(self.bigrams[last].values())
                return [(w, c/total) for w, c in self.bigrams[last].most_common(k)]
        
        return [(w, c/sum(self.word_freq.values())) 
                for w, c in self.word_freq.most_common(k)]
    
    def similarity(self, text1: str, text2: str) -> float:
        """Jaccard similarity with TF-IDF weighting"""
        t1 = set(text1.lower().split()) - self.stop_words
        t2 = set(text2.lower().split()) - self.stop_words
        
        if not t1 or not t2:
            return 0.0
        
        intersection = t1 & t2
        union = t1 | t2
        
        return len(intersection) / len(union)

# ==================== CREATIVE GENERATOR ====================

class CreativeGenerator:
    """Template + HDV + CA based text generation"""
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
        """Generate creative text"""
        # Get related words from HDV
        related = []
        for word in seed_words:
            related.extend([w for w, _ in self.hdv.get_related(word, k=5)])
        
        # Add random words for creativity
        all_words = list(set(seed_words + related))
        if len(all_words) < 5:
            all_words.extend(['mystery', 'shadow', 'dream', 'echo', 'void'])
        
        # Use CA to select
        self.ca.step(15)
        ca_state = self.ca.features()
        
        # Select template
        template_idx = int(np.sum(ca_state)) % len(self.templates)
        template = self.templates[template_idx]
        
        # Fill slots
        words = all_words.copy()
        random.shuffle(words)
        
        result = template
        slots = ['adj', 'noun', 'verb', 'noun2', 'adj2', 'noun3', 'Noun']
        
        for i, slot in enumerate(slots):
            if '{' + slot + '}' in result:
                if i < len(words):
                    word = words[i]
                else:
                    word = random.choice(all_words)
                
                if slot == 'Noun':
                    word = word.capitalize()
                elif slot.startswith('adj') and not word.endswith(('ous', 'ive', 'al', 'ful', 'less')):
                    word = word + 'ish' if random.random() < 0.3 else word
                
                result = result.replace('{' + slot + '}', word, 1)
        
        return result
    
    def generate_poem(self, seed_words: List[str], lines=4) -> str:
        """Generate a short poem"""
        poems = []
        for _ in range(lines):
            poems.append(self.generate(seed_words, creativity=0.9))
        return '\n'.join(poems)

# ==================== ANALOGICAL REASONER ====================

class AnalogicalReasoner:
    """Solve A:B :: C:? problems"""
    def __init__(self, hdv: HDV):
        self.hdv = hdv
        self.analogies = []
    
    def add_analogy(self, a: str, b: str, relation: str):
        vec_a = self.hdv._vec(a)
        vec_b = self.hdv._vec(b)
        self.analogies.append({
            "pair": (a, b),
            "relation": relation,
            "transform": vec_b.astype(np.int16) - vec_a.astype(np.int16)
        })
    
    def solve(self, a: str, b: str, c: str) -> Optional[str]:
        """If a:b :: c:?, find ?"""
        # Find transform
        transform = None
        for analogy in self.analogies:
            if analogy["pair"] == (a, b):
                transform = analogy["transform"]
                break
        
        if transform is None:
            # Try reverse
            for analogy in self.analogies:
                if analogy["pair"] == (b, a):
                    transform = -analogy["transform"]
                    break
        
        if transform is None:
            return None
        
        # Apply to c
        vec_c = self.hdv._vec(c).astype(np.int16)
        target = vec_c + transform
        
        # Find closest
        best_word = None
        best_sim = -1
        for word, vec in self.hdv.symbols.items():
            if word not in [a, b, c]:
                sim = float(np.sum(target * vec.astype(np.int16))) / self.hdv.dim
                if sim > best_sim:
                    best_sim = sim
                    best_word = word
        
        return best_word if best_sim > 0.2 else None

# ==================== MAIN AI CLASS ====================

class SymbolicAI:
    def __init__(self):
        self.hdv = HDV()
        self.lcs = LCS()
        self.ca = CA()
        self.ps = ProductionSystem()
        self.nlp = NLPProcessor()
        self.creative = CreativeGenerator(self.hdv, self.ca, self.nlp)
        self.analogies = AnalogicalReasoner(self.hdv)
        
        self.meta_w = {"hdv": 0.35, "lcs": 0.25, "ca": 0.1, "nlp": 0.2, "ps": 0.1}
        self.interactions = 0
        self.conversation_history = []
        
        # Initialize database
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY,
                topic TEXT,
                content TEXT,
                source TEXT,
                priority INTEGER DEFAULT 5,
                created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS training_log (
                id INTEGER PRIMARY KEY,
                input TEXT,
                output TEXT,
                source TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.db.commit()
        
        # Load default knowledge
        self._init_defaults()
    
    def _init_defaults(self):
        """Initialize with default greetings and knowledge"""
        # Greetings
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
        
        # Basic facts
        facts = [
            ("name", "I am Symbolic AI, a learning system."),
            ("creator", "I was created by a curious developer."),
            ("purpose", "I learn from conversations and try to be helpful."),
        ]
        
        for topic, content in facts:
            self.add_knowledge(topic, content, priority=8)
        
        # Default production rules
        self.ps.add_rule(["human"], "mortal", 0.95)
        self.ps.add_rule(["alive"], "human", 0.7)
        self.ps.add_rule(["mortal"], "will_die", 0.9)
    
    def _update_meta_weights(self, source: str, success: bool):
        """Reinforcement learning for meta-weights"""
        lr = 0.05
        if success:
            self.meta_w[source] = min(0.6, self.meta_w.get(source, 0.2) + lr)
        else:
            self.meta_w[source] = max(0.05, self.meta_w.get(source, 0.2) - lr)
        
        # Normalize
        total = sum(self.meta_w.values())
        self.meta_w = {k: v/total for k, v in self.meta_w.items()}
    
    def think(self, query: str) -> Result:
        """Main inference pipeline"""
        self.interactions += 1
        tokens = self.nlp.tokenize(query)
        
        if not tokens:
            return Result("I didn't understand that.", 0.1, "error")
        
        # Update NLP stats
        self.nlp.update_stats(tokens)
        
        # Get predictions from all subsystems
        candidates = {}
        
        # 1. HDV semantic memory
        hdv_preds = self.hdv.predict(tokens, k=3)
        for text, conf in hdv_preds:
            candidates[text] = candidates.get(text, 0) + conf * self.meta_w["hdv"]
        
        # 2. LCS rules
        lcs_preds = self.lcs.predict(tokens)
        for text, conf in lcs_preds:
            candidates[text] = candidates.get(text, 0) + conf * self.meta_w["lcs"]
        
        # 3. NLP n-grams
        nlp_preds = self.nlp.predict_next(tokens, k=3)
        for text, conf in nlp_preds:
            candidates[text] = candidates.get(text, 0) + conf * self.meta_w["nlp"]
        
        # 4. Production system
        ps_result = self._try_production_system(query)
        if ps_result:
            candidates[ps_result] = candidates.get(ps_result, 0) + 0.5 * self.meta_w["ps"]
        
        # 5. Check database
        db_result = self._search_knowledge(query)
        if db_result:
            candidates[db_result] = candidates.get(db_result, 0) + 0.4
        
        # Select best
        if not candidates:
            # Creative fallback
            creative = self.creative.generate(tokens[:3])
            return Result(creative, 0.3, "creative", {"fallback": True})
        
        best = max(candidates.items(), key=lambda x: x[1])
        conf = min(best[1], 1.0)
        
        # Learn from this interaction
        self._learn(query, best[0], tokens)
        
        return Result(best[0], conf, "ensemble", {
            "candidates": len(candidates),
            "tokens": len(tokens)
        })
    
    def _try_production_system(self, query: str) -> Optional[str]:
        """Try to answer using logical inference"""
        tokens = self.nlp.tokenize(query)
        
        # Add tokens as facts
        for token in tokens:
            if token not in self.nlp.stop_words:
                self.ps.add_fact(token)
        
        # Check for known answers
        for fact in self.ps.facts:
            if fact in self.hdv.symbols:
                # This fact has associations
                pass
        
        # Simple pattern: "what is X"
        if "what" in tokens and "is" in tokens:
            idx = tokens.index("is")
            if idx + 1 < len(tokens):
                subject = tokens[idx + 1]
                # Check facts
                for fact in self.ps.facts:
                    if subject in fact:
                        return f"Based on my knowledge, {subject} relates to {fact}."
        
        return None
    
    def _search_knowledge(self, query: str) -> Optional[str]:
        """Search knowledge base"""
        cursor = self.db.execute(
            "SELECT content FROM knowledge WHERE topic LIKE ? OR content LIKE ? ORDER BY priority DESC LIMIT 1",
            (f"%{query}%", f"%{query}%")
        )
        result = cursor.fetchone()
        return result[0] if result else None
    
    def _learn(self, query: str, response: str, tokens: List[str]):
        """Learn from interaction"""
        response_tokens = self.nlp.tokenize(response)
        
        if response_tokens:
            # HDV learning
            for i in range(len(tokens)):
                self.hdv.learn(tokens[:i+1], response_tokens[0], strength=0.5)
            
            # LCS learning
            self.lcs.learn(tokens, response_tokens[0], reward=0.5)
        
        # Evolve CA occasionally
        if self.interactions % 20 == 0:
            self.ca.step(10)
            self.ca.evolve_if_stable()
        
        # Log interaction
        self.db.execute(
            "INSERT INTO training_log (input, output, source) VALUES (?, ?, ?)",
            (query, response, "conversation")
        )
        self.db.commit()
        
        # Update conversation history
        self.conversation_history.append({
            "user": query,
            "ai": response,
            "timestamp": self.interactions
        })
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-25:]
    
    def teach(self, question: str, answer: str, source="user") -> Dict:
        """Explicit teaching"""
        q_tokens = self.nlp.tokenize(question)
        a_tokens = self.nlp.tokenize(answer)
        
        # Store in knowledge base
        topic = q_tokens[0] if q_tokens else "general"
        self.db.execute(
            "INSERT INTO knowledge (topic, content, source, priority) VALUES (?, ?, ?, ?)",
            (topic, answer, source, 7 if source == "user" else 5)
        )
        self.db.commit()
        
        if a_tokens:
            # HDV: learn full context
            for i in range(len(q_tokens)):
                self.hdv.learn(q_tokens[:i+1], a_tokens[0], strength=1.0)
            
            # Also learn response structure
            for i in range(len(a_tokens) - 1):
                self.hdv.learn([a_tokens[i]], a_tokens[i+1], strength=0.8)
            
            # LCS: learn rule
            self.lcs.learn(q_tokens, a_tokens[0], reward=1.0)
            
            # NLP stats
            self.nlp.update_stats(q_tokens)
            self.nlp.update_stats(a_tokens)
        
        return {
            "status": "Learned!",
            "hdv_symbols": len(self.hdv.symbols),
            "lcs_rules": len(self.lcs.rules),
            "db_entries": self.db.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
        }
    
    def train_from_text(self, text: str, source="file") -> Dict:
        """Train from arbitrary text (like .rb files)"""
        lines = text.split('\n')
        trained = 0
        
        # Try to extract Q&A pairs or learn patterns
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Learn the line itself
            tokens = self.nlp.tokenize(line)
            if len(tokens) >= 3:
                self.nlp.update_stats(tokens)
                
                # Learn token transitions
                for i in range(len(tokens) - 1):
                    self.hdv.learn([tokens[i]], tokens[i+1], strength=0.3)
                
                trained += 1
        
        # Also try to learn Ruby syntax patterns
        if '.rb' in source or 'ruby' in source.lower():
            self._learn_ruby_patterns(text)
        
        return {
            "lines_processed": len(lines),
            "patterns_learned": trained,
            "source": source
        }
    
    def _learn_ruby_patterns(self, code: str):
        """Learn Ruby-specific patterns"""
        # Common Ruby keywords and patterns
        ruby_patterns = [
            (["def"], "method"),
            (["class"], "Class"),
            (["end"], "block_end"),
            (["if"], "condition"),
            (["else"], "alternative"),
            (["puts"], "output"),
            (["require"], "import"),
        ]
        
        for context, target in ruby_patterns:
            self.hdv.learn(context, target, strength=0.5)
            self.lcs.learn(context, target, reward=0.5)
    
    def generate_creative(self, seed: str, poem=False) -> str:
        """Generate creative text"""
        tokens = self.nlp.tokenize(seed)
        if poem:
            return self.creative.generate_poem(tokens)
        return self.creative.generate(tokens)
    
    def solve_analogy(self, a: str, b: str, c: str) -> Optional[str]:
        """Solve A:B :: C:?"""
        return self.analogies.solve(a, b, c)
    
    def add_analogy(self, a: str, b: str, relation: str):
        """Teach an analogy"""
        self.analogies.add_analogy(a, b, relation)
    
    def reason(self, facts: List[str]) -> List[str]:
        """Add facts and run inference"""
        for fact in facts:
            self.ps.add_fact(fact)
        return self.ps.infer()
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
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
            "meta_weights": self.meta_w
        }
    
    def get_rules(self) -> List[Dict]:
        """Get LCS rules"""
        return self.lcs.get_rules()
    
    def get_facts(self) -> List[str]:
        """Get production system facts"""
        return list(self.ps.facts)
    
    def save_state(self) -> str:
        """Save AI state"""
        return self.hdv.save()
    
    def load_state(self, state: str):
        """Load AI state"""
        self.hdv.load(state)
