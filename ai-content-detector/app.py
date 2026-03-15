from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import tempfile
from werkzeug.utils import secure_filename
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import re
from collections import defaultdict
import math
import json
import random

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'docx', 'pdf'}

class AIContentDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load multiple models for ensemble approach
        self.models = {}
        self.tokenizers = {}
        
        # Model 1: RoBERTa-base for AI detection
        try:
            model_name = "roberta-base"
            self.tokenizers['roberta'] = AutoTokenizer.from_pretrained(model_name)
            self.models['roberta'] = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2
            ).to(self.device)
        except Exception as e:
            print(f"Error loading RoBERTa: {e}")
        
        # Model 2: DistilBERT for faster processing
        try:
            model_name = "distilbert-base-uncased"
            self.tokenizers['distilbert'] = AutoTokenizer.from_pretrained(model_name)
            self.models['distilbert'] = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2
            ).to(self.device)
        except Exception as e:
            print(f"Error loading DistilBERT: {e}")
    
    def extract_text_features(self, text):
        """Extract linguistic features that indicate AI-generated content"""
        features = {}
        
        # Perplexity-based features
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences:
            avg_sentence_length = np.mean([len(s.split()) for s in sentences])
            features['avg_sentence_length'] = avg_sentence_length
            
            # Vocabulary diversity
            all_words = text.lower().split()
            unique_words = set(all_words)
            features['vocab_diversity'] = len(unique_words) / len(all_words) if all_words else 0
            
            # Punctuation patterns
            punctuation_count = len(re.findall(r'[^\w\s]', text))
            features['punctuation_ratio'] = punctuation_count / len(text) if text else 0
            
            # Repetition patterns
            word_freq = defaultdict(int)
            for word in all_words:
                word_freq[word] += 1
            if word_freq:
                features['repetition_score'] = np.std(list(word_freq.values())) / np.mean(list(word_freq.values()))
            else:
                features['repetition_score'] = 0
        
        return features
    
    def analyze_with_model(self, text, model_name):
        """Analyze text with a specific model"""
        if model_name not in self.models or model_name not in self.tokenizers:
            return 0.5  # Neutral score if model not available
        
        try:
            tokenizer = self.tokenizers[model_name]
            model = self.models[model_name]
            
            # Tokenize and prepare input
            inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                             max_length=512, padding=True).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                ai_probability = probabilities[0][1].item()  # Assuming class 1 is AI-generated
            
            return ai_probability
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            return 0.5
    
    def detect_ai_content(self, text):
        """Main detection function using ensemble approach"""
        # Limit to 1200 words
        words = text.split()
        if len(words) > 1200:
            text = ' '.join(words[:1200])
        
        if not text or len(text.strip()) < 50:
            return {
                'ai_probability': 0.1,
                'confidence': 'low',
                'analysis': 'Text too short for reliable analysis',
                'features': {},
                'model_scores': {}
            }
        
        # Extract linguistic features
        features = self.extract_text_features(text)
        
        # Get predictions from multiple models
        model_scores = {}
        for model_name in self.models.keys():
            score = self.analyze_with_model(text, model_name)
            model_scores[model_name] = score
        
        # Ensemble prediction
        if model_scores:
            ai_probability = np.mean(list(model_scores.values()))
        else:
            # Fallback to feature-based detection
            ai_probability = self.feature_based_detection(features)
        
        # Determine confidence level
        confidence = self.calculate_confidence(model_scores, features)
        
        # Generate analysis
        analysis = self.generate_analysis(ai_probability, features, model_scores)
        
        return {
            'ai_probability': round(ai_probability, 3),
            'confidence': confidence,
            'analysis': analysis,
            'features': features,
            'model_scores': model_scores,
            'text_length': len(text),
            'word_count': len(text.split())
        }
    
    def feature_based_detection(self, features):
        """Fallback detection using only linguistic features"""
        score = 0.5  # Base score
        
        # Adjust score based on features
        if 'avg_sentence_length' in features:
            # Very uniform sentence lengths can indicate AI
            if features['avg_sentence_length'] > 25:
                score += 0.1
            elif features['avg_sentence_length'] < 10:
                score -= 0.05
        
        if 'vocab_diversity' in features:
            # High vocabulary diversity might indicate human writing
            if features['vocab_diversity'] > 0.7:
                score -= 0.1
            elif features['vocab_diversity'] < 0.4:
                score += 0.1
        
        if 'repetition_score' in features:
            # Low repetition might indicate AI
            if features['repetition_score'] < 0.5:
                score += 0.05
        
        return max(0, min(1, score))
    
    def calculate_confidence(self, model_scores, features):
        """Calculate confidence level of the detection"""
        if not model_scores:
            return 'low'
        
        # Check agreement between models
        scores = list(model_scores.values())
        if len(scores) > 1:
            std_dev = np.std(scores)
            if std_dev < 0.1:
                return 'high'
            elif std_dev < 0.2:
                return 'medium'
        
        # Consider text length
        if 'text_length' in locals() and len(locals()['text']) > 1000:
            return 'medium'
        
        return 'low'
    
    def generate_analysis(self, ai_probability, features, model_scores):
        """Generate detailed analysis report"""
        analysis_parts = []
        
        # Overall assessment
        if ai_probability < 0.3:
            analysis_parts.append("The text appears to be primarily human-written.")
        elif ai_probability < 0.7:
            analysis_parts.append("The text shows mixed characteristics, with both human and AI-like elements.")
        else:
            analysis_parts.append("The text exhibits strong indicators of AI-generated content.")
        
        # Feature-based insights
        if 'avg_sentence_length' in features:
            if features['avg_sentence_length'] > 20:
                analysis_parts.append("Sentences are consistently long, which can be characteristic of AI writing.")
        
        if 'vocab_diversity' in features:
            if features['vocab_diversity'] < 0.5:
                analysis_parts.append("Limited vocabulary diversity detected, common in AI-generated text.")
        
        # Model agreement
        if len(model_scores) > 1:
            scores = list(model_scores.values())
            agreement = 1 - np.std(scores)
            if agreement > 0.8:
                analysis_parts.append("Multiple detection models show strong agreement in their assessment.")
        
        return " ".join(analysis_parts)

class TextHumanizer:
    def __init__(self):
        # Human-like transition words and phrases
        self.transition_words = [
            "however", "therefore", "moreover", "furthermore", "consequently",
            "nevertheless", "nonetheless", "meanwhile", "in addition", "on the other hand",
            "as a result", "for this reason", "in fact", "indeed", "actually"
        ]
        
        # Human-like sentence starters
        self.sentence_starters = [
            "It's interesting to note that", "One could argue that", "It's worth mentioning that",
            "As it turns out", "Surprisingly", "Notably", "Interestingly enough",
            "When you think about it", "If you consider", "Looking at it from this perspective"
        ]
        
        # Contractions to make text more natural
        self.contractions = {
            "do not": "don't", "does not": "doesn't", "did not": "didn't",
            "will not": "won't", "would not": "wouldn't", "cannot": "can't",
            "could not": "couldn't", "should not": "shouldn't", "might not": "mightn't",
            "must not": "mustn't", "are not": "aren't", "is not": "isn't",
            "was not": "wasn't", "were not": "weren't", "have not": "haven't",
            "has not": "hasn't", "had not": "hadn't", "I am": "I'm",
            "you are": "you're", "we are": "we're", "they are": "they're"
        }
        
        # Filler words that humans use
        self.filler_words = [
            "basically", "actually", "literally", "honestly", "frankly",
            "personally", "in my opinion", "I think", "I believe", "it seems"
        ]
    
    def add_variations(self, text):
        """Add human-like variations to the text"""
        # Randomly add some filler words
        sentences = re.split(r'([.!?]+)', text)
        result_sentences = []
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i].strip()
            if sentence and random.random() < 0.15:  # 15% chance to add filler
                filler = random.choice(self.filler_words)
                sentence = f"{filler}, {sentence.lower()}"
            result_sentences.append(sentence)
            if i + 1 < len(sentences):
                result_sentences.append(sentences[i + 1])
        
        return ''.join(result_sentences)
    
    def vary_sentence_structure(self, text):
        """Vary sentence structure to appear more human"""
        sentences = re.split(r'([.!?]+)', text)
        result_sentences = []
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i].strip()
            if sentence:
                # Randomly add sentence starters
                if random.random() < 0.1:  # 10% chance
                    starter = random.choice(self.sentence_starters)
                    sentence = f"{starter} {sentence.lower()}"
                
                # Vary sentence length by splitting long sentences
                words = sentence.split()
                if len(words) > 20 and random.random() < 0.3:
                    mid_point = len(words) // 2
                    first_half = ' '.join(words[:mid_point])
                    second_half = ' '.join(words[mid_point:])
                    sentence = f"{first_half}. {second_half}"
                
                result_sentences.append(sentence)
            
            if i + 1 < len(sentences):
                result_sentences.append(sentences[i + 1])
        
        return ''.join(result_sentences)
    
    def add_contractions(self, text):
        """Add contractions to make text more natural"""
        for formal, contraction in self.contractions.items():
            # Only replace if the formal version appears as a whole word
            text = re.sub(r'\b' + re.escape(formal) + r'\b', contraction, text, flags=re.IGNORECASE)
        return text
    
    def add_transitions(self, text):
        """Add transition words between sentences"""
        sentences = re.split(r'([.!?]+)', text)
        result_sentences = []
        
        for i in range(0, len(sentences), 2):
            if i > 0 and sentences[i].strip():  # Not the first sentence
                if random.random() < 0.2:  # 20% chance to add transition
                    transition = random.choice(self.transition_words)
                    sentences[i] = f"{transition}, {sentences[i].lower()}"
            
            result_sentences.append(sentences[i])
            if i + 1 < len(sentences):
                result_sentences.append(sentences[i + 1])
        
        return ''.join(result_sentences)
    
    def add_punctuation_variations(self, text):
        """Add human-like punctuation variations"""
        # Add occasional commas where they might naturally appear
        text = re.sub(r'\b(and|but|or|so|yet)\b', lambda m: 
                     f",{m.group(0)}" if random.random() < 0.1 else m.group(0), text)
        
        # Add occasional em dashes for emphasis
        if random.random() < 0.1:
            text = re.sub(r'\b(\w+)\s+(\w+)\b', r'\1—\2', text, count=1)
        
        return text
    
    def humanize_text(self, text, intensity='medium'):
        """Main humanization function"""
        if not text or len(text.strip()) < 20:
            return text
        
        humanized = text
        
        # Apply different techniques based on intensity
        if intensity == 'light':
            # Light humanization - just add contractions
            humanized = self.add_contractions(humanized)
            humanized = self.add_punctuation_variations(humanized)
        
        elif intensity == 'medium':
            # Medium humanization - add several techniques
            humanized = self.add_contractions(humanized)
            humanized = self.add_transitions(humanized)
            humanized = self.add_punctuation_variations(humanized)
            humanized = self.add_variations(humanized)
        
        elif intensity == 'strong':
            # Strong humanization - apply all techniques
            humanized = self.add_contractions(humanized)
            humanized = self.add_transitions(humanized)
            humanized = self.vary_sentence_structure(humanized)
            humanized = self.add_punctuation_variations(humanized)
            humanized = self.add_variations(humanized)
        
        return humanized.strip()

class SentenceAnalyzer:
    def __init__(self, detector):
        self.detector = detector
        self.humanizer = TextHumanizer()
    
    def analyze_sentence(self, sentence):
        """Analyze a single sentence for AI content"""
        if not sentence or len(sentence.strip()) < 10:
            return None
        
        # Get AI probability for this sentence
        result = self.detector.detect_ai_content(sentence)
        ai_probability = result['ai_probability']
        
        # Categorize confidence level
        if ai_probability < 0.3:
            category = 'low'
            label = 'Slightly'
            color = '#48dbfb'  # Light blue
        elif ai_probability < 0.7:
            category = 'medium'
            label = 'High'
            color = '#feca57'  # Yellow
        else:
            category = 'high'
            label = 'Very High'
            color = '#ff6b6b'  # Red
        
        return {
            'text': sentence.strip(),
            'ai_probability': ai_probability,
            'category': category,
            'label': label,
            'color': color,
            'confidence': result['confidence'],
            'suggestions': self.generate_suggestions(sentence, ai_probability, category)
        }
    
    def generate_suggestions(self, sentence, ai_probability, category):
        """Generate specific suggestions to humanize the sentence"""
        suggestions = []
        
        # Analyze sentence characteristics
        words = sentence.split()
        avg_word_length = sum(len(word.strip('.,!?;:')) for word in words) / len(words) if words else 0
        
        # Length-based suggestions
        if len(words) > 25:
            suggestions.append({
                'type': 'length',
                'suggestion': 'Break this long sentence into shorter, more natural sentences.',
                'example': self.split_long_sentence(sentence)
            })
        elif len(words) < 8:
            suggestions.append({
                'type': 'length',
                'suggestion': 'Consider adding more detail or combining with related ideas.',
                'example': f"Consider expanding: '{sentence}' with more context or examples."
            })
        
        # Formal language suggestions
        formal_words = ['furthermore', 'moreover', 'consequently', 'therefore', 'nevertheless', 'additionally']
        if any(word in sentence.lower() for word in formal_words):
            suggestions.append({
                'type': 'formality',
                'suggestion': 'Replace formal transition words with more natural alternatives.',
                'example': self.replace_formal_transitions(sentence)
            })
        
        # Repetition suggestions
        word_freq = {}
        for word in words:
            clean_word = word.lower().strip('.,!?;:')
            word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
        
        repeated_words = [word for word, count in word_freq.items() if count > 2 and len(word) > 3]
        if repeated_words:
            suggestions.append({
                'type': 'repetition',
                'suggestion': f'Avoid overusing words like: {", ".join(repeated_words[:3])}. Use synonyms or rephrase.',
                'example': self.fix_repetition(sentence, repeated_words[0] if repeated_words else '')
            })
        
        # Structure suggestions
        if sentence.startswith(('In addition', 'Furthermore', 'Moreover', 'Therefore')):
            suggestions.append({
                'type': 'structure',
                'suggestion': 'Start sentences more naturally instead of formal transitions.',
                'example': f"Try starting with: 'Also,' or 'Plus,' or just rephrasing without the formal opener."
            })
        
        # Punctuation suggestions
        if ',' not in sentence and len(words) > 10:
            suggestions.append({
                'type': 'punctuation',
                'suggestion': 'Add commas to create natural pauses and improve readability.',
                'example': self.add_natural_commas(sentence)
            })
        
        # Vocabulary suggestions
        if avg_word_length > 6:
            suggestions.append({
                'type': 'vocabulary',
                'suggestion': 'Use simpler, more common words to sound more natural.',
                'example': self.simplify_vocabulary(sentence)
            })
        
        # Category-specific suggestions
        if category == 'high':
            suggestions.append({
                'type': 'overall',
                'suggestion': 'This sentence shows strong AI characteristics. Consider completely rewriting it in your own words.',
                'example': self.humanizer.humanize_text(sentence, 'strong')
            })
        elif category == 'medium':
            suggestions.append({
                'type': 'overall',
                'suggestion': 'This sentence has some AI-like patterns. Apply medium humanization techniques.',
                'example': self.humanizer.humanize_text(sentence, 'medium')
            })
        
        return suggestions[:4]  # Limit to top 4 suggestions
    
    def split_long_sentence(self, sentence):
        """Split a long sentence into shorter ones"""
        words = sentence.split()
        if len(words) <= 25:
            return sentence
        
        mid_point = len(words) // 2
        first_half = ' '.join(words[:mid_point])
        second_half = ' '.join(words[mid_point:])
        
        return f"{first_half}. {second_half}"
    
    def replace_formal_transitions(self, sentence):
        """Replace formal transitions with natural alternatives"""
        replacements = {
            'furthermore': 'also',
            'moreover': 'plus',
            'consequently': 'so',
            'therefore': 'so',
            'nevertheless': 'still',
            'additionally': 'also'
        }
        
        result = sentence
        for formal, informal in replacements.items():
            if formal in result.lower():
                result = re.sub(r'\b' + re.escape(formal) + r'\b', informal, result, flags=re.IGNORECASE)
                break
        
        return result
    
    def fix_repetition(self, sentence, repeated_word):
        """Fix word repetition in sentence"""
        words = sentence.split()
        result = []
        used_words = set()
        
        for word in words:
            clean_word = word.lower().strip('.,!?;:')
            if clean_word == repeated_word.lower() and clean_word in used_words:
                # Try to find a simple replacement
                alternatives = {
                    'very': 'extremely',
                    'really': 'truly',
                    'good': 'great',
                    'bad': 'poor',
                    'big': 'large',
                    'small': 'tiny'
                }
                if clean_word in alternatives:
                    result.append(alternatives[clean_word])
                else:
                    result.append(word)
            else:
                result.append(word)
            used_words.add(clean_word)
        
        return ' '.join(result)
    
    def add_natural_commas(self, sentence):
        """Add natural commas to sentence"""
        # Simple comma addition for introductory phrases
        if sentence.startswith(('In fact', 'Actually', 'Basically', 'Generally')):
            return sentence.replace(' ', ', ', 1)
        
        # Add comma before conjunctions in long sentences
        conjunctions = ['and', 'but', 'or', 'so', 'yet']
        words = sentence.split()
        
        for i, word in enumerate(words):
            if word.lower() in conjunctions and i > 5:  # Add comma if conjunction appears after 5+ words
                words[i] = f',{word}'
                break
        
        return ' '.join(words)
    
    def simplify_vocabulary(self, sentence):
        """Simplify complex vocabulary"""
        simplifications = {
            'utilize': 'use',
            'demonstrate': 'show',
            'establish': 'create',
            'subsequently': 'then',
            'consequently': 'so',
            'nevertheless': 'still',
            'furthermore': 'also',
            'approximately': 'about',
            'numerous': 'many',
            'sufficient': 'enough'
        }
        
        result = sentence
        for complex_word, simple_word in simplifications.items():
            result = re.sub(r'\b' + re.escape(complex_word) + r'\b', simple_word, result, flags=re.IGNORECASE)
        
        return result
    
    def analyze_text_sentences(self, text):
        """Analyze all sentences in text"""
        if not text:
            return []
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        analyzed_sentences = []
        for sentence in sentences:
            analysis = self.analyze_sentence(sentence)
            if analysis:
                analyzed_sentences.append(analysis)
        
        return analyzed_sentences

import hashlib
from difflib import SequenceMatcher
import requests
from urllib.parse import quote_plus

class PlagiarismChecker:
    def __init__(self):
        self.min_sentence_length = 10
        self.similarity_threshold = 0.6
        self.web_match_threshold = 0.5
        # Google Custom Search API credentials from environment
        self.google_api_key = os.environ.get('GOOGLE_API_KEY')
        self.google_cx = os.environ.get('GOOGLE_CX')  # Custom Search Engine ID
        self.use_google_api = bool(self.google_api_key and self.google_cx)
        
    def extract_key_phrases(self, text, max_phrases=8):
        """Extract key phrases for web search - skip citations"""
        # Remove citations like (Smith, 2020) or [1]
        text_clean = re.sub(r'\([^)]*\d{4}[^)]*\)', '', text)
        text_clean = re.sub(r'\[\d+\]', '', text_clean)
        
        sentences = re.split(r'[.!?]+', text_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30 and len(s.strip()) < 200]
        
        # Sort by length and pick the longest/most substantial sentences
        sentences.sort(key=len, reverse=True)
        return sentences[:max_phrases]
    
    def calculate_similarity(self, text1, text2):
        """Calculate similarity ratio between two texts using SequenceMatcher"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def search_google_api(self, query):
        """Search using Google Custom Search API - more accurate and reliable"""
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.google_api_key,
                'cx': self.google_cx,
                'q': query[:200],
                'num': 10  # Get up to 10 results
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get('items', []):
                    title = item.get('title', 'Unknown')
                    snippet = item.get('snippet', '')
                    link = item.get('link', '')
                    
                    if snippet and len(snippet) > 30:
                        results.append({
                            'title': title,
                            'snippet': snippet,
                            'url': link
                        })
                
                return results
            else:
                print(f"Google API error: {response.status_code} - {response.text[:200]}")
                return []
                
        except Exception as e:
            print(f"Google API search error: {e}")
            return []
    
    def search_web_for_similar(self, query):
        """Search web for similar content - uses Google API if available, else DuckDuckGo"""
        # Try Google API first if configured
        if self.use_google_api:
            print(f"Using Google API for search: {query[:50]}...")
            results = self.search_google_api(query)
            if results:
                return results
            print("Google API returned no results, falling back to DuckDuckGo...")
        
        # Fallback to DuckDuckGo scraping
        try:
            print(f"Using DuckDuckGo for search: {query[:50]}...")
            search_query = quote_plus(query[:150])
            url = f"https://html.duckduckgo.com/html/?q={search_query}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                results = []
                for result in soup.find_all('div', class_='result')[:5]:
                    snippet_elem = result.find('a', class_='result__snippet')
                    title_elem = result.find('a', class_='result__a')
                    
                    if snippet_elem and title_elem:
                        snippet = snippet_elem.get_text()
                        title = title_elem.get_text()
                        link = title_elem.get('href', '')
                        
                        if snippet and len(snippet) > 50:
                            results.append({
                                'title': title,
                                'snippet': snippet,
                                'url': link
                            })
                
                return results
            
            return []
        except Exception as e:
            print(f"Web search error: {e}")
            return []
    
    def detect_citation(self, text, sentence_start, sentence_end):
        """Detect if a sentence has proper citation"""
        # Look for citation patterns
        citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Smith, 2020)
            r'\[\d+\]',              # [1]
            r'\"[^\"]+\"',            # "quoted text"
            r'\"[^\"]+\"\s*\(',     # "quoted text" (citation)
            r'said|stated|according to|cited|quoted|references?|cites?',
            r'\b(?:et\s+al|pp?\.\s*\d+|vol\.\s*\d+)',
        ]
        
        # Check context around the sentence (100 chars before and after)
        context_start = max(0, sentence_start - 100)
        context_end = min(len(text), sentence_end + 100)
        context = text[context_start:context_end]
        
        for pattern in citation_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                return True
        
        return False
    
    def find_text_in_sources(self, text, sources):
        """Find matching text in web sources"""
        matches = []
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > self.min_sentence_length]
        
        for i, sentence in enumerate(sentences[:20]):  # Check first 20 sentences
            best_match = None
            best_similarity = 0
            
            for source in sources:
                snippet = source.get('snippet', '')
                if not snippet:
                    continue
                    
                # Check similarity with snippet
                similarity = self.calculate_similarity(sentence, snippet)
                
                # Also check for partial matches (phrase matching)
                words = sentence.lower().split()
                if len(words) > 5:
                    # Check for 5-grams
                    for j in range(len(words) - 4):
                        phrase = ' '.join(words[j:j+5])
                        if phrase in snippet.lower():
                            similarity = max(similarity, 0.6)
                            break
                
                if similarity > self.web_match_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = source
            
            if best_match:
                # Check if cited
                char_pos = text.find(sentence)
                is_cited = self.detect_citation(text, char_pos, char_pos + len(sentence)) if char_pos >= 0 else False
                
                matches.append({
                    'sentence': sentence,
                    'index': i,
                    'similarity': round(best_similarity * 100, 1),
                    'source_title': best_match.get('title', 'Unknown Source'),
                    'source_url': best_match.get('url', ''),
                    'source_snippet': best_match.get('snippet', '')[:200],
                    'is_cited': is_cited,
                    'match_type': 'web_source'
                })
        
        return matches
    
    def check_internal_similarity(self, text):
        """Check for internal repetition within the text"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > self.min_sentence_length]
        
        suspicious = []
        
        for i, sentence in enumerate(sentences):
            for j, other in enumerate(sentences):
                if i != j:
                    similarity = self.calculate_similarity(sentence, other)
                    if similarity > self.similarity_threshold:
                        suspicious.append({
                            'sentence': sentence,
                            'index': i,
                            'similarity': round(similarity * 100, 1),
                            'matched_sentence': other,
                            'match_type': 'internal_repetition',
                            'is_cited': False
                        })
                        break
        
        return suspicious
    
    def check_plagiarism(self, text):
        """Main plagiarism detection function with Turnitin-style results"""
        if not text or len(text.strip()) < 50:
            return {
                'similarity_percentage': 0,
                'originality_score': 100,
                'analysis': 'Text too short for plagiarism analysis',
                'match_groups': {},
                'sources': [],
                'matches': [],
                'total_matches': 0
            }
        
        # Limit text for performance
        words = text.split()
        if len(words) > 1200:
            text = ' '.join(words[:1200])
        
        # Extract key phrases for web search
        key_phrases = self.extract_key_phrases(text)
        
        # Search web for each key phrase
        all_sources = []
        for phrase in key_phrases[:3]:  # Search top 3 phrases
            sources = self.search_web_for_similar(phrase)
            all_sources.extend(sources)
        
        # Remove duplicate sources
        seen_urls = set()
        unique_sources = []
        for source in all_sources:
            url = source.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(source)
        
        # Find matches in sources
        web_matches = self.find_text_in_sources(text, unique_sources)
        
        # Check internal similarity
        internal_matches = self.check_internal_similarity(text)
        
        # Combine all matches
        all_matches = web_matches + internal_matches
        
        # Categorize matches
        not_cited = [m for m in web_matches if not m['is_cited']]
        cited = [m for m in web_matches if m['is_cited']]
        internal = internal_matches
        
        # Calculate similarity percentage
        total_sentences = len(re.split(r'[.!?]+', text))
        matched_sentences = len(set([m['index'] for m in all_matches]))
        
        if total_sentences > 0:
            similarity_percentage = (matched_sentences / total_sentences) * 100
        else:
            similarity_percentage = 0
        
        # Cap at 100%
        similarity_percentage = min(100, similarity_percentage)
        originality_score = round(100 - similarity_percentage, 1)
        
        # Generate analysis based on similarity
        if similarity_percentage < 10:
            analysis = "Your text appears to be highly original with minimal detected similarities."
        elif similarity_percentage < 25:
            analysis = f"Your text shows {similarity_percentage:.1f}% similarity with external sources. Review uncited matches."
        elif similarity_percentage < 50:
            analysis = f"Your text contains {similarity_percentage:.1f}% similar content. Careful review recommended."
        else:
            analysis = f"High similarity detected ({similarity_percentage:.1f}%). Significant review needed."
        
        # Build sources summary for display
        sources_summary = []
        for source in unique_sources[:10]:
            # Count matches from this source
            source_matches = [m for m in web_matches if m.get('source_title') == source.get('title')]
            if source_matches:
                sources_summary.append({
                    'title': source.get('title', 'Unknown'),
                    'url': source.get('url', ''),
                    'match_count': len(source_matches),
                    'similarity': max([m['similarity'] for m in source_matches]),
                    'type': 'student_papers' if 'edu' in source.get('url', '') else 'web'
                })
        
        # Sort sources by match count
        sources_summary.sort(key=lambda x: x['match_count'], reverse=True)
        
        return {
            'similarity_percentage': round(similarity_percentage, 1),
            'originality_score': originality_score,
            'analysis': analysis,
            'match_groups': {
                'not_cited': {
                    'count': len(not_cited),
                    'percentage': round((len(not_cited) / max(total_sentences, 1)) * 100, 1),
                    'matches': not_cited[:5]
                },
                'cited': {
                    'count': len(cited),
                    'percentage': round((len(cited) / max(total_sentences, 1)) * 100, 1),
                    'matches': cited[:5]
                },
                'internal': {
                    'count': len(internal),
                    'percentage': round((len(internal) / max(total_sentences, 1)) * 100, 1),
                    'matches': internal[:5]
                }
            },
            'sources': sources_summary[:8],
            'total_sources': len(sources_summary),
            'total_matches': len(all_matches),
            'text_preview': text[:500] + ('...' if len(text) > 500 else '')
        }

# Initialize detector, humanizer, sentence analyzer, and plagiarism checker
detector = AIContentDetector()
humanizer = TextHumanizer()
sentence_analyzer = SentenceAnalyzer(detector)
plagiarism_checker = PlagiarismChecker()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path):
    """Extract text from uploaded file"""
    file_ext = file_path.rsplit('.', 1)[1].lower()
    
    if file_ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_ext == 'docx':
        try:
            import docx
            doc = docx.Document(file_path)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        except ImportError:
            return "Error: python-docx library not installed"
    elif file_ext == 'pdf':
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except ImportError:
            return "Error: PyPDF2 library not installed"
    
    return ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get text from request
        text = request.form.get('text', '')
        
        # Check if file was uploaded
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Extract text from file
                file_text = extract_text_from_file(file_path)
                text = text + '\n\n' + file_text if text else file_text
                
                # Clean up temporary file
                os.remove(file_path)
        
        if not text.strip():
            return jsonify({'error': 'No text provided for analysis'}), 400
        
        # Perform AI content detection
        result = detector.detect_ai_content(text)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/analyze_sentences', methods=['POST'])
def analyze_sentences():
    try:
        # Get text from request
        text = request.form.get('text', '')
        
        # Check if file was uploaded
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Extract text from file
                file_text = extract_text_from_file(file_path)
                text = text + '\n\n' + file_text if text else file_text
                
                # Clean up temporary file
                os.remove(file_path)
        
        if not text.strip():
            return jsonify({'error': 'No text provided for analysis'}), 400
        
        # Limit to 1200 words
        words = text.split()
        if len(words) > 1200:
            text = ' '.join(words[:1200])
        
        # Analyze sentences
        analyzed_sentences = sentence_analyzer.analyze_text_sentences(text)
        
        # Get overall text analysis
        overall_analysis = detector.detect_ai_content(text)
        
        # Categorize sentences by confidence level
        categories = {
            'low': [],
            'medium': [],
            'high': []
        }
        
        for sentence in analyzed_sentences:
            categories[sentence['category']].append(sentence)
        
        return jsonify({
            'overall_analysis': overall_analysis,
            'sentences': analyzed_sentences,
            'categories': categories,
            'total_sentences': len(analyzed_sentences),
            'text_length': len(text),
            'word_count': len(text.split())
        })
        
    except Exception as e:
        return jsonify({'error': f'Sentence analysis failed: {str(e)}'}), 500

@app.route('/check_plagiarism', methods=['POST'])
def check_plagiarism():
    try:
        # Get text from request
        text = request.form.get('text', '')
        
        # Check if file was uploaded
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Extract text from file
                file_text = extract_text_from_file(file_path)
                text = text + '\n\n' + file_text if text else file_text
                
                # Clean up temporary file
                os.remove(file_path)
        
        if not text.strip():
            return jsonify({'error': 'No text provided for plagiarism check'}), 400
        
        # Limit to 1200 words
        words = text.split()
        if len(words) > 1200:
            text = ' '.join(words[:1200])
        
        # Perform plagiarism detection
        result = plagiarism_checker.check_plagiarism(text)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Plagiarism check failed: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'models_loaded': len(detector.models)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
