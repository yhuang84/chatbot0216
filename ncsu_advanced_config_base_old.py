#!/usr/bin/env python3
"""
NCSU Advanced Research Assistant - OPTIMIZED VERSION
====================================================

Performance Optimizations:
- Parallel content extraction and LLM grading
- Content truncation for faster grading
- Separate models for grading (fast) vs answers (quality)
- Early stopping when high-quality content found
- Caching for pages and grades
- Reduced default parameters for speed
- Token limit management to prevent context overflow

KEY CHANGE: research() no longer calls generate_answer() (Step 5 removed).
The UI calls the streaming API directly so the answer appears immediately
after extraction instead of waiting for a full blocking LLM call first.

Usage:
    1. Edit the config dictionary in main()
    2. Run: python ncsu_advanced_config_base.py
"""

import json
import os
import sys
import yaml
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from scraper.ncsu_scraper import NCSUScraper
from scraper.content_aggregator import ContentAggregator
from scraper.models import ScrapingConfig
from utils.logger import setup_logger


# ========================================
# 🔧 LLM MODEL CONFIGURATION
# ========================================

ANSWER_LLM_PROVIDER = 'openai'
ANSWER_LLM_MODEL = 'gpt-4o'        # gpt-5-mini does not exist — use gpt-4o or gpt-4o-mini
ANSWER_LLM_TEMPERATURE = 0.1            # lower = more factual, less hallucination
ANSWER_LLM_MAX_TOKENS = 4000
MAX_CONTEXT_TOKENS = 120000             # CRITICAL: was 4000 → model received almost NO source content

GRADING_LLM_PROVIDER = 'openai'
GRADING_LLM_MODEL = 'gpt-4o-mini'      # gpt-5-mini does not exist
GRADING_LLM_TEMPERATURE = 0.0          # grading should be fully deterministic
GRADING_LLM_MAX_TOKENS = 10
MAX_GRADING_CONTENT_LENGTH = 2000


class ContentCache:
    """Thread-safe cache for scraped content and grades"""

    def __init__(self):
        self.page_cache = {}
        self.grade_cache = {}
        self.lock = threading.Lock()

    def get_page(self, url: str) -> Optional[str]:
        with self.lock:
            return self.page_cache.get(url)

    def set_page(self, url: str, content: str):
        with self.lock:
            self.page_cache[url] = content

    def get_grade(self, content: str, query: str) -> Optional[float]:
        content_hash = hashlib.md5(content.encode()).hexdigest()
        with self.lock:
            return self.grade_cache.get((content_hash, query))

    def set_grade(self, content: str, query: str, score: float):
        content_hash = hashlib.md5(content.encode()).hexdigest()
        with self.lock:
            self.grade_cache[(content_hash, query)] = score


class LLMProvider:
    """Base class for LLM providers"""

    def __init__(self, provider_name: str, model: str = None, temperature: float = 0.7, max_tokens: int = 1000):
        self.provider_name = provider_name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_response(self, prompt: str) -> str:
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing"""

    def __init__(self):
        super().__init__("mock", "mock-model", 0.7, 1000)

    def generate_response(self, prompt: str) -> str:
        if "grade" in prompt.lower() or "relevance" in prompt.lower():
            return "0.75"
        query_text = prompt.split('Question:')[-1].split('Content:')[0].strip() if 'Question:' in prompt else 'your query'
        content_preview = prompt.split('Content:')[-1][:200] if 'Content:' in prompt else 'Content analyzed'
        return f"""Based on NCSU website analysis for: "{query_text}"

**Content Analysis:**
- Analyzed {len(prompt.split())} words from NCSU website
- Applied relevance filtering and grading
- Selected most relevant content

**Key Information:**
{content_preview}...

**Summary:**
The NCSU website provides comprehensive information. Content selected based on relevance scoring.

*Note: Mock response. Configure real LLM provider (OpenAI/Anthropic) for AI-generated answers.*"""


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider"""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.7, max_tokens: int = 8000):
        super().__init__("openai", model, temperature, max_tokens)
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("❌ OPENAI_API_KEY not found. Set via: export OPENAI_API_KEY='your-key'")
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider"""

    def __init__(self, model: str = "claude-3-sonnet-20240229", temperature: float = 0.7, max_tokens: int = 1000):
        super().__init__("anthropic", model, temperature, max_tokens)
        try:
            import anthropic
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("❌ ANTHROPIC_API_KEY not found. Set via: export ANTHROPIC_API_KEY='your-key'")
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"


class NCSUAdvancedResearcher:
    """Advanced NCSU research assistant with performance optimizations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger("ncsu_advanced_researcher")
        self.cache = ContentCache() if config.get('enable_caching', True) else None
        self.progress_callback = config.get('progress_callback', None)

        self.grading_provider = self._setup_grading_provider()
        self.answer_provider = self._setup_answer_provider()

        scraper_config = ScrapingConfig(
            selenium_enabled=config.get('selenium_enabled', False),
            enhanced_extraction=config.get('enhanced_extraction', True),
            timeout=config.get('timeout', 30)
        )
        self.scraper = NCSUScraper(config=scraper_config)
        self.aggregator = ContentAggregator()

        self.output_dir = Path(config.get('output_dir', 'results'))
        self.output_dir.mkdir(exist_ok=True)

        print(f"🎯 NCSU Advanced Researcher (OPTIMIZED)")
        print(f"🤖 Grading: {self.grading_provider.provider_name} ({self.grading_provider.model})")
        print(f"🤖 Answer: {self.answer_provider.provider_name} ({self.answer_provider.model})")
        print(f"🔍 Top-K: {config.get('top_k', 10)}, Max Pages: {config.get('max_pages', 5)}")
        print(f"📊 Threshold: {config.get('relevance_threshold', 0.6)}")
        print(f"⚡ Parallel: Extract={config.get('parallel_extraction', True)} ({config.get('extraction_workers', 5)}w), "
              f"Grade={config.get('parallel_grading', True)} ({config.get('grading_workers', 5)}w)")
        print(f"💾 Cache: {config.get('enable_caching', True)}, 🛑 Early Stop: {config.get('enable_early_stopping', True)}")

    def _setup_grading_provider(self) -> LLMProvider:
        provider = self.config.get('grading_provider', self.config.get('llm_provider', 'mock')).lower()
        if provider == 'openai':
            return OpenAIProvider(
                model=self.config.get('grading_model', 'gpt-4o-mini'),
                temperature=self.config.get('grading_temperature', 0.3),
                max_tokens=self.config.get('grading_max_tokens', 10)
            )
        elif provider == 'anthropic':
            return AnthropicProvider(
                model=self.config.get('grading_model', 'claude-3-haiku-20240307'),
                temperature=self.config.get('grading_temperature', 0.3),
                max_tokens=self.config.get('grading_max_tokens', 10)
            )
        return MockLLMProvider()

    def _setup_answer_provider(self) -> LLMProvider:
        provider = self.config.get('llm_provider', 'mock').lower()
        if provider == 'openai':
            return OpenAIProvider(
                model=self.config.get('llm_model', 'gpt-4o'),
                temperature=self.config.get('llm_temperature', 0.7),
                max_tokens=self.config.get('llm_max_tokens', 4000)
            )
        elif provider == 'anthropic':
            return AnthropicProvider(
                model=self.config.get('llm_model', 'claude-3-sonnet-20240229'),
                temperature=self.config.get('llm_temperature', 0.7),
                max_tokens=self.config.get('llm_max_tokens', 4000)
            )
        return MockLLMProvider()

    def grade_content_relevance(self, content: str, query: str) -> float:
        """Grade content relevance with caching and truncation"""
        if self.cache:
            cached = self.cache.get_grade(content, query)
            if cached is not None:
                return cached

        max_chars = self.config.get('max_grading_content_length', 2000)
        content_truncated = content[:max_chars]

        prompt = f"""Grade relevance of content to query (0.0-1.0):

QUERY: {query}

CONTENT:
{content_truncated}

Scale: 1.0=Perfect, 0.8-0.9=Highly relevant, 0.6-0.7=Moderate, 0.4-0.5=Somewhat, 0.2-0.3=Minimal, 0.0-0.1=Irrelevant

Return ONLY a number (e.g., 0.85):"""

        try:
            response = self.grading_provider.generate_response(prompt)
            import re
            match = re.search(r'(\d+\.?\d*)', response)
            if match:
                score = max(0.0, min(1.0, float(match.group(1))))
                if self.cache:
                    self.cache.set_grade(content, query, score)
                return score
            return 0.5
        except Exception as e:
            self.logger.warning(f"Grading error: {e}")
            return 0.5

    def _extract_single_page(self, result) -> Optional[Dict]:
        """Extract content from single page (parallel)"""
        try:
            if self.cache:
                cached = self.cache.get_page(str(result.url))
                if cached:
                    return {
                        'title': result.title, 'url': str(result.url), 'content': cached,
                        'word_count': len(cached.split()), 'extraction_success': True, 'cached': True
                    }
            pages = self.scraper.scrape_pages([result])
            if pages and pages[0].extraction_success:
                page = pages[0]
                if self.cache:
                    self.cache.set_page(str(page.url), page.content)
                return {
                    'title': page.title, 'url': str(page.url), 'content': page.content,
                    'word_count': len(page.content.split()), 'extraction_success': True, 'cached': False
                }
        except Exception as e:
            self.logger.warning(f"Extract error {result.url}: {e}")
        return None

    def _grade_single_page(self, page: Dict, query: str) -> Dict:
        """Grade single page (parallel)"""
        try:
            score = self.grade_content_relevance(page['content'], query)
            return {**page, 'relevance_score': score}
        except Exception as e:
            self.logger.warning(f"Grade error {page['url']}: {e}")
            return {**page, 'relevance_score': 0.5}

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    
    def _truncate_sources_to_fit(self, sources: List[Dict], query: str, max_tokens: int = 120000) -> List[Dict]:
        prompt_overhead = 500
        response_reserve = self.config.get('llm_max_tokens', 4000)
        available_tokens = max_tokens - prompt_overhead - response_reserve - self._estimate_tokens(query)

        truncated_sources = []
        used_tokens = 0

        for source in sources:
            content_tokens = self._estimate_tokens(source['content'])
            if used_tokens + content_tokens <= available_tokens:
                truncated_sources.append(source)
                used_tokens += content_tokens
            else:
                remaining_tokens = available_tokens - used_tokens
                if remaining_tokens > 500:
                    remaining_chars = remaining_tokens * 4
                    truncated_content = source['content'][:remaining_chars]
                    truncated_sources.append({
                        **source,
                        'content': truncated_content,
                        'truncated': True,
                        'original_word_count': source['word_count'],
                        'word_count': len(truncated_content.split())
                    })
                break

        return truncated_sources

    def build_prompt(self, query: str, sources: List[Dict]) -> str:
        """
        Build the answer prompt from filtered pages.
        Called by both generate_answer() and the UI streaming helpers.
        Extracted so the prompt is always identical regardless of path.
        """
        max_context_tokens = self.config.get('max_context_tokens', 120000)
        sources_to_use = self._truncate_sources_to_fit(sources, query, max_context_tokens)

        sources_text = "\n".join([
            f"=== SOURCE {i+1}: {s['title']} (Relevance: {s.get('relevance_score', 'N/A')}) ===\n"
            f"URL: {s['url']}\nContent: {s['content']}\n"
            for i, s in enumerate(sources_to_use)
        ])

        return f"""You are an expert research assistant. Based on the NCSU website content provided below, answer the user's question comprehensively and accurately.

USER QUESTION: {query}

NCSU WEBSITE CONTENT:
{sources_text}

INSTRUCTIONS:
- Analyze all the provided content thoroughly
- Extract and synthesize relevant information to answer the question
- Provide a comprehensive, well-structured response
- Use specific details and facts from the content
- If the content contains the answer, provide it in full detail
- If the content is incomplete, mention what information is available
- Be accurate and factual - only use information from the provided content
- Organize your response logically with clear paragraphs
- Include specific details, names, dates, and facts when available

COMPREHENSIVE ANSWER:"""

    def generate_answer(self, content: str, query: str, sources: List[Dict]) -> str:
        """Non-streaming answer generation (used by main() terminal mode)"""
        prompt = self.build_prompt(query, sources)
        estimated_tokens = self._estimate_tokens(prompt)
        print(f"📊 Estimated prompt tokens: {estimated_tokens:,}")
        return self.answer_provider.generate_response(prompt)

    def research(self, query: str) -> Dict[str, Any]:
        """
        Conduct optimized research: search → extract → grade → filter.

        *** Step 5 (generate_answer) is intentionally NOT called here. ***
        The UI calls st.write_stream() immediately after this returns,
        so streaming starts the moment extraction finishes — no blocking wait.
        For terminal use, main() calls generate_answer() separately.
        """
        print(f"\n🔍 OPTIMIZED RESEARCH")
        print("=" * 70)
        print(f"📋 Query: '{query}'")

        results = {
            'query': query, 'timestamp': datetime.now().isoformat(), 'config': self.config,
            'search_results': [], 'extracted_pages': [], 'graded_pages': [], 'filtered_pages': [],
            'final_answer': '', 'sources': [],
            'performance_stats': {'cached_pages': 0, 'cached_grades': 0, 'early_stopped': False}
        }

        # Step 1: Search (silently append "at ncsu" if not already present)
        search_query = query if "ncsu" in query.lower() or "nc state" in query.lower() else query + " at ncsu"
        print(f"\n📋 STEP 1: Searching NCSU...")
        search_results = self.scraper.search(search_query, max_results=self.config.get('top_k', 10))
        results['search_results'] = [{'title': r.title, 'url': str(r.url), 'snippet': r.snippet} for r in search_results]
        print(f"✅ Found {len(search_results)} results")

        if not search_results:
            print("❌ No results")
            return results

        # Step 2: Extract (Parallel)
        max_pages = self.config.get('max_pages', 5)
        pages_to_extract = search_results[:max_pages]
        print(f"\n📋 STEP 2: Extracting {len(pages_to_extract)} pages (PARALLEL)...")

        extracted_pages = []
        if self.config.get('parallel_extraction', True):
            workers = self.config.get('extraction_workers', 5)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(self._extract_single_page, r): r for r in pages_to_extract}
                for future in as_completed(futures):
                    page = future.result()
                    if page:
                        extracted_pages.append(page)
                        cached_str = " (cached)" if page.get('cached') else ""
                        print(f"  ✅ {page['title'][:50]}{cached_str} ({page['word_count']:,} words)")
                        if page.get('cached'):
                            results['performance_stats']['cached_pages'] += 1
                        # Call progress callback if provided
                        if self.progress_callback:
                            self.progress_callback('extraction', page)
        else:
            for r in pages_to_extract:
                page = self._extract_single_page(r)
                if page:
                    extracted_pages.append(page)

        results['extracted_pages'] = extracted_pages
        print(f"✅ Extracted {len(extracted_pages)} pages ({sum(p['word_count'] for p in extracted_pages):,} words)")

        if not extracted_pages:
            print("❌ No content extracted")
            return results

        # Step 3: Grade (Parallel)
        if self.config.get('enable_grading', True):
            print(f"\n📋 STEP 3: Grading (PARALLEL)...")
            graded_pages = []
            if self.config.get('parallel_grading', True):
                workers = self.config.get('grading_workers', 5)
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(self._grade_single_page, p, query): p for p in extracted_pages}
                    for i, future in enumerate(as_completed(futures), 1):
                        graded = future.result()
                        graded_pages.append(graded)
                        print(f"  [{i}/{len(extracted_pages)}] {graded['title'][:40]}: {graded['relevance_score']:.3f}")
            else:
                for i, page in enumerate(extracted_pages, 1):
                    graded = self._grade_single_page(page, query)
                    graded_pages.append(graded)
                    print(f"  [{i}/{len(extracted_pages)}] {graded['title'][:40]}: {graded['relevance_score']:.3f}")
            results['graded_pages'] = graded_pages
            print(f"✅ Graded {len(graded_pages)} pages")
        else:
            graded_pages = [{**p, 'relevance_score': 1.0} for p in extracted_pages]
            results['graded_pages'] = graded_pages

        # Step 4: Filter + Early Stop
        print(f"\n📋 STEP 4: Filtering (threshold: {self.config.get('relevance_threshold', 0.6)})...")
        threshold = self.config.get('relevance_threshold', 0.6)
        filtered_pages = sorted(
            [p for p in graded_pages if p['relevance_score'] >= threshold],
            key=lambda x: x['relevance_score'], reverse=True
        )

        if self.config.get('enable_early_stopping', True):
            early_threshold = self.config.get('early_stop_threshold', 0.85)
            early_min = self.config.get('early_stop_min_pages', 3)
            high_quality = [p for p in filtered_pages if p['relevance_score'] >= early_threshold]
            if len(high_quality) >= early_min:
                print(f"🛑 Early stop: {len(high_quality)} pages ≥ {early_threshold}")
                filtered_pages = high_quality[:early_min]
                results['performance_stats']['early_stopped'] = True

        if not filtered_pages:
            print(f"⚠️ No pages meet threshold, using top page")
            filtered_pages = [max(graded_pages, key=lambda x: x['relevance_score'])]

        results['filtered_pages'] = filtered_pages
        print(f"✅ {len(filtered_pages)} pages filtered ({sum(p['word_count'] for p in filtered_pages):,} words)")

        results['sources'] = [
            {'title': p['title'], 'url': p['url'], 'relevance_score': p['relevance_score'], 'word_count': p['word_count']}
            for p in filtered_pages
        ]

        # ── Step 5 intentionally removed ──────────────────────────────────────
        # The UI streams the answer directly via st.write_stream() right after
        # this function returns. This eliminates the 15-20s blocking wait.
        # Terminal use: main() calls generate_answer() below instead.
        print(f"\n✅ Research complete — ready for answer generation")

        return results

    def save_results(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_safe = "".join(c for c in results['query'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        query_short = query_safe[:50].replace(' ', '_')

        files = {}

        answer_file = self.output_dir / f"answer_{query_short}_{timestamp}.txt"
        with open(answer_file, 'w', encoding='utf-8') as f:
            f.write(f"Query: {results['query']}\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Grading: {self.grading_provider.provider_name} ({self.grading_provider.model})\n")
            f.write(f"Answer: {self.answer_provider.provider_name} ({self.answer_provider.model})\n\n")
            f.write("=" * 50 + "\nANSWER:\n" + "=" * 50 + "\n")
            f.write(results['final_answer'])
            f.write("\n\n" + "=" * 50 + "\nSOURCES:\n" + "=" * 50 + "\n")
            for i, s in enumerate(results['sources'], 1):
                f.write(f"[{i}] {s['title']} (Relevance: {s['relevance_score']:.3f})\n")
                f.write(f"    {s['url']} ({s['word_count']:,} words)\n\n")
            f.write("=" * 50 + "\nPERFORMANCE:\n" + "=" * 50 + "\n")
            f.write(f"Cached pages: {results['performance_stats']['cached_pages']}\n")
            f.write(f"Early stopped: {results['performance_stats']['early_stopped']}\n")
        files['answer'] = str(answer_file)

        # Create a serializable copy of config (remove function objects like progress_callback)
        config_for_json = {k: v for k, v in results['config'].items() if not callable(v)}
        results_for_json = {**results, 'config': config_for_json}
        
        data_file = self.output_dir / f"data_{query_short}_{timestamp}.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(results_for_json, f, indent=2, ensure_ascii=False)
        files['data'] = str(data_file)

        # Also filter out callables from YAML config
        config_for_yaml = {k: v for k, v in results['config'].items() if not callable(v)}
        config_file = self.output_dir / f"config_{query_short}_{timestamp}.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_for_yaml, f, default_flow_style=False)
        files['config'] = str(config_file)

        return files

    def display_results(self, results: Dict[str, Any]):
        print(f"\n{'=' * 70}")
        print(f"RESULTS")
        print(f"{'=' * 70}")
        print(f"\n🔍 QUERY: {results['query']}")
        print(f"\n🤖 ANSWER:\n{results['final_answer']}")
        print(f"\n📚 SOURCES:")
        for i, s in enumerate(results['sources'], 1):
            print(f"[{i}] {s['title']} (Relevance: {s['relevance_score']:.3f})")
            print(f"    {s['url']} ({s['word_count']:,} words)")
        print(f"\n⚡ PERFORMANCE:")
        print(f"💾 Cached: {results['performance_stats']['cached_pages']} pages")
        print(f"🛑 Early stopped: {results['performance_stats']['early_stopped']}")


def main():
    """Main function — terminal mode calls generate_answer() separately"""

    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded .env")
    except:
        pass

    config = {
        'query': 'What are the requirements for the Computer Science major at NC State University?',

        # Answer LLM
        'llm_provider': ANSWER_LLM_PROVIDER,
        'llm_model': ANSWER_LLM_MODEL,
        'llm_temperature': ANSWER_LLM_TEMPERATURE,
        'llm_max_tokens': ANSWER_LLM_MAX_TOKENS,
        'max_context_tokens': MAX_CONTEXT_TOKENS,   # 120000 — sends full content to model

        # Grading LLM
        'grading_provider': GRADING_LLM_PROVIDER,
        'grading_model': GRADING_LLM_MODEL,
        'grading_temperature': GRADING_LLM_TEMPERATURE,
        'grading_max_tokens': GRADING_LLM_MAX_TOKENS,
        'max_grading_content_length': MAX_GRADING_CONTENT_LENGTH,

        # Search & extraction
        'top_k': 10,
        'max_pages': 10,
        'relevance_threshold': 0.3,         # lowered: don't discard borderline pages

        # Grading ON so pages are actually filtered meaningfully
        'enable_grading': True,
        'parallel_extraction': True,
        'extraction_workers': 5,
        'parallel_grading': True,
        'grading_workers': 5,

        # Caching ON, early stopping OFF so all graded pages reach the LLM
        'enable_caching': True,
        'enable_early_stopping': False,

        # Scraping
        'selenium_enabled': True,
        'enhanced_extraction': True,
        'timeout': 30,
        'output_dir': 'results',
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
    }

    if config.get('openai_api_key'):
        os.environ['OPENAI_API_KEY'] = config['openai_api_key']
    if config.get('anthropic_api_key'):
        os.environ['ANTHROPIC_API_KEY'] = config['anthropic_api_key']

    print("🎯 NCSU Advanced Research Assistant (OPTIMIZED)")
    print("=" * 50)
    print(f"📋 Query: {config['query']}")
    print(f"🤖 Grading: {config['grading_provider']} ({config['grading_model']})")
    print(f"🤖 Answer: {config['llm_provider']} ({config['llm_model']})")
    print("=" * 50)

    try:
        researcher = NCSUAdvancedResearcher(config)

        # research() does steps 1-4 only
        results = researcher.research(config['query'])

        # Step 5: generate answer in terminal (non-streaming)
        print(f"\n📋 STEP 5: Generating answer...")
        final_answer = researcher.generate_answer('', config['query'], results['filtered_pages'])
        results['final_answer'] = final_answer

        researcher.display_results(results)

        print(f"\n📋 STEP 6: Saving results...")
        saved_files = researcher.save_results(results)
        for file_type, path in saved_files.items():
            print(f"💾 {file_type.title()}: {path}")

        print(f"\n🎉 COMPLETE!")
        print(f"✅ Found {len(results['search_results'])} results")
        print(f"✅ Extracted {len(results['extracted_pages'])} pages ({sum(p['word_count'] for p in results['extracted_pages']):,} words)")
        print(f"✅ Graded {len(results['graded_pages'])} pages")
        print(f"✅ Filtered {len(results['filtered_pages'])} pages ({sum(p['word_count'] for p in results['filtered_pages']):,} words)")
        print(f"✅ Generated answer ({len(results['final_answer']):,} chars)")
        print(f"💾 Cached: {results['performance_stats']['cached_pages']} pages")
        print(f"🛑 Early stopped: {results['performance_stats']['early_stopped']}")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()




