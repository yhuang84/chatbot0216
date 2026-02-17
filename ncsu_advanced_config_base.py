#!/usr/bin/env python3
"""
NCSU Advanced Research Assistant - COMPLETE VERSION WITH UI SUPPORT
===================================================================

This version combines:
- OLD VERSION: Sends ALL filtered sources with FULL content (no truncation)
- NEW VERSION: UI streaming compatibility and parallel processing

Key Features:
- ✅ Sends ALL filtered sources to LLM (no skipping)
- ✅ Sends FULL content from each source (no truncation)
- ✅ Compatible with user_interface.py streaming
- ✅ Parallel extraction and grading for speed
- ✅ Caching for efficiency
- ✅ Separate grading/answer LLM providers

Usage:
    1. Edit the config dictionary in main()
    2. Run: python ncsu_advanced_config_base.py
    OR
    3. Use with user_interface.py for web UI
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
ANSWER_LLM_MODEL = 'gpt-4o'
ANSWER_LLM_TEMPERATURE = 0.1
ANSWER_LLM_MAX_TOKENS = 8000
MAX_CONTEXT_TOKENS = 120000  # Not used for truncation, kept for compatibility

GRADING_LLM_PROVIDER = 'openai'
GRADING_LLM_MODEL = 'gpt-4o-mini'
GRADING_LLM_TEMPERATURE = 0.0
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
        else:
            return f"""Based on the NCSU website content, here's a comprehensive answer.

**Analysis Complete:**
This response synthesizes information from {len(prompt.split('SOURCE'))-1} sources from NC State's official website.

**Mock Answer:**
The provided content contains detailed information addressing your query. This is a mock response demonstrating the system's capability to generate comprehensive answers.

*Note: Configure OpenAI or Anthropic for AI-generated answers.*"""


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider"""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.7, max_tokens: int = 8000):
        super().__init__("openai", model, temperature, max_tokens)
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found. Set via: export OPENAI_API_KEY='your-key'")
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

    def __init__(self, model: str = "claude-3-sonnet-20240229", temperature: float = 0.7, max_tokens: int = 8000):
        super().__init__("anthropic", model, temperature, max_tokens)
        try:
            import anthropic
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found. Set via: export ANTHROPIC_API_KEY='your-key'")
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
    """Advanced NCSU research assistant - COMPLETE VERSION"""

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

        print(f"🎯 NCSU Advanced Researcher (COMPLETE VERSION)")
        print(f"🤖 Grading: {self.grading_provider.provider_name} ({self.grading_provider.model})")
        print(f"🤖 Answer: {self.answer_provider.provider_name} ({self.answer_provider.model})")
        print(f"🔍 Top-K: {config.get('top_k', 10)}, Max Pages: {config.get('max_pages', 5)}")
        print(f"📊 Threshold: {config.get('relevance_threshold', 0.6)}")
        print(f"⚡ Parallel: Extract={config.get('parallel_extraction', True)}, Grade={config.get('parallel_grading', True)}")
        print(f"💾 Cache: {config.get('enable_caching', True)}")
        print(f"📝 Content Policy: SEND ALL FILTERED SOURCES WITH FULL CONTENT (no truncation)")

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
                max_tokens=self.config.get('llm_max_tokens', 8000)
            )
        elif provider == 'anthropic':
            return AnthropicProvider(
                model=self.config.get('llm_model', 'claude-3-sonnet-20240229'),
                temperature=self.config.get('llm_temperature', 0.7),
                max_tokens=self.config.get('llm_max_tokens', 8000)
            )
        return MockLLMProvider()

    def grade_content_relevance(self, content: str, query: str) -> float:
        """Grade content relevance using LLM"""
        if self.cache:
            cached = self.cache.get_grade(content, query)
            if cached is not None:
                return cached
        
        # ✅ Use full content for grading - no truncation
        content_to_grade = content
        
        prompt = f"""You are an expert content grader. Grade how relevant this content is to answering the user's query.

USER QUERY: {query}

CONTENT TO GRADE:
{content_to_grade}

GRADING INSTRUCTIONS:
- Analyze the entire content thoroughly
- Consider how well the content answers or relates to the query
- Ignore navigation menus, headers, and boilerplate text
- Focus on the substantive information that addresses the query
- Consider information quality, accuracy, and completeness

SCORING SCALE:
- 1.0 = Perfect match - content directly and comprehensively answers the query
- 0.8-0.9 = Highly relevant - content strongly relates and provides good information
- 0.6-0.7 = Moderately relevant - content relates but may be incomplete or tangential
- 0.4-0.5 = Somewhat relevant - content has some connection but limited usefulness
- 0.2-0.3 = Minimally relevant - content barely relates to the query
- 0.0-0.1 = Irrelevant - content does not relate to the query

Return ONLY a decimal number between 0.0 and 1.0 (e.g., 0.85):"""
        
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

    def build_prompt(self, query: str, sources: List[Dict]) -> str:
        """
        Build the answer prompt from filtered pages.
        ✅ CRITICAL CHANGE: NO TRUNCATION - sends ALL sources with FULL content
        
        This is called by both:
        - generate_answer() for terminal use
        - user_interface.py for streaming
        """
        
        # ✅ Use ALL sources without any truncation
        sources_to_use = sources  # NO truncation, NO skipping
        
        print(f"\n{'='*80}")
        print(f"📊 CONTENT BEING SENT TO LLM")
        print(f"{'='*80}")
        print(f"   Sources: {len(sources_to_use)} (ALL filtered sources)")
        print(f"   Policy: FULL content from each source (NO truncation)")
        total_words = sum(s.get('word_count', 0) for s in sources_to_use)
        print(f"   Total words: {total_words:,}")
        print(f"\n   Sources included:")
        for i, s in enumerate(sources_to_use, 1):
            print(f"   [{i}] {s['title'][:70]}")
            print(f"       Score: {s.get('relevance_score', 'N/A'):.3f}, Words: {s.get('word_count', 0):,}")
        print(f"{'='*80}\n")

        sources_text = "\n".join([
            f"=== SOURCE {i+1}: {s['title']} (Relevance: {s.get('relevance_score', 'N/A')}) ===\n"
            f"URL: {s['url']}\nContent: {s['content']}\n"  # ✅ FULL content
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
        """Generate answer using LLM (for terminal/non-streaming use)"""
        prompt = self.build_prompt(query, sources)
        return self.answer_provider.generate_response(prompt)

    def research(self, query: str) -> Dict[str, Any]:
        """
        Conduct complete research: search → extract → grade → filter
        
        Compatible with both:
        - Terminal use (main() will call generate_answer separately)
        - UI use (user_interface.py will call build_prompt for streaming)
        """
        print(f"\n🔍 COMPLETE RESEARCH (NO TRUNCATION)")
        print("=" * 70)
        print(f"📋 Query: '{query}'")

        results = {
            'query': query, 'timestamp': datetime.now().isoformat(), 'config': self.config,
            'search_results': [], 'extracted_pages': [], 'graded_pages': [], 'filtered_pages': [],
            'final_answer': '', 'sources': [],
            'performance_stats': {'cached_pages': 0, 'cached_grades': 0}
        }

        # Step 1: Search
        search_query = query
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
        print(f"\n📋 STEP 2: Extracting {len(pages_to_extract)} pages...")

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
            print(f"\n📋 STEP 3: Grading...")
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

        # Step 4: Filter
        print(f"\n📋 STEP 4: Filtering (threshold: {self.config.get('relevance_threshold', 0.6)})...")
        threshold = self.config.get('relevance_threshold', 0.6)
        filtered_pages = sorted(
            [p for p in graded_pages if p['relevance_score'] >= threshold],
            key=lambda x: x['relevance_score'], reverse=True
        )

        if not filtered_pages:
            print(f"⚠️ No pages meet threshold, using top page")
            filtered_pages = [max(graded_pages, key=lambda x: x['relevance_score'])]

        results['filtered_pages'] = filtered_pages
        print(f"✅ {len(filtered_pages)} pages filtered ({sum(p['word_count'] for p in filtered_pages):,} words)")
        print(f"\n📊 ALL {len(filtered_pages)} FILTERED PAGES WILL BE SENT TO LLM WITH FULL CONTENT")

        results['sources'] = [
            {'title': p['title'], 'url': p['url'], 'relevance_score': p['relevance_score'], 'word_count': p['word_count']}
            for p in filtered_pages
        ]

        # ✅ Step 5 NOT called here - UI will call build_prompt() for streaming
        # Terminal use: main() calls generate_answer() separately
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
            f.write(f"Answer: {self.answer_provider.provider_name} ({self.answer_provider.model})\n")
            f.write(f"Content Policy: ALL filtered sources, FULL content (no truncation)\n\n")
            f.write("=" * 50 + "\nANSWER:\n" + "=" * 50 + "\n")
            f.write(results['final_answer'])
            f.write("\n\n" + "=" * 50 + "\nSOURCES:\n" + "=" * 50 + "\n")
            for i, s in enumerate(results['sources'], 1):
                f.write(f"[{i}] {s['title']} (Relevance: {s['relevance_score']:.3f})\n")
                f.write(f"    {s['url']} ({s['word_count']:,} words)\n\n")
            f.write("=" * 50 + "\nPERFORMANCE:\n" + "=" * 50 + "\n")
            f.write(f"Cached pages: {results['performance_stats']['cached_pages']}\n")
        files['answer'] = str(answer_file)

        # Create serializable copy
        config_for_json = {k: v for k, v in results['config'].items() if not callable(v)}
        results_for_json = {**results, 'config': config_for_json}
        
        data_file = self.output_dir / f"data_{query_short}_{timestamp}.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(results_for_json, f, indent=2, ensure_ascii=False)
        files['data'] = str(data_file)

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


def main():
    """Main function for terminal use"""

    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded .env")
    except:
        pass

    config = {
        'query': 'What are the computer science graduate programs at NC State University?',
        
        # LLM Configuration
        'llm_provider': ANSWER_LLM_PROVIDER,
        'llm_model': ANSWER_LLM_MODEL,
        'llm_temperature': ANSWER_LLM_TEMPERATURE,
        'llm_max_tokens': ANSWER_LLM_MAX_TOKENS,
        
        # Grading Configuration
        'grading_provider': GRADING_LLM_PROVIDER,
        'grading_model': GRADING_LLM_MODEL,
        'grading_temperature': GRADING_LLM_TEMPERATURE,
        'grading_max_tokens': GRADING_LLM_MAX_TOKENS,
        
        # Search & Content
        'top_k': 20,
        'max_pages': 20,
        'relevance_threshold': 0.1,
        'enable_grading': True,
        
        # Performance
        'parallel_extraction': True,
        'extraction_workers': 5,
        'parallel_grading': True,
        'grading_workers': 5,
        'enable_caching': True,
        
        # Scraper
        'selenium_enabled': True,
        'enhanced_extraction': True,
        'timeout': 30,
        
        # Output
        'output_dir': 'results',
        'log_level': 'INFO'
    }

    try:
        researcher = NCSUAdvancedResearcher(config)
        
        print(f"\n{'='*70}")
        print(f"RUNNING COMPLETE RESEARCH")
        print(f"{'='*70}")
        
        results = researcher.research(config['query'])
        
        # ✅ Generate answer for terminal use
        print(f"\n📋 STEP 5: Generating answer...")
        final_answer = researcher.generate_answer('', config['query'], results['filtered_pages'])
        results['final_answer'] = final_answer
        
        saved_files = researcher.save_results(results)
        researcher.display_results(results)
        
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"✅ Found {len(results['search_results'])} results")
        print(f"✅ Extracted {len(results['extracted_pages'])} pages ({sum(p['word_count'] for p in results['extracted_pages']):,} words)")
        print(f"✅ Graded {len(results['graded_pages'])} pages")
        print(f"✅ Filtered {len(results['filtered_pages'])} pages ({sum(p['word_count'] for p in results['filtered_pages']):,} words)")
        print(f"✅ Generated answer ({len(results['final_answer']):,} chars)")
        print(f"💾 Cached: {results['performance_stats']['cached_pages']} pages")
        print(f"\n📄 Files saved:")
        print(f"   Answer: {saved_files['answer']}")
        print(f"   Data: {saved_files['data']}")
        print(f"   Config: {saved_files['config']}")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
