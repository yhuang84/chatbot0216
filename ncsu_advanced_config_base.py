#!/usr/bin/env python3
"""
NCSU Advanced Research Assistant - OPTIMIZED VERSION WITH STREAMING
====================================================================

Key change: stream_response() is now a GENERATOR that yields text chunks.
This lets both the terminal (print) and Streamlit (st.write_stream) consume it.

Usage:
    1. Edit the config dictionary in main()
    2. Run: python ncsu_advanced_config_base_optimized.py
"""

import json
import os
import sys
import yaml
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from scraper.ncsu_scraper import NCSUScraper
from scraper.content_aggregator import ContentAggregator
from scraper.models import ScrapingConfig
from utils.logger import setup_logger


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

    def stream_response(self, prompt: str) -> Generator[str, None, None]:
        """
        GENERATOR that yields text chunks one at a time.
        Works with both:
          - Terminal:   for chunk in provider.stream_response(p): print(chunk, end="", flush=True)
          - Streamlit:  st.write_stream(provider.stream_response(p))
        """
        # Default: yield the whole response as one chunk
        yield self.generate_response(prompt)


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

**Key Information:**
{content_preview}...

**Summary:** The NCSU website provides comprehensive information.

*Note: Mock response. Configure a real LLM provider for AI-generated answers.*"""

    def stream_response(self, prompt: str) -> Generator[str, None, None]:
        """Yield word by word for mock demo effect"""
        import time
        response = self.generate_response(prompt)
        for word in response.split(' '):
            yield word + ' '
            time.sleep(0.02)


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider — stream_response() is a generator"""

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

    def stream_response(self, prompt: str) -> Generator[str, None, None]:
        """
        Generator that yields each token chunk from OpenAI streaming API.
        Compatible with st.write_stream() and manual iteration.
        """
        try:
            with self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            ) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta  # ← yield, not print
        except Exception as e:
            yield f"\n\nError during streaming: {str(e)}"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider — stream_response() is a generator"""

    def __init__(self, model: str = "claude-3-sonnet-20240229", temperature: float = 0.7, max_tokens: int = 4000):
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

    def stream_response(self, prompt: str) -> Generator[str, None, None]:
        """
        Generator that yields each text chunk from Anthropic streaming API.
        Compatible with st.write_stream() and manual iteration.
        """
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    yield text  # ← yield, not print
        except Exception as e:
            yield f"\n\nError during streaming: {str(e)}"


class NCSUAdvancedResearcher:
    """Advanced NCSU research assistant with generator-based streaming"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger("ncsu_advanced_researcher")
        self.cache = ContentCache() if config.get('enable_caching', True) else None

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
        """Grade content relevance — always non-streaming"""
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

    def _build_answer_prompt(self, query: str, sources: List[Dict]) -> str:
        sources_text = "\n".join([
            f"=== SOURCE {i+1}: {s['title']} (Relevance: {s.get('relevance_score', 'N/A')}) ===\n"
            f"URL: {s['url']}\nContent: {s['content']}\n"
            for i, s in enumerate(sources)
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
- Be accurate and factual - only use information from the provided content
- Organize your response logically with clear paragraphs

COMPREHENSIVE ANSWER:"""

    def get_answer_stream(self, query: str, sources: List[Dict]) -> Generator[str, None, None]:
        """
        PUBLIC method: returns a generator of text chunks.

        Use in Streamlit:
            full_text = st.write_stream(researcher.get_answer_stream(query, sources))

        Use in terminal:
            full_text = ""
            for chunk in researcher.get_answer_stream(query, sources):
                print(chunk, end="", flush=True)
                full_text += chunk
        """
        max_context_tokens = self.config.get('max_context_tokens', 120000)
        sources_to_use = self._truncate_sources_to_fit(sources, query, max_context_tokens)
        prompt = self._build_answer_prompt(query, sources_to_use)
        yield from self.answer_provider.stream_response(prompt)

    def research(self, query: str) -> Dict[str, Any]:
        """
        Conducts all steps EXCEPT answer generation.
        Returns results dict with filtered_pages ready for streaming.
        Call get_answer_stream() separately to stream the answer.
        """
        results = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'search_results': [],
            'extracted_pages': [],
            'graded_pages': [],
            'filtered_pages': [],
            'final_answer': '',
            'sources': [],
            'performance_stats': {'cached_pages': 0, 'cached_grades': 0, 'early_stopped': False}
        }

        # Step 1: Search
        search_results = self.scraper.search(query, max_results=self.config.get('top_k', 10))
        results['search_results'] = [{'title': r.title, 'url': str(r.url), 'snippet': r.snippet} for r in search_results]

        if not search_results:
            return results

        # Step 2: Extract (Parallel)
        max_pages = self.config.get('max_pages', 5)
        pages_to_extract = search_results[:max_pages]
        extracted_pages = []

        if self.config.get('parallel_extraction', True):
            workers = self.config.get('extraction_workers', 5)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(self._extract_single_page, r): r for r in pages_to_extract}
                for future in as_completed(futures):
                    page = future.result()
                    if page:
                        extracted_pages.append(page)
                        if page.get('cached'):
                            results['performance_stats']['cached_pages'] += 1
        else:
            for r in pages_to_extract:
                page = self._extract_single_page(r)
                if page:
                    extracted_pages.append(page)

        results['extracted_pages'] = extracted_pages

        if not extracted_pages:
            return results

        # Step 3: Grade (Parallel)
        if self.config.get('enable_grading', True):
            graded_pages = []
            if self.config.get('parallel_grading', True):
                workers = self.config.get('grading_workers', 5)
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(self._grade_single_page, p, query): p for p in extracted_pages}
                    for future in as_completed(futures):
                        graded_pages.append(future.result())
            else:
                for page in extracted_pages:
                    graded_pages.append(self._grade_single_page(page, query))
            results['graded_pages'] = graded_pages
        else:
            graded_pages = [{**p, 'relevance_score': 1.0} for p in extracted_pages]
            results['graded_pages'] = graded_pages

        # Step 4: Filter + Early Stop
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
                filtered_pages = high_quality[:early_min]
                results['performance_stats']['early_stopped'] = True

        if not filtered_pages:
            filtered_pages = [max(graded_pages, key=lambda x: x['relevance_score'])]

        results['filtered_pages'] = filtered_pages
        results['sources'] = [
            {'title': p['title'], 'url': p['url'],
             'relevance_score': p['relevance_score'], 'word_count': p['word_count']}
            for p in filtered_pages
        ]

        return results

    def save_results(self, results: Dict[str, Any]) -> Dict[str, str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_safe = "".join(c for c in results['query'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        query_short = query_safe[:50].replace(' ', '_')
        files = {}

        answer_file = self.output_dir / f"answer_{query_short}_{timestamp}.txt"
        with open(answer_file, 'w', encoding='utf-8') as f:
            f.write(f"Query: {results['query']}\n")
            f.write(f"Timestamp: {results['timestamp']}\n\n")
            f.write("=" * 50 + "\nANSWER:\n" + "=" * 50 + "\n")
            f.write(results.get('final_answer', ''))
            f.write("\n\n" + "=" * 50 + "\nSOURCES:\n" + "=" * 50 + "\n")
            for i, s in enumerate(results['sources'], 1):
                f.write(f"[{i}] {s['title']} (Relevance: {s['relevance_score']:.3f})\n")
                f.write(f"    {s['url']} ({s['word_count']:,} words)\n\n")
        files['answer'] = str(answer_file)

        data_file = self.output_dir / f"data_{query_short}_{timestamp}.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        files['data'] = str(data_file)

        config_file = self.output_dir / f"config_{query_short}_{timestamp}.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(results['config'], f, default_flow_style=False)
        files['config'] = str(config_file)

        return files


def main():
    """Terminal entry point — streams answer to stdout"""

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    config = {
        'query': 'What are the requirements for the Computer Science major at NC State University?',
        'llm_provider': 'openai',
        'llm_model': 'gpt-4o',
        'llm_temperature': 0.3,
        'llm_max_tokens': 4000,
        'max_context_tokens': 120000,
        'grading_provider': 'openai',
        'grading_model': 'gpt-4o-mini',
        'grading_temperature': 0.3,
        'grading_max_tokens': 10,
        'max_grading_content_length': 2000,
        'top_k': 10,
        'max_pages': 5,
        'relevance_threshold': 0.6,
        'enable_grading': True,
        'parallel_extraction': True,
        'extraction_workers': 5,
        'parallel_grading': True,
        'grading_workers': 5,
        'enable_caching': True,
        'enable_early_stopping': True,
        'early_stop_threshold': 0.85,
        'early_stop_min_pages': 3,
        'selenium_enabled': True,
        'enhanced_extraction': True,
        'timeout': 30,
        'output_dir': 'results',
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
    }

    if config.get('openai_api_key'):
        os.environ['OPENAI_API_KEY'] = config['openai_api_key']

    try:
        researcher = NCSUAdvancedResearcher(config)

        print(f"\n🔍 Researching: {config['query']}\n")
        results = researcher.research(config['query'])

        print(f"\n{'=' * 70}")
        print("🤖 ANSWER (streaming):")
        print(f"{'=' * 70}\n")

        # Terminal streaming — iterate the generator manually
        full_answer = ""
        for chunk in researcher.get_answer_stream(config['query'], results['filtered_pages']):
            print(chunk, end="", flush=True)
            full_answer += chunk
        print()

        results['final_answer'] = full_answer
        saved_files = researcher.save_results(results)

        print(f"\n📚 SOURCES:")
        for i, s in enumerate(results['sources'], 1):
            print(f"[{i}] {s['title']} ({s['relevance_score']:.3f}) — {s['url']}")

        print(f"\n💾 Saved: {saved_files.get('answer')}")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
