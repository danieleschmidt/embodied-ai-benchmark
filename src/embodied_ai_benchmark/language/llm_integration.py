"""Production-ready LLM integration for curriculum and task analysis."""

import json
import logging
import requests
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class LLMResponse:
    """Structured LLM response with confidence scoring."""
    text: str
    structured_data: Dict[str, Any]
    confidence: float
    processing_time: float
    model_used: str


class LLMIntegration:
    """Production-ready LLM integration with multiple providers and fallback."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LLM integration.
        
        Args:
            config: Configuration dict with API keys and settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Provider configurations
        self.providers = {
            "openai": {
                "api_key": config.get("openai_api_key"),
                "models": ["gpt-4", "gpt-3.5-turbo"],
                "endpoint": "https://api.openai.com/v1/chat/completions"
            },
            "anthropic": {
                "api_key": config.get("anthropic_api_key"), 
                "models": ["claude-3-opus", "claude-3-sonnet"],
                "endpoint": "https://api.anthropic.com/v1/messages"
            },
            "local": {
                "endpoint": config.get("local_llm_endpoint", "http://localhost:8000/generate"),
                "models": ["local-llm"]
            }
        }
        
        # Fallback priorities
        self.provider_priority = ["openai", "anthropic", "local"]
        
        # Performance tracking
        self.response_times = []
        self.success_rates = {}
        
        # Caching
        self.cache = {}
        self.cache_ttl = config.get("cache_ttl", 3600)  # 1 hour
    
    def analyze_performance(self, 
                          results: List[Dict[str, Any]], 
                          context: str = "") -> LLMResponse:
        """Analyze agent performance with LLM."""
        
        prompt = self._build_performance_analysis_prompt(results, context)
        
        response = self._query_with_fallback(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1000
        )
        
        # Parse structured insights
        structured_data = self._extract_structured_insights(response.text)
        
        return LLMResponse(
            text=response.text,
            structured_data=structured_data,
            confidence=response.confidence,
            processing_time=response.processing_time,
            model_used=response.model_used
        )
    
    def generate_task_variation(self, 
                              base_task: Dict[str, Any], 
                              difficulty_target: float) -> LLMResponse:
        """Generate task variation using LLM."""
        
        prompt = self._build_task_generation_prompt(base_task, difficulty_target)
        
        response = self._query_with_fallback(
            prompt=prompt,
            temperature=0.7,  # Higher creativity for task generation
            max_tokens=800
        )
        
        # Parse task specification
        task_spec = self._parse_task_specification(response.text)
        
        return LLMResponse(
            text=response.text,
            structured_data=task_spec,
            confidence=response.confidence,
            processing_time=response.processing_time,
            model_used=response.model_used
        )
    
    def parse_natural_language_task(self, description: str) -> LLMResponse:
        """Parse natural language task description."""
        
        prompt = self._build_task_parsing_prompt(description)
        
        response = self._query_with_fallback(
            prompt=prompt,
            temperature=0.2,  # Low temperature for precise parsing
            max_tokens=600
        )
        
        # Parse executable task format
        executable_task = self._parse_executable_task(response.text)
        
        return LLMResponse(
            text=response.text,
            structured_data=executable_task,
            confidence=response.confidence,
            processing_time=response.processing_time,
            model_used=response.model_used
        )
    
    def _query_with_fallback(self, 
                           prompt: str, 
                           temperature: float = 0.5,
                           max_tokens: int = 800) -> LLMResponse:
        """Query LLM with provider fallback."""
        
        # Check cache first
        cache_key = self._get_cache_key(prompt, temperature, max_tokens)
        cached = self._get_cached_response(cache_key)
        if cached:
            return cached
        
        last_error = None
        
        for provider in self.provider_priority:
            try:
                if not self._is_provider_available(provider):
                    continue
                
                start_time = time.time()
                response = self._query_provider(provider, prompt, temperature, max_tokens)
                processing_time = time.time() - start_time
                
                # Track performance
                self._track_performance(provider, processing_time, True)
                
                llm_response = LLMResponse(
                    text=response["text"],
                    structured_data={},
                    confidence=response.get("confidence", 0.8),
                    processing_time=processing_time,
                    model_used=f"{provider}:{response.get('model', 'unknown')}"
                )
                
                # Cache successful response
                self._cache_response(cache_key, llm_response)
                
                return llm_response
                
            except Exception as e:
                self.logger.warning(f"Provider {provider} failed: {e}")
                self._track_performance(provider, 0, False)
                last_error = e
                continue
        
        # All providers failed - use heuristic fallback
        self.logger.error(f"All LLM providers failed, using heuristic fallback. Last error: {last_error}")
        return self._heuristic_fallback(prompt)
    
    def _query_provider(self, 
                       provider: str, 
                       prompt: str, 
                       temperature: float,
                       max_tokens: int) -> Dict[str, Any]:
        """Query specific LLM provider."""
        
        if provider == "openai":
            return self._query_openai(prompt, temperature, max_tokens)
        elif provider == "anthropic":
            return self._query_anthropic(prompt, temperature, max_tokens)
        elif provider == "local":
            return self._query_local_llm(prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _query_openai(self, prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Query OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.providers['openai']['api_key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            self.providers["openai"]["endpoint"],
            headers=headers,
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        
        return {
            "text": data["choices"][0]["message"]["content"],
            "model": data["model"],
            "confidence": 0.9  # OpenAI doesn't provide confidence scores
        }
    
    def _query_anthropic(self, prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Query Anthropic API."""
        headers = {
            "x-api-key": self.providers['anthropic']['api_key'],
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = requests.post(
            self.providers["anthropic"]["endpoint"],
            headers=headers,
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        
        return {
            "text": data["content"][0]["text"],
            "model": data["model"],
            "confidence": 0.85
        }
    
    def _query_local_llm(self, prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Query local LLM endpoint."""
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            self.providers["local"]["endpoint"],
            json=payload,
            timeout=60
        )
        
        response.raise_for_status()
        data = response.json()
        
        return {
            "text": data.get("text", ""),
            "model": "local-llm",
            "confidence": data.get("confidence", 0.7)
        }
    
    def _build_performance_analysis_prompt(self, results: List[Dict[str, Any]], context: str) -> str:
        """Build performance analysis prompt."""
        
        # Aggregate statistics
        success_rates = [r.get("success", False) for r in results]
        rewards = [r.get("reward", 0) for r in results]
        step_counts = [r.get("steps", 0) for r in results]
        
        success_rate = np.mean(success_rates) if success_rates else 0
        avg_reward = np.mean(rewards) if rewards else 0
        avg_steps = np.mean(step_counts) if step_counts else 0
        
        # Extract failure patterns
        failures = [r for r in results if not r.get("success", False)]
        failure_reasons = [f.get("failure_reason", "unknown") for f in failures]
        
        prompt = f\"\"\"Analyze the following agent performance data and provide structured insights:

**Performance Summary:**
- Success Rate: {success_rate:.2%} ({sum(success_rates)}/{len(success_rates)} episodes)
- Average Reward: {avg_reward:.3f}
- Average Episode Length: {avg_steps:.1f} steps

**Context:** {context}

**Failure Analysis:**
- Total Failures: {len(failures)}
- Common Failure Reasons: {', '.join(set(failure_reasons))}

**Recent Episodes (last 5):**
{json.dumps(results[-5:], indent=2)}

Please provide analysis in the following JSON format:
{{
  "performance_assessment": "overall assessment",
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"], 
  "failure_modes": ["mode1", "mode2"],
  "recommendations": ["rec1", "rec2"],
  "difficulty_adjustment": "increase/maintain/decrease",
  "confidence": 0.85
}}

Focus on actionable insights for curriculum adaptation.\"\"\"
        
        return prompt
    
    def _build_task_generation_prompt(self, base_task: Dict[str, Any], difficulty_target: float) -> str:
        """Build task generation prompt."""
        
        prompt = f\"\"\"Generate a task variation based on the following parameters:

**Base Task:**
{json.dumps(base_task, indent=2)}

**Target Difficulty:** {difficulty_target:.2f} (0.0 = very easy, 1.0 = very hard)

Create a task variation that maintains the core objectives while adjusting difficulty. 

Provide response in JSON format:
{{
  "task_name": "descriptive name",
  "description": "natural language description",
  "objectives": ["obj1", "obj2"],
  "constraints": ["constraint1", "constraint2"],
  "success_criteria": "what constitutes success",
  "estimated_difficulty": 0.75,
  "time_limit": 300,
  "required_skills": ["skill1", "skill2"],
  "environment_setup": {{
    "objects": [...],
    "obstacles": [...],
    "lighting": "normal/dim/bright",
    "physics": "standard/challenging"
  }}
}}

Focus on creating engaging, learnable challenges.\"\"\"
        
        return prompt
    
    def _build_task_parsing_prompt(self, description: str) -> str:
        """Build task parsing prompt."""
        
        prompt = f\"\"\"Parse the following natural language task description into executable format:

**Task Description:**
{description}

Convert this into a structured task specification:

{{
  "task_type": "navigation/manipulation/coordination",
  "objectives": ["primary objective", "secondary objectives"],
  "objects": [
    {{
      "name": "object1",
      "type": "furniture/tool/target",
      "properties": {{"color": "red", "size": "large"}}
    }}
  ],
  "actions_required": ["move", "grasp", "place"],
  "success_conditions": ["condition1", "condition2"],
  "failure_conditions": ["avoid this", "don't do that"],
  "constraints": ["spatial", "temporal", "safety"],
  "estimated_complexity": 0.6,
  "required_capabilities": ["vision", "manipulation"],
  "safety_considerations": ["safety note1", "safety note2"]
}}

Be precise and comprehensive in the specification.\"\"\"
        
        return prompt
    
    def _extract_structured_insights(self, text: str) -> Dict[str, Any]:
        """Extract structured data from LLM response."""
        try:
            # Look for JSON blocks in the response
            lines = text.split('\n')
            json_lines = []
            in_json = False
            
            for line in lines:
                if '{' in line:
                    in_json = True
                if in_json:
                    json_lines.append(line)
                if '}' in line and in_json:
                    break
            
            json_text = '\n'.join(json_lines)
            return json.loads(json_text)
            
        except (json.JSONDecodeError, Exception) as e:
            self.logger.warning(f"Failed to parse structured insights: {e}")
            
            # Fallback to keyword extraction
            return self._extract_keywords(text)
    
    def _extract_keywords(self, text: str) -> Dict[str, Any]:
        """Extract keywords as fallback parsing."""
        return {
            "performance_assessment": "analysis available in text",
            "strengths": ["performance analysis available"],
            "weaknesses": ["see detailed text"],
            "failure_modes": ["text analysis required"],
            "recommendations": ["review full analysis"],
            "difficulty_adjustment": "maintain",
            "confidence": 0.6
        }
    
    def _parse_task_specification(self, text: str) -> Dict[str, Any]:
        """Parse task specification from LLM response."""
        return self._extract_structured_insights(text)
    
    def _parse_executable_task(self, text: str) -> Dict[str, Any]:
        """Parse executable task from LLM response."""
        return self._extract_structured_insights(text)
    
    def _heuristic_fallback(self, prompt: str) -> LLMResponse:
        """Heuristic fallback when all LLM providers fail."""
        
        # Simple heuristic analysis based on prompt content
        if "performance" in prompt.lower():
            analysis = {
                "performance_assessment": "Unable to perform LLM analysis - using heuristics",
                "strengths": ["Basic functionality observed"],
                "weaknesses": ["Detailed analysis unavailable"],
                "failure_modes": ["LLM analysis failed"],
                "recommendations": ["Manual review recommended"],
                "difficulty_adjustment": "maintain",
                "confidence": 0.3
            }
        else:
            analysis = {
                "fallback": True,
                "message": "LLM analysis unavailable - manual intervention required"
            }
        
        return LLMResponse(
            text="Heuristic fallback analysis - LLM providers unavailable",
            structured_data=analysis,
            confidence=0.3,
            processing_time=0.001,
            model_used="heuristic-fallback"
        )
    
    def _is_provider_available(self, provider: str) -> bool:
        """Check if provider is available."""
        config = self.providers.get(provider, {})
        
        if provider in ["openai", "anthropic"]:
            return bool(config.get("api_key"))
        elif provider == "local":
            return bool(config.get("endpoint"))
        
        return False
    
    def _get_cache_key(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate cache key for prompt."""
        import hashlib
        content = f"{prompt}|{temperature}|{max_tokens}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """Get cached response if valid."""
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
            else:
                del self.cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: LLMResponse):
        """Cache LLM response."""
        self.cache[cache_key] = (response, time.time())
        
        # Cleanup old cache entries
        if len(self.cache) > 1000:  # Limit cache size
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
    
    def _track_performance(self, provider: str, processing_time: float, success: bool):
        """Track provider performance."""
        if provider not in self.success_rates:
            self.success_rates[provider] = []
        
        self.success_rates[provider].append(success)
        self.response_times.append(processing_time)
        
        # Keep only recent history
        if len(self.success_rates[provider]) > 100:
            self.success_rates[provider] = self.success_rates[provider][-50:]
        
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-500:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get LLM integration performance statistics."""
        stats = {
            "avg_response_time": np.mean(self.response_times) if self.response_times else 0,
            "cache_hit_rate": len(self.cache) / max(1, len(self.response_times)),
            "provider_success_rates": {}
        }
        
        for provider, successes in self.success_rates.items():
            if successes:
                stats["provider_success_rates"][provider] = np.mean(successes)
        
        return stats