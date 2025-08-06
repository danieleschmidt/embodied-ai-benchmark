"""Tests for LLM-guided curriculum learning system."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from embodied_ai_benchmark.curriculum.llm_curriculum import (
    PerformanceAnalysis, 
    LLMCurriculum, 
    CurriculumTrainer
)


class TestPerformanceAnalysis:
    """Test performance analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = PerformanceAnalysis()
    
    def test_analyze_agent_performance_basic(self):
        """Test basic performance analysis."""
        metrics_history = [
            {"success_rate": 0.8, "efficiency": 0.7, "completion_time": 50},
            {"success_rate": 0.85, "efficiency": 0.75, "completion_time": 45},
            {"success_rate": 0.9, "efficiency": 0.8, "completion_time": 40},
        ]
        
        analysis = self.analyzer.analyze_agent_performance(metrics_history)
        
        assert "trends" in analysis
        assert "strengths" in analysis
        assert "weaknesses" in analysis
        assert "recommendations" in analysis
        
        # Should identify improvement trend
        trends = analysis["trends"]
        assert "improving" in trends["success_rate"].lower()
        assert "improving" in trends["efficiency"].lower()
    
    def test_analyze_empty_history(self):
        """Test analysis with empty metrics history."""
        analysis = self.analyzer.analyze_agent_performance([])
        
        assert analysis["trends"] == {}
        assert "insufficient data" in analysis["recommendations"][0].lower()
    
    def test_analyze_declining_performance(self):
        """Test analysis with declining performance."""
        declining_metrics = [
            {"success_rate": 0.9, "efficiency": 0.8, "completion_time": 40},
            {"success_rate": 0.7, "efficiency": 0.6, "completion_time": 60},
            {"success_rate": 0.5, "efficiency": 0.4, "completion_time": 80},
        ]
        
        analysis = self.analyzer.analyze_agent_performance(declining_metrics)
        
        trends = analysis["trends"]
        assert "declining" in trends["success_rate"].lower()
        assert "declining" in trends["efficiency"].lower()
        
        # Should identify performance issues as weakness
        weaknesses = analysis["weaknesses"]
        assert len(weaknesses) > 0
    
    def test_identify_learning_plateaus(self):
        """Test identification of learning plateaus."""
        plateau_metrics = [
            {"success_rate": 0.75, "efficiency": 0.7, "completion_time": 50},
            {"success_rate": 0.76, "efficiency": 0.71, "completion_time": 49},
            {"success_rate": 0.74, "efficiency": 0.69, "completion_time": 51},
            {"success_rate": 0.75, "efficiency": 0.7, "completion_time": 50},
        ]
        
        analysis = self.analyzer.analyze_agent_performance(plateau_metrics)
        
        trends = analysis["trends"]
        # Should detect stable/plateau performance
        for metric_name, trend in trends.items():
            assert "stable" in trend.lower() or "plateau" in trend.lower()


class TestLLMCurriculum:
    """Test LLM curriculum learning system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = Mock()
        self.curriculum = LLMCurriculum(llm_client=self.mock_llm)
    
    def test_initialization(self):
        """Test curriculum initialization."""
        assert self.curriculum.llm_client == self.mock_llm
        assert self.curriculum.difficulty_range == (0.0, 1.0)
        assert self.curriculum.adaptation_strategy == "performance_based"
    
    @patch('json.loads')
    def test_generate_curriculum_success(self, mock_json_loads):
        """Test successful curriculum generation."""
        # Mock LLM response
        mock_response = {
            "tasks": [
                {
                    "name": "basic_navigation",
                    "difficulty": 0.3,
                    "focus_areas": ["movement", "obstacle_avoidance"],
                    "estimated_duration": "2 hours"
                },
                {
                    "name": "advanced_manipulation", 
                    "difficulty": 0.7,
                    "focus_areas": ["grasping", "precision"],
                    "estimated_duration": "4 hours"
                }
            ],
            "reasoning": "Progressive difficulty to build foundational skills"
        }
        
        mock_json_loads.return_value = mock_response
        self.mock_llm.generate_text.return_value = json.dumps(mock_response)
        
        curriculum = self.curriculum.generate_curriculum(
            agent_performance={"success_rate": 0.6},
            learning_objectives=["navigation", "manipulation"],
            constraints={"max_difficulty": 0.8}
        )
        
        assert isinstance(curriculum, dict)
        assert "tasks" in curriculum
        assert len(curriculum["tasks"]) == 2
        assert curriculum["tasks"][0]["difficulty"] < curriculum["tasks"][1]["difficulty"]
    
    def test_generate_curriculum_with_constraints(self):
        """Test curriculum generation with specific constraints."""
        mock_response = {
            "tasks": [{"name": "test_task", "difficulty": 0.5}],
            "reasoning": "Test reasoning"
        }
        
        self.mock_llm.generate_text.return_value = json.dumps(mock_response)
        
        constraints = {
            "max_difficulty": 0.6,
            "min_tasks": 3,
            "focus_areas": ["navigation"],
            "time_budget": "1 day"
        }
        
        curriculum = self.curriculum.generate_curriculum(
            agent_performance={"success_rate": 0.4},
            learning_objectives=["basic_skills"],
            constraints=constraints
        )
        
        # Verify LLM was called with constraints
        call_args = self.mock_llm.generate_text.call_args[1]
        prompt = call_args["prompt"]
        
        assert "0.6" in prompt  # max_difficulty
        assert "navigation" in prompt  # focus_areas
        assert "1 day" in prompt  # time_budget
    
    def test_adapt_existing_curriculum(self):
        """Test adaptation of existing curriculum based on performance."""
        existing_curriculum = {
            "tasks": [
                {"name": "task1", "difficulty": 0.3, "performance": 0.9},
                {"name": "task2", "difficulty": 0.7, "performance": 0.4}
            ]
        }
        
        mock_adaptation = {
            "adaptations": [
                {
                    "task": "task1", 
                    "action": "increase_difficulty",
                    "new_difficulty": 0.5,
                    "reason": "High performance indicates task is too easy"
                },
                {
                    "task": "task2",
                    "action": "decrease_difficulty", 
                    "new_difficulty": 0.5,
                    "reason": "Low performance indicates task is too hard"
                }
            ]
        }
        
        self.mock_llm.generate_text.return_value = json.dumps(mock_adaptation)
        
        adapted = self.curriculum.adapt_curriculum(
            current_curriculum=existing_curriculum,
            performance_data={"overall_progress": 0.65}
        )
        
        assert "adaptations" in adapted
        assert len(adapted["adaptations"]) == 2
    
    def test_invalid_json_response_handling(self):
        """Test handling of invalid JSON responses from LLM."""
        # Mock invalid JSON response
        self.mock_llm.generate_text.return_value = "Invalid JSON response"
        
        curriculum = self.curriculum.generate_curriculum(
            agent_performance={"success_rate": 0.5},
            learning_objectives=["test"]
        )
        
        # Should return fallback curriculum
        assert isinstance(curriculum, dict)
        assert "tasks" in curriculum
        assert curriculum["tasks"][0]["name"] == "fallback_basic_task"
    
    def test_llm_client_error_handling(self):
        """Test handling of LLM client errors."""
        # Mock LLM client exception
        self.mock_llm.generate_text.side_effect = Exception("LLM service unavailable")
        
        curriculum = self.curriculum.generate_curriculum(
            agent_performance={"success_rate": 0.5},
            learning_objectives=["test"]
        )
        
        # Should return fallback curriculum
        assert isinstance(curriculum, dict)
        assert "error" in curriculum
        assert "fallback" in curriculum


class TestCurriculumTrainer:
    """Test curriculum trainer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_curriculum = Mock()
        self.trainer = CurriculumTrainer(curriculum_system=self.mock_curriculum)
    
    def test_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.curriculum_system == self.mock_curriculum
        assert self.trainer.training_history == []
        assert self.trainer.current_task_index == 0
    
    def test_train_agent_with_curriculum(self):
        """Test agent training with curriculum."""
        # Mock curriculum
        mock_tasks = [
            {"name": "easy_task", "difficulty": 0.3, "max_episodes": 10},
            {"name": "medium_task", "difficulty": 0.6, "max_episodes": 15},
            {"name": "hard_task", "difficulty": 0.9, "max_episodes": 20}
        ]
        
        self.mock_curriculum.generate_curriculum.return_value = {"tasks": mock_tasks}
        
        # Mock agent and environment
        mock_agent = Mock()
        mock_env = Mock()
        
        # Mock training results
        def mock_train_episode(task_config):
            return {
                "success": True,
                "episode_reward": 100,
                "completion_time": 50,
                "task_name": task_config["name"]
            }
        
        results = self.trainer.train_agent_with_curriculum(
            agent=mock_agent,
            env=mock_env,
            learning_objectives=["navigation"],
            max_episodes_per_task=10,
            train_episode_func=mock_train_episode
        )
        
        assert isinstance(results, dict)
        assert "curriculum_results" in results
        assert "final_performance" in results
        assert len(results["curriculum_results"]) == len(mock_tasks)
    
    def test_progressive_difficulty_training(self):
        """Test progressive difficulty increase during training."""
        easy_task = {"name": "easy", "difficulty": 0.2, "max_episodes": 5}
        hard_task = {"name": "hard", "difficulty": 0.8, "max_episodes": 5}
        
        self.mock_curriculum.generate_curriculum.return_value = {
            "tasks": [easy_task, hard_task]
        }
        
        mock_agent = Mock()
        mock_env = Mock()
        
        episode_count = 0
        def mock_train_episode(task_config):
            nonlocal episode_count
            episode_count += 1
            # Simulate better performance on easier tasks
            base_reward = 50 if task_config["difficulty"] < 0.5 else 30
            return {
                "success": episode_count <= 5,  # First 5 episodes succeed
                "episode_reward": base_reward,
                "completion_time": 50,
                "task_name": task_config["name"]
            }
        
        results = self.trainer.train_agent_with_curriculum(
            agent=mock_agent,
            env=mock_env,
            learning_objectives=["test"],
            max_episodes_per_task=5,
            train_episode_func=mock_train_episode
        )
        
        curriculum_results = results["curriculum_results"]
        
        # First task should have better performance
        easy_results = curriculum_results[0]
        hard_results = curriculum_results[1]
        
        assert easy_results["average_reward"] >= hard_results["average_reward"]
    
    def test_curriculum_adaptation_during_training(self):
        """Test curriculum adaptation based on training progress."""
        initial_task = {"name": "initial", "difficulty": 0.5, "max_episodes": 3}
        
        self.mock_curriculum.generate_curriculum.return_value = {
            "tasks": [initial_task]
        }
        
        # Mock adaptation response
        adapted_task = {"name": "adapted", "difficulty": 0.7, "max_episodes": 3}
        self.mock_curriculum.adapt_curriculum.return_value = {
            "adaptations": [adapted_task]
        }
        
        mock_agent = Mock()
        mock_env = Mock()
        
        def mock_train_episode(task_config):
            return {
                "success": True,
                "episode_reward": 100,
                "completion_time": 30,
                "task_name": task_config["name"]
            }
        
        results = self.trainer.train_agent_with_curriculum(
            agent=mock_agent,
            env=mock_env,
            learning_objectives=["test"],
            max_episodes_per_task=3,
            train_episode_func=mock_train_episode,
            adapt_curriculum=True
        )
        
        # Should have called adaptation
        self.mock_curriculum.adapt_curriculum.assert_called()
    
    def test_training_history_tracking(self):
        """Test proper tracking of training history."""
        task = {"name": "test_task", "difficulty": 0.5, "max_episodes": 2}
        self.mock_curriculum.generate_curriculum.return_value = {"tasks": [task]}
        
        mock_agent = Mock()
        mock_env = Mock()
        
        episode_results = []
        def mock_train_episode(task_config):
            result = {
                "success": True,
                "episode_reward": len(episode_results) * 10,  # Increasing rewards
                "completion_time": 50,
                "task_name": task_config["name"]
            }
            episode_results.append(result)
            return result
        
        self.trainer.train_agent_with_curriculum(
            agent=mock_agent,
            env=mock_env,
            learning_objectives=["test"],
            max_episodes_per_task=2,
            train_episode_func=mock_train_episode
        )
        
        # Check training history was recorded
        assert len(self.trainer.training_history) > 0
        
        # Verify history contains episode data
        for entry in self.trainer.training_history:
            assert "episode_result" in entry
            assert "task_config" in entry
            assert "timestamp" in entry
    
    def test_early_stopping_on_mastery(self):
        """Test early stopping when agent masters a task."""
        task = {"name": "masterable_task", "difficulty": 0.4, "max_episodes": 10}
        self.mock_curriculum.generate_curriculum.return_value = {"tasks": [task]}
        
        mock_agent = Mock()
        mock_env = Mock()
        
        episode_count = 0
        def mock_train_episode(task_config):
            nonlocal episode_count
            episode_count += 1
            # Perfect performance after 3 episodes
            return {
                "success": True,
                "episode_reward": 100,
                "completion_time": 20,
                "task_name": task_config["name"]
            }
        
        results = self.trainer.train_agent_with_curriculum(
            agent=mock_agent,
            env=mock_env,
            learning_objectives=["test"],
            max_episodes_per_task=10,
            train_episode_func=mock_train_episode,
            early_stopping_threshold=0.95  # Stop at 95% success rate
        )
        
        # Should have stopped early
        task_results = results["curriculum_results"][0]
        assert task_results["episodes_completed"] < 10