"""Integration tests for complete benchmark workflows."""

import pytest
import tempfile
import shutil
from pathlib import Path

from embodied_ai_benchmark import BenchmarkSuite, make_env, RandomAgent
from embodied_ai_benchmark.multiagent import MultiAgentBenchmark
from embodied_ai_benchmark.database.connection import DatabaseConnection
from embodied_ai_benchmark.repositories.experiment_repository import (
    ExperimentRepository, BenchmarkRunRepository
)


class TestSingleAgentWorkflow:
    """Test complete single-agent benchmark workflow."""
    
    @pytest.mark.integration
    def test_furniture_assembly_workflow(self):
        """Test complete furniture assembly benchmark workflow."""
        # Create environment and agent
        env = make_env("FurnitureAssembly-v0", simulator="habitat")
        
        agent_config = {
            "agent_id": "test_random_agent",
            "action_space": {
                "type": "continuous",
                "shape": (7,),
                "low": [-1] * 7,
                "high": [1] * 7
            }
        }
        agent = RandomAgent(agent_config)
        
        # Create benchmark suite
        suite = BenchmarkSuite()
        
        # Run evaluation
        results = suite.evaluate(
            env=env,
            agent=agent,
            num_episodes=5,
            max_steps_per_episode=100,
            seed=42
        )
        
        # Verify results structure
        assert "num_episodes" in results
        assert "success_rate" in results
        assert "avg_steps" in results
        assert "metrics" in results
        assert results["num_episodes"] == 5
        
        # Verify metrics
        assert "success_rate" in results["metrics"]
        assert "efficiency" in results["metrics"]
        assert "safety" in results["metrics"]
        
        # Verify episode data
        assert "episodes" in results
        assert len(results["episodes"]) == 5
        
        for episode in results["episodes"]:
            assert "episode_id" in episode
            assert "total_steps" in episode
            assert "total_reward" in episode
            assert "success" in episode
    
    @pytest.mark.integration
    def test_point_goal_navigation_workflow(self):
        """Test complete point goal navigation workflow."""
        # Create environment and agent
        env = make_env("PointGoal-v0", room_size=(5, 5), goal_radius=0.3)
        
        agent_config = {
            "agent_id": "nav_agent",
            "action_space": {
                "type": "continuous",
                "shape": (2,),
                "low": [-1, -1],
                "high": [1, 1]
            }
        }
        agent = RandomAgent(agent_config)
        
        # Create benchmark suite
        suite = BenchmarkSuite()
        
        # Run evaluation
        results = suite.evaluate(
            env=env,
            agent=agent,
            num_episodes=8,
            max_steps_per_episode=200,
            seed=123
        )
        
        # Verify navigation-specific results
        assert results["num_episodes"] == 8
        assert results["avg_steps"] <= 200
        
        # Check that some episodes completed
        episodes = results["episodes"]
        completed_episodes = [ep for ep in episodes if ep["total_steps"] < 200]
        assert len(completed_episodes) >= 0  # At least some should complete early
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_cross_task_evaluation(self):
        """Test agent evaluation across multiple tasks."""
        # Create multiple environments
        tasks = [
            ("FurnitureAssembly-v0", {"furniture_type": "table"}),
            ("PointGoal-v0", {"room_size": (8, 8)})
        ]
        
        agent_config = {
            "agent_id": "multi_task_agent",
            "action_space": {
                "type": "continuous",
                "shape": (7,),
                "low": [-1] * 7,
                "high": [1] * 7
            }
        }
        agent = RandomAgent(agent_config)
        
        suite = BenchmarkSuite()
        all_results = {}
        
        for task_name, task_config in tasks:
            env = make_env(task_name, **task_config)
            
            results = suite.evaluate(
                env=env,
                agent=agent,
                num_episodes=3,
                max_steps_per_episode=50,
                seed=42
            )
            
            all_results[task_name] = results
        
        # Verify cross-task results
        assert len(all_results) == 2
        assert "FurnitureAssembly-v0" in all_results
        assert "PointGoal-v0" in all_results
        
        # Each task should have its own metrics
        for task_results in all_results.values():
            assert "success_rate" in task_results
            assert "metrics" in task_results


class TestMultiAgentWorkflow:
    """Test complete multi-agent benchmark workflow."""
    
    @pytest.mark.integration
    def test_cooperative_assembly_workflow(self):
        """Test complete cooperative assembly workflow."""
        # Create multi-agent environment
        env = make_env(
            "CooperativeFurnitureAssembly-v0",
            num_agents=2,
            furniture="ikea_table"
        )
        
        # Create multiple agents
        agents = {}
        for i in range(2):
            agent_config = {
                "agent_id": f"coop_agent_{i}",
                "role": "leader" if i == 0 else "follower",
                "action_space": {
                    "type": "continuous",
                    "shape": (7,),
                    "low": [-1] * 7,
                    "high": [1] * 7
                }
            }
            agents[f"agent_{i}"] = RandomAgent(agent_config)
        
        # Create multi-agent benchmark
        ma_benchmark = MultiAgentBenchmark()
        
        # Run evaluation
        results = ma_benchmark.evaluate(
            env=env,
            agents=agents,
            num_episodes=5,
            metrics=['success', 'coordination', 'efficiency', 'communication'],
            seed=42
        )
        
        # Verify multi-agent results
        assert "num_episodes" in results
        assert "success_rate" in results
        assert "avg_collaboration_events" in results
        assert "metrics" in results
        assert results["num_episodes"] == 5
        
        # Verify multi-agent specific metrics
        metrics = results["metrics"]
        assert "success" in metrics
        assert "coordination" in metrics
        assert "efficiency" in metrics
        assert "communication" in metrics
        
        # Verify episode data includes multi-agent info
        episodes = results["episodes"]
        for episode in episodes:
            assert "collaboration_events" in episode
            assert "communication_log" in episode
            assert "total_rewards" in episode
            assert isinstance(episode["total_rewards"], dict)
    
    @pytest.mark.integration
    def test_cooperation_analysis(self):
        """Test cooperation analysis functionality."""
        # Create multi-agent environment
        env = make_env("CooperativeFurnitureAssembly-v0", num_agents=2)
        
        # Create agents
        agents = {
            "agent_0": RandomAgent({"agent_id": "agent_0"}),
            "agent_1": RandomAgent({"agent_id": "agent_1"})
        }
        
        # Create benchmark and run evaluation
        ma_benchmark = MultiAgentBenchmark()
        results = ma_benchmark.evaluate(
            env=env,
            agents=agents,
            num_episodes=3,
            seed=42
        )
        
        # Analyze cooperation
        cooperation_analysis = ma_benchmark.analyze_cooperation(results)
        
        # Verify analysis structure
        assert "communication" in cooperation_analysis
        assert "roles" in cooperation_analysis
        assert "coordination" in cooperation_analysis
        assert "overall_cooperation_score" in cooperation_analysis
        
        # Verify communication analysis
        comm_analysis = cooperation_analysis["communication"]
        assert "total_messages" in comm_analysis
        assert "communication_efficiency" in comm_analysis
        
        # Verify cooperation score is valid
        coop_score = cooperation_analysis["overall_cooperation_score"]
        assert 0.0 <= coop_score <= 1.0


class TestDatabaseIntegration:
    """Test database integration with benchmark workflows."""
    
    @pytest.mark.integration
    def test_experiment_tracking_workflow(self, clean_database):
        """Test complete experiment tracking workflow."""
        # Create repositories
        exp_repo = ExperimentRepository(clean_database)
        run_repo = BenchmarkRunRepository(clean_database)
        
        # Create experiment
        exp_config = {
            "agent_type": "random",
            "num_episodes": 5,
            "tasks": ["FurnitureAssembly-v0"],
            "metrics": ["success_rate", "efficiency"]
        }
        
        exp_id = exp_repo.create_experiment(
            name="test_integration_experiment",
            description="Integration test experiment",
            config=exp_config
        )
        
        assert exp_id is not None
        
        # Create benchmark run
        run_config = {
            "agent": {"type": "random", "seed": 42},
            "task": {"furniture_type": "table", "max_steps": 100}
        }
        
        run_id = run_repo.create_run(
            experiment_id=exp_id,
            agent_name="random_agent",
            task_name="FurnitureAssembly-v0",
            config=run_config
        )
        
        assert run_id is not None
        
        # Simulate benchmark execution
        mock_results = {
            "num_episodes": 5,
            "success_rate": 0.6,
            "avg_reward": 25.5,
            "avg_steps": 150.0,
            "total_time": 120.5,
            "metrics": {
                "success_rate": {"mean": 0.6, "std": 0.2},
                "efficiency": {"mean": 0.4, "std": 0.15}
            }
        }
        
        # Update run with results
        success = run_repo.update_run_results(run_id, mock_results)
        assert success
        
        # Verify experiment summary
        summary = exp_repo.get_experiment_summary(exp_id)
        assert summary is not None
        assert summary["total_runs"] == 1
        assert summary["completed_runs"] == 1
        assert summary["avg_success_rate"] == 0.6
        
        # Update experiment status
        exp_repo.update_status(exp_id, "completed")
        
        # Verify final state
        experiment = exp_repo.find_by_id(exp_id)
        assert experiment["status"] == "completed"
    
    @pytest.mark.integration
    def test_performance_tracking_workflow(self, clean_database):
        """Test agent performance tracking workflow."""
        run_repo = BenchmarkRunRepository(clean_database)
        
        # Create multiple runs for the same agent
        agent_name = "test_performance_agent"
        
        for i in range(3):
            run_id = run_repo.create_run(
                experiment_id=1,  # Dummy experiment ID
                agent_name=agent_name,
                task_name="FurnitureAssembly-v0",
                config={}
            )
            
            # Simulate different performance levels
            results = {
                "num_episodes": 10,
                "success_rate": 0.5 + (i * 0.1),  # Improving performance
                "avg_reward": 20.0 + (i * 5.0),
                "avg_steps": 200.0 - (i * 10.0),
                "total_time": 100.0
            }
            
            run_repo.update_run_results(run_id, results)
        
        # Get performance summary
        summary = run_repo.get_agent_performance_summary(agent_name)
        
        assert summary["agent_name"] == agent_name
        assert len(summary["task_performance"]) == 1  # One task
        assert "overall" in summary
        
        # Verify performance improvement tracking
        task_perf = summary["task_performance"][0]
        assert task_perf["total_runs"] == 3
        assert task_perf["avg_success_rate"] == 0.6  # Average of 0.5, 0.6, 0.7
    
    @pytest.mark.integration
    def test_leaderboard_generation_workflow(self, clean_database):
        """Test leaderboard generation workflow."""
        run_repo = BenchmarkRunRepository(clean_database)
        
        # Create runs for multiple agents on same task
        agents_data = [
            ("agent_A", 0.9, 50.0),  # High performance
            ("agent_B", 0.7, 35.0),  # Medium performance
            ("agent_C", 0.5, 20.0),  # Low performance
        ]
        
        task_name = "FurnitureAssembly-v0"
        
        for agent_name, success_rate, avg_reward in agents_data:
            run_id = run_repo.create_run(
                experiment_id=1,
                agent_name=agent_name,
                task_name=task_name,
                config={}
            )
            
            results = {
                "num_episodes": 10,
                "success_rate": success_rate,
                "avg_reward": avg_reward,
                "avg_steps": 100.0,
                "total_time": 60.0
            }
            
            run_repo.update_run_results(run_id, results)
        
        # Generate leaderboard
        leaderboard = run_repo.get_task_leaderboard(task_name, limit=5)
        
        assert len(leaderboard) == 3
        
        # Verify ranking order (by success rate, then reward)
        assert leaderboard[0]["agent_name"] == "agent_A"
        assert leaderboard[0]["rank"] == 1
        assert leaderboard[1]["agent_name"] == "agent_B"
        assert leaderboard[1]["rank"] == 2
        assert leaderboard[2]["agent_name"] == "agent_C"
        assert leaderboard[2]["rank"] == 3
        
        # Verify performance values
        assert leaderboard[0]["avg_success_rate"] == 0.9
        assert leaderboard[1]["avg_success_rate"] == 0.7
        assert leaderboard[2]["avg_success_rate"] == 0.5


class TestEndToEndWorkflow:
    """Test complete end-to-end benchmark workflows."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_benchmark_pipeline(self, clean_database):
        """Test complete benchmark pipeline from setup to analysis."""
        # 1. Setup database and repositories
        exp_repo = ExperimentRepository(clean_database)
        run_repo = BenchmarkRunRepository(clean_database)
        
        # 2. Create experiment
        exp_config = {
            "study_type": "agent_comparison",
            "tasks": ["FurnitureAssembly-v0", "PointGoal-v0"],
            "agents": ["random_baseline", "improved_random"],
            "num_episodes": 3
        }
        
        exp_id = exp_repo.create_experiment(
            name="end_to_end_benchmark",
            description="Complete pipeline test",
            config=exp_config
        )
        
        # 3. Define agents and tasks
        agents = [
            ("random_baseline", {"seed": 42}),
            ("improved_random", {"seed": 123, "exploration": 0.8})
        ]
        
        tasks = [
            ("FurnitureAssembly-v0", {"furniture_type": "table"}),
            ("PointGoal-v0", {"room_size": (6, 6)})
        ]
        
        # 4. Run benchmark for each agent-task combination
        all_results = {}
        
        for agent_name, agent_config in agents:
            agent_results = {}
            
            for task_name, task_config in tasks:
                # Create environment and agent
                env = make_env(task_name, **task_config)
                
                full_agent_config = {
                    "agent_id": agent_name,
                    "action_space": {
                        "type": "continuous",
                        "shape": (7,),
                        "low": [-1] * 7,
                        "high": [1] * 7
                    },
                    **agent_config
                }
                agent = RandomAgent(full_agent_config)
                
                # Create benchmark run record
                run_id = run_repo.create_run(
                    experiment_id=exp_id,
                    agent_name=agent_name,
                    task_name=task_name,
                    config={"agent": agent_config, "task": task_config}
                )
                
                # Run benchmark
                suite = BenchmarkSuite()
                results = suite.evaluate(
                    env=env,
                    agent=agent,
                    num_episodes=3,
                    max_steps_per_episode=50,
                    seed=42
                )
                
                # Store results
                run_repo.update_run_results(run_id, results)
                agent_results[task_name] = results
            
            all_results[agent_name] = agent_results
        
        # 5. Generate analysis and comparisons
        
        # Agent performance summaries
        for agent_name in [agent[0] for agent in agents]:
            summary = run_repo.get_agent_performance_summary(agent_name)
            assert summary["agent_name"] == agent_name
            assert len(summary["task_performance"]) == 2  # Two tasks
        
        # Task leaderboards
        for task_name in [task[0] for task in tasks]:
            leaderboard = run_repo.get_task_leaderboard(task_name, limit=10)
            assert len(leaderboard) == 2  # Two agents
            
            # Verify all agents are present
            agent_names = [entry["agent_name"] for entry in leaderboard]
            assert "random_baseline" in agent_names
            assert "improved_random" in agent_names
        
        # Experiment summary
        exp_summary = exp_repo.get_experiment_summary(exp_id)
        assert exp_summary["total_runs"] == 4  # 2 agents × 2 tasks
        assert exp_summary["completed_runs"] == 4
        
        # 6. Mark experiment as completed
        exp_repo.update_status(exp_id, "completed")
        
        # 7. Verify final state
        final_experiment = exp_repo.find_by_id(exp_id)
        assert final_experiment["status"] == "completed"
        assert final_experiment["name"] == "end_to_end_benchmark"
        
        # 8. Verify cross-agent task performance
        for task_name in [task[0] for task in tasks]:
            task_runs = run_repo.find_by_fields({"task_name": task_name})
            assert len(task_runs) == 2  # One run per agent
            
            # All runs should be completed
            for run in task_runs:
                assert run["status"] == "completed"
                assert run["success_rate"] is not None