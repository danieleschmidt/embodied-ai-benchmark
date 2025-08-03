"""Main CLI entry point for embodied AI benchmark."""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..tasks.task_factory import make_env, get_available_tasks
from ..core.base_agent import RandomAgent
from ..evaluation.benchmark_suite import BenchmarkSuite
from ..multiagent.multi_agent_benchmark import MultiAgentBenchmark
from ..database.connection import get_database
from ..database.migrations.001_create_tables import run_migration
from ..database.seeds.task_seed import run_seeds
from ..repositories.experiment_repository import ExperimentRepository, BenchmarkRunRepository


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('benchmark.log')
        ]
    )


def create_random_agent(agent_config: Dict[str, Any]) -> RandomAgent:
    """Create a random agent with given configuration."""
    default_config = {
        "agent_id": "random_agent",
        "action_space": {
            "type": "continuous",
            "shape": (7,),
            "low": [-1] * 7,
            "high": [1] * 7
        }
    }
    default_config.update(agent_config)
    return RandomAgent(default_config)


def run_single_task(args) -> int:
    """Run benchmark on a single task."""
    print(f"Running benchmark on task: {args.task}")
    
    try:
        # Create environment
        env_config = {}
        if args.config:
            with open(args.config) as f:
                env_config = json.load(f).get("env", {})
        
        env = make_env(
            args.task, 
            simulator=args.simulator,
            **env_config
        )
        
        # Create agent
        agent_config = {"agent_id": f"agent_{args.task}"}
        if args.config:
            with open(args.config) as f:
                agent_config.update(json.load(f).get("agent", {}))
        
        agent = create_random_agent(agent_config)
        
        # Create benchmark suite
        suite = BenchmarkSuite()
        
        # Run evaluation
        print(f"Running {args.episodes} episodes...")
        results = suite.evaluate(
            env=env,
            agent=agent,
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            seed=args.seed,
            parallel=args.parallel,
            num_workers=args.workers
        )
        
        # Print results
        print(f"\\nResults:")
        print(f"Success Rate: {results['success_rate']:.2%}")
        print(f"Average Steps: {results['avg_steps']:.1f}")
        print(f"Average Reward: {results['avg_reward']:.2f}")
        print(f"Total Time: {results['total_time']:.1f}s")
        
        # Print metrics
        if "metrics" in results:
            print(f"\\nMetrics:")
            for metric_name, metric_data in results["metrics"].items():
                mean_val = metric_data.get("mean", 0)
                print(f"  {metric_name}: {mean_val:.3f}")
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\\nResults saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return 1


def run_multi_agent_task(args) -> int:
    """Run benchmark on a multi-agent task."""
    print(f"Running multi-agent benchmark on task: {args.task}")
    
    try:
        # Create environment
        env_config = {"num_agents": args.num_agents}
        if args.config:
            with open(args.config) as f:
                env_config.update(json.load(f).get("env", {}))
        
        env = make_env(args.task, **env_config)
        
        # Create agents
        agents = {}
        for i in range(args.num_agents):
            agent_config = {
                "agent_id": f"agent_{i}",
                "role": "leader" if i == 0 else "follower"
            }
            
            if args.config:
                with open(args.config) as f:
                    agent_config.update(json.load(f).get("agent", {}))
            
            agents[f"agent_{i}"] = create_random_agent(agent_config)
        
        # Create multi-agent benchmark
        ma_benchmark = MultiAgentBenchmark()
        
        # Run evaluation
        print(f"Running {args.episodes} episodes with {args.num_agents} agents...")
        results = ma_benchmark.evaluate(
            env=env,
            agents=agents,
            num_episodes=args.episodes,
            metrics=['success', 'coordination', 'efficiency', 'communication'],
            seed=args.seed
        )
        
        # Print results
        print(f"\\nResults:")
        print(f"Success Rate: {results['success_rate']:.2%}")
        print(f"Average Collaboration Events: {results['avg_collaboration_events']:.1f}")
        print(f"Average Messages: {results['avg_messages']:.1f}")
        
        # Print metrics
        if "metrics" in results:
            print(f"\\nMetrics:")
            for metric_name, metric_data in results["metrics"].items():
                mean_val = metric_data.get("mean", 0)
                print(f"  {metric_name}: {mean_val:.3f}")
        
        # Analyze cooperation
        cooperation_analysis = ma_benchmark.analyze_cooperation(results)
        print(f"\\nCooperation Analysis:")
        print(f"Overall Cooperation Score: {cooperation_analysis['overall_cooperation_score']:.3f}")
        
        # Save results if requested
        if args.output:
            output_data = {
                "results": results,
                "cooperation_analysis": cooperation_analysis
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\\nResults saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error running multi-agent benchmark: {e}")
        return 1


def list_tasks(args) -> int:
    """List available tasks."""
    print("Available tasks:")
    
    tasks = get_available_tasks()
    for task_id in tasks:
        try:
            from ..tasks.task_factory import get_task_info
            info = get_task_info(task_id)
            print(f"  {task_id}:")
            print(f"    Task Type: {info.get('task_class', 'Unknown')}")
            print(f"    Environment: {info.get('env_class', 'Unknown')}")
            print()
        except Exception as e:
            print(f"  {task_id}: (Error getting info: {e})")
    
    return 0


def init_database(args) -> int:
    """Initialize database schema."""
    print("Initializing database...")
    
    try:
        db = get_database()
        
        # Run migrations
        print("Running migrations...")
        run_migration(db)
        
        # Run seeds if requested
        if args.seed_data:
            print("Seeding data...")
            run_seeds(db)
        
        print("Database initialized successfully!")
        return 0
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        return 1


def run_experiment(args) -> int:
    """Run a full experiment with database tracking."""
    print(f"Running experiment: {args.name}")
    
    try:
        # Load experiment configuration
        if not Path(args.config).exists():
            print(f"Configuration file not found: {args.config}")
            return 1
        
        with open(args.config) as f:
            config = json.load(f)
        
        # Setup database
        db = get_database()
        exp_repo = ExperimentRepository(db)
        run_repo = BenchmarkRunRepository(db)
        
        # Create experiment record
        exp_id = exp_repo.create_experiment(
            name=args.name,
            description=config.get("description", "CLI experiment"),
            config=config
        )
        
        print(f"Created experiment with ID: {exp_id}")
        
        # Run tasks
        tasks = config.get("tasks", [])
        agents_config = config.get("agents", [{"type": "random"}])
        
        for task_config in tasks:
            task_name = task_config["name"]
            
            for agent_config in agents_config:
                print(f"\\nRunning {agent_config.get('name', 'agent')} on {task_name}...")
                
                # Create run record
                run_id = run_repo.create_run(
                    experiment_id=exp_id,
                    agent_name=agent_config.get("name", "random_agent"),
                    task_name=task_name,
                    config={"agent": agent_config, "task": task_config}
                )
                
                try:
                    # Create environment and agent
                    env = make_env(task_name, **task_config.get("config", {}))
                    agent = create_random_agent(agent_config.get("config", {}))
                    
                    # Run benchmark
                    suite = BenchmarkSuite()
                    results = suite.evaluate(
                        env=env,
                        agent=agent,
                        num_episodes=task_config.get("episodes", 10),
                        max_steps_per_episode=task_config.get("max_steps", 1000),
                        seed=args.seed
                    )
                    
                    # Update run with results
                    run_repo.update_run_results(run_id, results)
                    
                    print(f"  Success Rate: {results['success_rate']:.2%}")
                    print(f"  Avg Steps: {results['avg_steps']:.1f}")
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    run_repo.mark_run_failed(run_id, str(e))
        
        # Mark experiment as completed
        exp_repo.update_status(exp_id, "completed")
        
        # Generate summary
        summary = exp_repo.get_experiment_summary(exp_id)
        print(f"\\nExperiment Summary:")
        print(f"Total Runs: {summary['total_runs']}")
        print(f"Completed: {summary['completed_runs']}")
        print(f"Failed: {summary['failed_runs']}")
        print(f"Average Success Rate: {summary.get('avg_success_rate', 0):.2%}")
        
        return 0
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        return 1


def show_leaderboard(args) -> int:
    """Show leaderboard for a task."""
    print(f"Leaderboard for task: {args.task}")
    
    try:
        db = get_database()
        run_repo = BenchmarkRunRepository(db)
        
        leaderboard = run_repo.get_task_leaderboard(args.task, limit=args.limit)
        
        if not leaderboard:
            print("No results found for this task.")
            return 0
        
        print(f"\\n{'Rank':<6} {'Agent':<20} {'Success Rate':<12} {'Avg Reward':<12} {'Runs':<6}")
        print("-" * 60)
        
        for entry in leaderboard:
            print(f"{entry['rank']:<6} {entry['agent_name']:<20} "
                  f"{entry['avg_success_rate']:.2%}        "
                  f"{entry['avg_reward']:<12.2f} {entry['num_runs']:<6}")
        
        return 0
        
    except Exception as e:
        print(f"Error showing leaderboard: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Embodied AI Benchmark++ CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Single task benchmark
    single_parser = subparsers.add_parser("run", help="Run benchmark on single task")
    single_parser.add_argument("task", help="Task name")
    single_parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    single_parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    single_parser.add_argument("--simulator", default="habitat", help="Simulator backend")
    single_parser.add_argument("--config", help="Configuration file")
    single_parser.add_argument("--output", help="Output file for results")
    single_parser.add_argument("--seed", type=int, help="Random seed")
    single_parser.add_argument("--parallel", action="store_true", help="Run episodes in parallel")
    single_parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    single_parser.set_defaults(func=run_single_task)
    
    # Multi-agent benchmark
    multi_parser = subparsers.add_parser("run-multi", help="Run multi-agent benchmark")
    multi_parser.add_argument("task", help="Multi-agent task name")
    multi_parser.add_argument("--num-agents", type=int, default=2, help="Number of agents")
    multi_parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    multi_parser.add_argument("--config", help="Configuration file")
    multi_parser.add_argument("--output", help="Output file for results")
    multi_parser.add_argument("--seed", type=int, help="Random seed")
    multi_parser.set_defaults(func=run_multi_agent_task)
    
    # List tasks
    list_parser = subparsers.add_parser("list", help="List available tasks")
    list_parser.set_defaults(func=list_tasks)
    
    # Database initialization
    db_parser = subparsers.add_parser("init-db", help="Initialize database")
    db_parser.add_argument("--seed-data", action="store_true", help="Seed with sample data")
    db_parser.set_defaults(func=init_database)
    
    # Experiment runner
    exp_parser = subparsers.add_parser("experiment", help="Run full experiment")
    exp_parser.add_argument("name", help="Experiment name")
    exp_parser.add_argument("config", help="Experiment configuration file")
    exp_parser.add_argument("--seed", type=int, help="Random seed")
    exp_parser.set_defaults(func=run_experiment)
    
    # Leaderboard
    board_parser = subparsers.add_parser("leaderboard", help="Show task leaderboard")
    board_parser.add_argument("task", help="Task name")
    board_parser.add_argument("--limit", type=int, default=10, help="Number of entries to show")
    board_parser.set_defaults(func=show_leaderboard)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Run command
    if hasattr(args, 'func'):
        return args.func(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())