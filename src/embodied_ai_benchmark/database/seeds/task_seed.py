"""Seed data for task metadata."""

import json
from ..connection import DatabaseConnection


def seed_task_metadata(db: DatabaseConnection):
    """Seed the task_metadata table with default tasks."""
    
    tasks = [
        {
            "task_name": "FurnitureAssembly-v0",
            "task_type": "manipulation",
            "difficulty": "medium",
            "description": "Single-agent furniture assembly task requiring precise manipulation and planning",
            "requirements": {
                "action_space": "continuous",
                "observation_space": ["rgb", "depth", "proprioception"],
                "skills": ["pick", "place", "connect"],
                "coordination": False
            },
            "success_criteria": "All furniture parts connected with stability > 0.9",
            "time_limit": 300,
            "max_steps": 1000,
            "multi_agent": False
        },
        {
            "task_name": "PointGoal-v0", 
            "task_type": "navigation",
            "difficulty": "easy",
            "description": "Navigate to a goal position while avoiding obstacles",
            "requirements": {
                "action_space": "continuous",
                "observation_space": ["rgb", "depth", "occupancy_grid"],
                "skills": ["navigation", "path_planning"],
                "coordination": False
            },
            "success_criteria": "Reach goal position within radius tolerance",
            "time_limit": 180,
            "max_steps": 500,
            "multi_agent": False
        },
        {
            "task_name": "CooperativeFurnitureAssembly-v0",
            "task_type": "multi_agent_manipulation", 
            "difficulty": "hard",
            "description": "Multi-agent cooperative furniture assembly requiring coordination",
            "requirements": {
                "action_space": "continuous",
                "observation_space": ["rgb", "depth", "proprioception", "agent_states"],
                "skills": ["pick", "place", "connect", "communicate", "coordinate"],
                "coordination": True,
                "min_agents": 2,
                "max_agents": 4
            },
            "success_criteria": "All parts assembled with successful coordination events",
            "time_limit": 400,
            "max_steps": 1500,
            "multi_agent": True
        },
        {
            "task_name": "SearchAndRescue-v0",
            "task_type": "multi_agent_navigation",
            "difficulty": "hard", 
            "description": "Multi-agent search and rescue in complex environment",
            "requirements": {
                "action_space": "continuous",
                "observation_space": ["rgb", "depth", "thermal", "map"],
                "skills": ["navigation", "search", "rescue", "communicate"],
                "coordination": True,
                "min_agents": 3,
                "max_agents": 6
            },
            "success_criteria": "All victims found and evacuated safely",
            "time_limit": 600,
            "max_steps": 2000,
            "multi_agent": True
        },
        {
            "task_name": "KitchenCooking-v0",
            "task_type": "manipulation",
            "difficulty": "hard",
            "description": "Complex cooking task with sequential dependencies", 
            "requirements": {
                "action_space": "continuous",
                "observation_space": ["rgb", "depth", "tactile", "temperature"],
                "skills": ["pick", "place", "pour", "heat", "stir"],
                "coordination": False
            },
            "success_criteria": "Recipe completed with quality score > 0.8",
            "time_limit": 900,
            "max_steps": 2000,
            "multi_agent": False
        },
        {
            "task_name": "WarehouseLogistics-v0",
            "task_type": "multi_agent_navigation",
            "difficulty": "medium",
            "description": "Multi-agent warehouse order fulfillment and logistics",
            "requirements": {
                "action_space": "continuous", 
                "observation_space": ["rgb", "depth", "lidar", "inventory"],
                "skills": ["navigation", "pick", "sort", "coordinate"],
                "coordination": True,
                "min_agents": 2,
                "max_agents": 8
            },
            "success_criteria": "All orders fulfilled within time and accuracy limits",
            "time_limit": 300,
            "max_steps": 1000,
            "multi_agent": True
        }
    ]
    
    for task in tasks:
        # Check if task already exists
        existing = db.execute_query(
            "SELECT id FROM task_metadata WHERE task_name = ?",
            (task["task_name"],)
        )
        
        if not existing:
            # Insert new task
            if db.db_type == "sqlite":
                requirements_json = json.dumps(task["requirements"])
                db.execute_insert(
                    """
                    INSERT INTO task_metadata 
                    (task_name, task_type, difficulty, description, requirements, 
                     success_criteria, time_limit, max_steps, multi_agent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task["task_name"],
                        task["task_type"],
                        task["difficulty"],
                        task["description"],
                        requirements_json,
                        task["success_criteria"],
                        task["time_limit"],
                        task["max_steps"],
                        task["multi_agent"]
                    )
                )
            else:  # PostgreSQL
                db.execute_insert(
                    """
                    INSERT INTO task_metadata 
                    (task_name, task_type, difficulty, description, requirements,
                     success_criteria, time_limit, max_steps, multi_agent)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        task["task_name"],
                        task["task_type"], 
                        task["difficulty"],
                        task["description"],
                        json.dumps(task["requirements"]),
                        task["success_criteria"],
                        task["time_limit"],
                        task["max_steps"],
                        task["multi_agent"]
                    )
                )
    
    print(f"Seeded {len(tasks)} tasks to database")


def seed_sample_experiments(db: DatabaseConnection):
    """Seed sample experiment configurations."""
    
    experiments = [
        {
            "name": "baseline_random_agents",
            "description": "Baseline evaluation using random agents across all tasks",
            "config": {
                "agent_type": "random",
                "num_episodes": 100,
                "tasks": ["FurnitureAssembly-v0", "PointGoal-v0"],
                "metrics": ["success_rate", "efficiency", "safety"]
            },
            "status": "pending"
        },
        {
            "name": "multi_agent_coordination_study",
            "description": "Study of coordination strategies in multi-agent tasks",
            "config": {
                "agent_type": "scripted",
                "num_episodes": 50,
                "tasks": ["CooperativeFurnitureAssembly-v0", "SearchAndRescue-v0"],
                "metrics": ["success_rate", "coordination", "communication"],
                "num_agents": [2, 3, 4]
            },
            "status": "pending"
        },
        {
            "name": "difficulty_progression_analysis",
            "description": "Analysis of task difficulty progression and learning curves",
            "config": {
                "agent_type": "learning",
                "num_episodes": 200,
                "tasks": ["PointGoal-v0", "FurnitureAssembly-v0", "KitchenCooking-v0"],
                "difficulty_progression": True,
                "adaptive_curriculum": True
            },
            "status": "pending"
        }
    ]
    
    for exp in experiments:
        # Check if experiment already exists
        existing = db.execute_query(
            "SELECT id FROM experiments WHERE name = ?",
            (exp["name"],)
        )
        
        if not existing:
            if db.db_type == "sqlite":
                config_json = json.dumps(exp["config"])
                db.execute_insert(
                    """
                    INSERT INTO experiments (name, description, config, status)
                    VALUES (?, ?, ?, ?)
                    """,
                    (exp["name"], exp["description"], config_json, exp["status"])
                )
            else:  # PostgreSQL
                db.execute_insert(
                    """
                    INSERT INTO experiments (name, description, config, status)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (exp["name"], exp["description"], 
                     json.dumps(exp["config"]), exp["status"])
                )
    
    print(f"Seeded {len(experiments)} experiments to database")


def run_seeds(db: DatabaseConnection):
    """Run all seed functions."""
    seed_task_metadata(db)
    seed_sample_experiments(db)