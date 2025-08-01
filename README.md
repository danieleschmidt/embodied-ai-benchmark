# Embodied-AI Benchmark++

Extends Habitat and ManiSkill with multi-agent tasks including cooperative furniture assembly and LLM-guided curriculum learning. Following the "future directions" blueprint from the 2025 survey, this benchmark pushes the boundaries of embodied AI evaluation.

## Overview

Embodied-AI Benchmark++ provides a comprehensive evaluation suite for next-generation embodied AI systems. By combining multi-agent cooperation, language-guided task learning, and realistic physics simulation, we create challenging scenarios that test the full stack of embodied intelligence - from perception to planning to physical interaction.

## Key Features

- **Multi-Agent Tasks**: 2-8 agents cooperating on complex physical tasks
- **LLM Integration**: Natural language task specification and guidance
- **Curriculum Learning**: Adaptive difficulty progression
- **Cross-Simulator**: Works with Habitat, ManiSkill, and Isaac Sim
- **Realistic Physics**: Soft-body dynamics, friction, and contact modeling
- **Comprehensive Metrics**: Success, efficiency, safety, and collaboration quality

## Installation

```bash
# Basic installation
pip install embodied-ai-benchmark

# With all simulators
pip install embodied-ai-benchmark[all]

# Specific simulator support
pip install embodied-ai-benchmark[habitat]
pip install embodied-ai-benchmark[maniskill]
pip install embodied-ai-benchmark[isaac]

# Development installation
git clone https://github.com/yourusername/embodied-ai-benchmark
cd embodied-ai-benchmark
pip install -e ".[dev]"
```

## Quick Start

### Single-Agent Benchmark

```python
from embodied_ai_benchmark import BenchmarkSuite, make_env

# Create benchmark environment
env = make_env(
    "FurnitureAssembly-v0",
    simulator="habitat",
    render_mode="human"
)

# Run random agent benchmark
benchmark = BenchmarkSuite()
results = benchmark.evaluate(
    env=env,
    agent=random_agent,
    num_episodes=100
)

print(f"Success rate: {results['success_rate']:.2%}")
print(f"Average steps: {results['avg_steps']:.1f}")
print(f"Efficiency: {results['efficiency']:.3f}")
```

### Multi-Agent Cooperative Task

```python
from embodied_ai_benchmark import MultiAgentBenchmark

# Create multi-agent furniture assembly
env = make_env(
    "CooperativeFurnitureAssembly-v0",
    num_agents=2,
    furniture="ikea_table",
    difficulty="medium"
)

# Initialize agents
agents = {
    "agent_0": YourAgent(role="leader"),
    "agent_1": YourAgent(role="follower")
}

# Run cooperative benchmark
ma_benchmark = MultiAgentBenchmark()
results = ma_benchmark.evaluate(
    env=env,
    agents=agents,
    num_episodes=50,
    metrics=['success', 'coordination', 'efficiency', 'communication']
)

# Analyze cooperation quality
ma_benchmark.analyze_cooperation(results)
```

## Task Categories

### Manipulation Tasks

```python
from embodied_ai_benchmark.tasks import ManipulationTasks

# Available manipulation benchmarks
tasks = ManipulationTasks()

# Precision assembly
precision_task = tasks.create_task(
    "precision_assembly",
    components=['gear', 'shaft', 'bearing'],
    tolerance=1e-3,  # 1mm precision required
    time_limit=300   # seconds
)

# Tool use
tool_task = tasks.create_task(
    "tool_use",
    tools=['screwdriver', 'hammer', 'wrench'],
    target_objects=['screw', 'nail', 'bolt'],
    require_proper_grip=True
)

# Deformable object manipulation
soft_task = tasks.create_task(
    "cloth_folding",
    cloth_size=(0.5, 0.5),  # meters
    target_folds=['half', 'quarter', 'diagonal'],
    physics="flex"  # NVIDIA FleX for soft-body
)
```

### Navigation Tasks

```python
from embodied_ai_benchmark.tasks import NavigationTasks

nav_tasks = NavigationTasks()

# Multi-floor navigation
multi_floor = nav_tasks.create_task(
    "multi_floor_delivery",
    building_size="large",
    num_floors=5,
    elevators=True,
    dynamic_obstacles=True
)

# Social navigation
social_nav = nav_tasks.create_task(
    "crowded_space_navigation",
    num_pedestrians=50,
    social_rules=True,
    cultural_norms="western"
)

# Exploration with mapping
exploration = nav_tasks.create_task(
    "unknown_space_exploration",
    area_size=1000,  # m²
    map_building=True,
    return_to_start=True
)
```

### Multi-Agent Collaboration

```python
from embodied_ai_benchmark.tasks import CollaborationTasks

collab_tasks = CollaborationTasks()

# Furniture assembly
furniture_assembly = collab_tasks.create_task(
    "ikea_furniture_assembly",
    furniture_model="kallax_shelf",
    num_agents=2,
    require_coordination=True,
    instruction_type="visual"  # or "text", "video"
)

# Construction
construction = collab_tasks.create_task(
    "block_tower_construction",
    target_height=2.0,  # meters
    num_agents=4,
    blocks_per_agent=10,
    stability_required=True
)

# Search and rescue
search_rescue = collab_tasks.create_task(
    "disaster_area_search",
    area_size=(100, 100),  # meters
    num_victims=10,
    num_agents=6,
    time_limit=600,  # seconds
    hazards=['fire', 'debris', 'unstable_structures']
)
```

## LLM-Guided Curriculum

### Adaptive Task Generation

```python
from embodied_ai_benchmark.curriculum import LLMCurriculum

# Initialize LLM-based curriculum
curriculum = LLMCurriculum(
    llm_model="gpt-4",
    student_model=your_agent,
    domain="manipulation"
)

# Generate curriculum based on current skill level
@curriculum.on_episode_end
def adapt_curriculum(episode_results):
    # LLM analyzes performance
    analysis = curriculum.analyze_performance(episode_results)
    
    # Generate next task
    if analysis['success_rate'] > 0.8:
        next_task = curriculum.increase_difficulty(
            current_task,
            aspects=['precision', 'speed', 'complexity']
        )
    elif analysis['success_rate'] < 0.3:
        next_task = curriculum.decrease_difficulty(
            current_task,
            focus_on=analysis['main_failure_mode']
        )
    else:
        next_task = curriculum.practice_variation(current_task)
    
    return next_task

# Run curriculum learning
curriculum.train(
    num_tasks=1000,
    save_progress=True
)
```

### Natural Language Task Specification

```python
from embodied_ai_benchmark.language import LanguageTaskInterface

# Create language interface
language_interface = LanguageTaskInterface()

# Natural language task specification
task_description = """
Assemble the red chair, but first organize all the parts on the table.
Make sure to use the appropriate tools for each connection type.
The final chair should be stable enough to support 100kg.
"""

# Convert to executable task
task = language_interface.parse_task(
    task_description,
    available_objects=env.get_objects(),
    available_tools=env.get_tools()
)

# Get step-by-step guidance
guidance = language_interface.get_guidance(
    task,
    current_state=env.get_state(),
    agent_capabilities=agent.get_capabilities()
)

print(guidance)
# Output: "First, locate all red chair parts. I see 4 legs, 1 seat, and 1 backrest..."
```

## Evaluation Metrics

### Comprehensive Metric Suite

```python
from embodied_ai_benchmark.metrics import MetricSuite

metrics = MetricSuite()

# Task success metrics
success_metrics = metrics.compute_success_metrics(
    episode_data,
    criteria={
        'task_completed': True,
        'time_limit_met': True,
        'no_damage': True,
        'constraints_satisfied': True
    }
)

# Efficiency metrics
efficiency_metrics = metrics.compute_efficiency_metrics(
    episode_data,
    factors={
        'time': 0.3,
        'energy': 0.2,
        'path_length': 0.2,
        'actions_taken': 0.3
    }
)

# Safety metrics
safety_metrics = metrics.compute_safety_metrics(
    episode_data,
    checks={
        'collisions': {'weight': -10, 'threshold': 0},
        'drops': {'weight': -5, 'threshold': 0},
        'near_misses': {'weight': -1, 'threshold': 5},
        'force_exceeded': {'weight': -10, 'threshold': 100}  # N
    }
)

# Collaboration metrics (multi-agent)
collab_metrics = metrics.compute_collaboration_metrics(
    multi_agent_data,
    measures={
        'coordination_score': coordination_analyzer,
        'communication_efficiency': message_analyzer,
        'workload_balance': workload_analyzer,
        'conflict_resolution': conflict_analyzer
    }
)
```

### Generalization Testing

```python
from embodied_ai_benchmark.evaluation import GeneralizationTest

gen_test = GeneralizationTest()

# Test across variations
variations = gen_test.create_test_variations(
    base_task="table_assembly",
    vary_along={
        'table_types': ['coffee', 'dining', 'desk'],
        'materials': ['wood', 'metal', 'glass'],
        'tools_available': ['all', 'limited', 'improvised'],
        'lighting': ['bright', 'normal', 'dim'],
        'distractors': [0, 5, 20]
    }
)

# Evaluate generalization
gen_results = gen_test.evaluate_generalization(
    agent=trained_agent,
    variations=variations,
    metrics=['success_rate', 'adaptation_speed', 'robustness']
)

# Generate generalization report
gen_test.plot_generalization_matrix(
    gen_results,
    save_path='generalization_report.pdf'
)
```

## Physics and Realism

### Advanced Physics Integration

```python
from embodied_ai_benchmark.physics import PhysicsConfig

# Configure realistic physics
physics = PhysicsConfig(
    simulator="flex",  # For soft bodies
    substeps=10,
    time_step=0.001,  # 1ms
    gravity=9.81,
    material_properties={
        'wood': {'density': 700, 'friction': 0.6, 'restitution': 0.1},
        'metal': {'density': 7850, 'friction': 0.4, 'restitution': 0.05},
        'fabric': {'density': 100, 'stretch_stiffness': 0.1, 'bend_stiffness': 0.01}
    }
)

# Create environment with realistic physics
env = make_env(
    "RealisticManipulation-v0",
    physics_config=physics,
    features={
        'soft_body_dynamics': True,
        'contact_forces': True,
        'friction_variation': True,
        'wear_and_tear': True
    }
)

# Add physics-based challenges
env.add_physics_challenge(
    'slippery_surface',
    affected_objects=['floor', 'table'],
    friction_multiplier=0.1
)

env.add_physics_challenge(
    'worn_tools',
    affected_objects=['screwdriver', 'wrench'],
    effectiveness_multiplier=0.7
)
```

### Sensor Realism

```python
from embodied_ai_benchmark.sensors import RealisticSensors

# Configure realistic sensor models
sensors = RealisticSensors(
    camera={
        'resolution': (640, 480),
        'fov': 90,
        'noise_model': 'gaussian',
        'noise_level': 0.01,
        'motion_blur': True,
        'lens_distortion': True
    },
    depth={
        'resolution': (320, 240),
        'range': (0.1, 10.0),
        'noise_model': 'kinect_v2',
        'missing_data_rate': 0.05
    },
    tactile={
        'resolution': 16,  # 4x4 taxels
        'force_range': (0, 100),  # N
        'noise_level': 0.1
    }
)

# Apply sensor models to environment
env.set_sensor_config(sensors)
```

## Benchmark Scenarios

### Household Tasks

```python
from embodied_ai_benchmark.scenarios import HouseholdScenarios

household = HouseholdScenarios()

# Kitchen tasks
kitchen_benchmark = household.create_benchmark(
    'kitchen_activities',
    tasks=[
        'prepare_simple_meal',
        'load_dishwasher',
        'organize_pantry',
        'clean_countertops'
    ],
    difficulty_progression=True,
    time_limits={'easy': 600, 'medium': 300, 'hard': 180}
)

# Cleaning tasks
cleaning_benchmark = household.create_benchmark(
    'whole_house_cleaning',
    tasks=[
        'vacuum_carpets',
        'mop_floors',
        'dust_furniture',
        'organize_clutter'
    ],
    multi_room=True,
    obstacles='dynamic'  # People, pets moving around
)
```

### Industrial Tasks

```python
from embodied_ai_benchmark.scenarios import IndustrialScenarios

industrial = IndustrialScenarios()

# Assembly line
assembly_benchmark = industrial.create_benchmark(
    'production_line',
    tasks=[
        'part_inspection',
        'component_assembly',
        'quality_control',
        'packaging'
    ],
    timing_constraints=True,
    error_tolerance=0.001  # 0.1% error rate
)

# Warehouse logistics
warehouse_benchmark = industrial.create_benchmark(
    'warehouse_operations',
    tasks=[
        'inventory_sorting',
        'order_picking',
        'pallet_stacking',
        'truck_loading'
    ],
    num_agents=4,
    coordination_required=True
)
```

### Emergency Response

```python
from embodied_ai_benchmark.scenarios import EmergencyScenarios

emergency = EmergencyScenarios()

# Search and rescue
sar_benchmark = emergency.create_benchmark(
    'urban_search_rescue',
    environment={
        'type': 'collapsed_building',
        'area': (50, 50, 10),  # meters
        'debris_density': 'high',
        'stability': 'unstable'
    },
    objectives=[
        'locate_survivors',
        'assess_medical_needs',
        'clear_access_paths',
        'evacuate_safely'
    ],
    num_agents=6,
    time_critical=True
)

# Fire response
fire_benchmark = emergency.create_benchmark(
    'structure_fire_response',
    hazards={
        'fire': {'spreading': True, 'intensity': 'variable'},
        'smoke': {'density': 'thick', 'toxic': True},
        'structural': {'collapse_risk': 0.3}
    },
    equipment=['hose', 'ladder', 'thermal_camera', 'breathing_apparatus'],
    coordination_protocol='incident_command'
)
```

## Multi-Agent Coordination

### Communication Protocols

```python
from embodied_ai_benchmark.multiagent import CommunicationProtocol

# Define communication protocol
protocol = CommunicationProtocol(
    message_types=['request', 'inform', 'confirm', 'coordinate'],
    bandwidth_limit=100,  # messages per second
    latency=0.1,  # seconds
    packet_loss=0.01  # 1% loss rate
)

# Structured communication
@protocol.message_handler('coordinate')
def handle_coordination_message(sender, receiver, content):
    if content['type'] == 'lift_sync':
        # Synchronize lifting action
        return {
            'ready': receiver.check_ready_state(),
            'position': receiver.get_gripper_position(),
            'force': receiver.get_applied_force()
        }

# Emergent communication
emergent_comm = protocol.create_emergent_channel(
    vocabulary_size=50,
    message_length=10,
    learned=True
)
```

### Role Assignment

```python
from embodied_ai_benchmark.multiagent import DynamicRoleAssignment

role_assigner = DynamicRoleAssignment(
    roles=['leader', 'supporter', 'scout', 'specialist'],
    assignment_method='capability_based'
)

# Dynamic role switching
@role_assigner.on_task_change
def reassign_roles(task, agents, current_performance):
    # Analyze task requirements
    requirements = analyze_task_requirements(task)
    
    # Match agents to roles
    new_assignment = role_assigner.optimize_assignment(
        agents=agents,
        requirements=requirements,
        constraints={
            'max_role_switches': 2,
            'maintain_coordination': True
        }
    )
    
    return new_assignment

# Hierarchical organization
hierarchy = role_assigner.create_hierarchy(
    structure='dynamic_tree',
    max_depth=3,
    span_of_control=4
)
```

## Training Integration

### Reinforcement Learning

```python
from embodied_ai_benchmark.training import RLTrainer

# Multi-task RL training
trainer = RLTrainer(
    algorithm='PPO',
    multi_task=True,
    task_sampling='curriculum'
)

# Train on benchmark suite
training_tasks = benchmark.get_training_tasks(
    difficulty_range=(0.1, 0.8),
    num_tasks=1000
)

trainer.train(
    tasks=training_tasks,
    eval_tasks=benchmark.get_eval_tasks(),
    total_steps=10_000_000,
    save_interval=100_000
)
```

### Imitation Learning

```python
from embodied_ai_benchmark.training import ImitationLearning

# Collect expert demonstrations
demonstrator = ExpertDemonstrator()
demos = demonstrator.collect_demonstrations(
    tasks=benchmark.get_all_tasks()[:100],
    num_demos_per_task=10
)

# Train with behavioral cloning
il_trainer = ImitationLearning(
    method='dagger',  # Dataset Aggregation
    expert=demonstrator,
    student=your_agent
)

il_trainer.train(
    demos=demos,
    interactive_rounds=20,
    queries_per_round=100
)
```

## Visualization and Analysis

### Performance Visualization

```python
from embodied_ai_benchmark.visualization import BenchmarkVisualizer

viz = BenchmarkVisualizer()

# Create comprehensive report
viz.generate_report(
    results=benchmark_results,
    include=[
        'task_success_matrix',
        'learning_curves',
        'failure_analysis',
        'efficiency_breakdown',
        'generalization_tests'
    ],
    format='interactive_html'
)

# Real-time monitoring dashboard
dashboard = viz.create_dashboard(
    metrics=['success_rate', 'avg_reward', 'collision_rate'],
    update_interval=1.0
)

dashboard.launch(port=8080)
```

### Behavior Analysis

```python
from embodied_ai_benchmark.analysis import BehaviorAnalyzer

analyzer = BehaviorAnalyzer()

# Analyze learned strategies
strategies = analyzer.extract_strategies(
    agent=trained_agent,
    tasks=benchmark_tasks,
    num_episodes=100
)

# Clustering similar behaviors
behavior_clusters = analyzer.cluster_behaviors(
    strategies,
    method='trajectory_similarity',
    num_clusters=10
)

# Generate behavior taxonomy
analyzer.create_behavior_taxonomy(
    behavior_clusters,
    save_path='behavior_taxonomy.pdf'
)
```

## Competition and Leaderboards

### Online Evaluation

```python
from embodied_ai_benchmark.competition import CompetitionServer

# Set up competition server
server = CompetitionServer(
    benchmark_suite=benchmark,
    evaluation_budget=1000,  # episodes
    time_limit=24  # hours
)

# Submit agent for evaluation
submission = server.submit_agent(
    agent=your_agent,
    team_name="YourTeam",
    agent_description="Multi-modal transformer policy with..."
)

# Track progress
while not submission.is_complete():
    status = submission.get_status()
    print(f"Progress: {status['episodes_completed']}/{status['total_episodes']}")
    print(f"Current score: {status['score']:.3f}")
    time.sleep(60)

# Get final results
results = submission.get_results()
ranking = server.get_leaderboard()
```

### Standardized Protocols

```python
from embodied_ai_benchmark.standards import EvaluationProtocol

# Standard evaluation protocol
protocol = EvaluationProtocol.load('embodied_ai_2025_v1')

# Ensure compliance
protocol.validate_environment(env)
protocol.validate_agent(agent)

# Run standardized evaluation
standard_results = protocol.evaluate(
    agent=agent,
    random_seeds=[42, 123, 456, 789, 1011],
    record_trajectories=True
)

# Generate certified results
certificate = protocol.generate_certificate(
    results=standard_results,
    timestamp=datetime.now(),
    verifier='embodied_ai_foundation'
)
```

## Configuration

### Benchmark Configuration

```yaml
# config/benchmark_config.yaml
benchmark:
  name: "Embodied-AI-Benchmark++"
  version: "1.0.0"
  
  simulators:
    habitat:
      version: "0.3.0"
      settings:
        physics_engine: "bullet"
        render_quality: "high"
    maniskill:
      version: "0.5.0"
      settings:
        contact_solver: "tgs"
        substeps: 10
        
  tasks:
    single_agent:
      - manipulation: ["pick_place", "tool_use", "assembly"]
      - navigation: ["point_goal", "object_goal", "exploration"]
      - mobile_manipulation: ["fetch", "rearrange", "clean"]
      
    multi_agent:
      - cooperative: ["furniture_assembly", "construction", "cleaning"]
      - competitive: ["resource_gathering", "territory_control"]
      - mixed: ["rescue_mission", "warehouse_logistics"]
      
  evaluation:
    metrics: ["success", "efficiency", "safety", "generalization"]
    trials_per_task: 100
    time_limit: 300  # seconds
    
  difficulty:
    progression: "adaptive"
    min_success_rate: 0.3
    max_success_rate: 0.8
```

### Agent Requirements

```yaml
# config/agent_requirements.yaml
agent_interface:
  observations:
    required: ["rgb", "depth", "proprioception"]
    optional: ["semantic", "tactile", "audio"]
    
  actions:
    continuous:
      navigation: ["linear_velocity", "angular_velocity"]
      manipulation: ["joint_positions", "gripper_force"]
    discrete:
      skills: ["pick", "place", "push", "pull"]
      
  communication:
    max_message_length: 128
    vocabulary_size: 1000
    
  compute_limits:
    max_inference_time: 100  # ms
    max_memory: "4GB"
    max_params: "1B"
```

## Extensibility

### Adding Custom Tasks

```python
from embodied_ai_benchmark.tasks import TaskBuilder

@TaskBuilder.register("custom_assembly")
class CustomAssemblyTask:
    def __init__(self, difficulty='medium'):
        self.difficulty = difficulty
        self.components = self.generate_components()
        self.target_structure = self.generate_target()
        
    def reset(self):
        # Initialize task state
        return self.get_observation()
        
    def step(self, action):
        # Execute action and compute reward
        self.execute_action(action)
        reward = self.compute_reward()
        done = self.check_completion()
        return self.get_observation(), reward, done, {}
        
    def compute_reward(self):
        # Custom reward logic
        alignment_score = self.compute_alignment()
        stability_score = self.compute_stability()
        return alignment_score + stability_score

# Register with benchmark
benchmark.add_task("custom_assembly", CustomAssemblyTask)
```

### Custom Metrics

```python
from embodied_ai_benchmark.metrics import MetricBuilder

@MetricBuilder.register("energy_efficiency")
class EnergyEfficiencyMetric:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.total_energy = 0
        self.task_completed = False
        
    def update(self, action, state):
        # Track energy consumption
        self.total_energy += self.compute_energy(action)
        self.task_completed = state.is_goal_reached
        
    def compute(self):
        if not self.task_completed:
            return 0
        # Energy per unit work
        work_done = self.estimate_work()
        return work_done / self.total_energy

# Add to evaluation
benchmark.add_metric("energy_efficiency", EnergyEfficiencyMetric)
```

## Troubleshooting

### Common Issues

```python
from embodied_ai_benchmark.diagnostics import DiagnosticTool

diag = DiagnosticTool()

# Check environment setup
env_check = diag.check_environment(env)
if not env_check.passed:
    print("Environment issues:")
    for issue in env_check.issues:
        print(f"- {issue.description}: {issue.solution}")

# Validate agent interface
agent_check = diag.check_agent_compatibility(agent, env)
if not agent_check.compatible:
    print("Agent compatibility issues:")
    diag.suggest_interface_adapter(agent, env)

# Performance profiling
profile = diag.profile_performance(
    agent, env,
    num_steps=1000,
    metrics=['inference_time', 'memory_usage', 'action_frequency']
)

if profile.inference_time > 100:  # ms
    print("Agent too slow for real-time control")
    diag.suggest_optimizations(agent, target_latency=50)
```

## Citation

```bibtex
@article{embodied_ai_benchmark_plus,
  title={Embodied-AI Benchmark++: Multi-Agent Tasks and LLM-Guided Curriculum},
  author={Your Name},
  journal={Conference on Robot Learning},
  year={2025}
}
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Habitat and ManiSkill teams for simulation platforms
- Embodied AI research community
- Contributors to multi-agent benchmarks

## Resources

- [Documentation](https://embodied-ai-benchmark.readthedocs.io)
- [Task Videos](https://embodied-ai-benchmark.github.io/videos)
- [Leaderboard](https://embodied-ai-benchmark.github.io/leaderboard)
- [Dataset Download](https://embodied-ai-benchmark.github.io/download)
