# Advanced Biologically-Inspired AI System

This project implements a cutting-edge, biologically-inspired AI system that combines advanced neural plasticity, reinforcement learning, language reasoning, and hierarchical task structures. It aims to achieve human-like performance on complex, multi-task learning benchmarks while demonstrating adaptive learning in dynamic environments.

## Key Features

1. **Advanced Neuroplasticity**: Implements sophisticated neuroplasticity rules based on recent neuroscience findings, including:
   - Astrocyte modulation
   - Dendritic computation
   - Spike-timing-dependent plasticity (STDP)
   - Synaptic tagging and capture
   - Homeostatic plasticity

2. **Reinforcement Learning**: Integrates an advanced Actor-Critic reinforcement learning agent for improved decision-making.

3. **Language Reasoning**: Utilizes GPT-2 for complex reasoning and abstraction, enabling the system to generate explanations and incorporate language understanding into its decision-making process.

4. **Hierarchical Task Structure**: Implements a two-level hierarchical agent that can handle different levels of cognitive processes, from low-level actions to high-level goal setting.

5. **Hybrid Learning System**: Combines all the above components into a unified learning system that can adapt to various tasks and environments.



## Project Structure

- `src/`: Contains the core implementation of all components
  - `neural_plasticity.py`: Advanced neuroplasticity models
  - `reinforcement_learning.py`: RL agent implementation
  - `language_reasoning.py`: GPT-2 based language reasoning module
  - `hierarchical_agent.py`: Hierarchical task structure implementation
  - `hybrid_learning_system.py`: Integration of all components
- `experiments/`: Contains scripts for running experiments
- `requirements.txt`: List of required Python packages
- `README.md`: Project documentation

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - On Unix or MacOS: `source venv/bin/activate`
   - On Windows: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`

## Usage

Run the main experiment script:

python experiments/run_experiment.py

This will initialize the hybrid learning system and run a series of experiments to demonstrate its capabilities across various tasks and environments.
