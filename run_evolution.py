# matsushibadenki/snn3/run_evolution.py
# Script to run the self-evolving agent and execute a meta-evolution cycle.

import argparse
from snn_research.agent.self_evolving_agent import SelfEvolvingAgent

def main():
    """
    Starts the process of giving an initial task to the self-evolving agent,
    making it reflect on its performance, and generate improvement proposals.
    """
    parser = argparse.ArgumentParser(
        description="SNN Self-Evolving Agent Execution Framework",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--task_description",
        type=str,
        required=True,
        help="The task for the agent to start its self-assessment.\nExample: 'Text Summarization'"
    )
    
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/models/small.yaml",
        help="The model architecture configuration file targeted for self-evolution."
    )

    # Dummy initial performance metrics as a starting point for self-assessment
    parser.add_argument(
        "--initial_accuracy", type=float, default=0.75, help="Initial accuracy for the task"
    )
    parser.add_argument(
        "--initial_spikes", type=float, default=1500.0, help="Initial average spikes for the task"
    )
    
    args = parser.parse_args()

    # Initialize the self-evolving agent (specify project root and model config)
    agent = SelfEvolvingAgent(project_root=".", model_config_path=args.model_config)

    # Create dummy initial metrics
    initial_metrics = {
        "accuracy": args.initial_accuracy,
        "avg_spikes_per_sample": args.initial_spikes
    }
    
    # Execute the evolution cycle
    agent.run_evolution_cycle(
        task_description=args.task_description,
        initial_metrics=initial_metrics
    )

if __name__ == "__main__":
    main()

