#AI Blackjack Trainer: Reinforcement Learning via Q-Learning
A Reinforcement Learning (RL) project that uses a Q-Learning agent to master the game of Blackjack. The project features a custom-built environment in Pygame, a "headless" fast-training mode, and real-time visualization of the AI's decision-making policy through heatmaps.

🚀 Features
Q-Learning Agent: Implements the Bellman Equation to update state-action values.

Dynamic Visualization: Generates heatmaps using Seaborn and Matplotlib to show the AI's learned strategy (Hit vs. Stand).

Epsilon-Greedy Strategy: Balances exploration and exploitation, with a decaying epsilon rate to transition from random guessing to optimal play.

Fast Training Mode: Bypasses graphical rendering to simulate thousands of hands per second for rapid learning.

Custom Pygame UI: A fully interactive Blackjack table that displays real-time statistics, episode counts, and AI "randomness" levels.

🛠️ Technical Stack
Language: Python 3.x

Graphics: Pygame

Data Science: NumPy, Pandas

Visualization: Matplotlib, Seaborn

Algorithm: Reinforcement Learning (Q-Learning)

🧠 How It Works
The AI perceives the game as a State consisting of three variables:

Player’s Current Total (4–21)

Dealer’s Visible Card (2–Ace)

Usable Ace (True/False)

The agent receives a Reward of +1.0 for a win, -1.0 for a loss or bust, and 0.0 for a draw. Over thousands of episodes, the agent updates its Q-Table, eventually recreating the mathematically optimal "Basic Strategy" for Blackjack.

📊 Performance Visualization
The project includes a policy visualizer. By pressing V during execution, the program generates a heatmap:

Green (1): The AI has learned to HIT.

Red (0): The AI has learned to STAND.

🎮 Controls
Automatic Play: The AI plays hands automatically at high speed.

V Key: Toggle the Policy Heatmap visualization.

ESC/Close: Exit the trainer.