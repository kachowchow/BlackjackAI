# 🃏 **AI Blackjack Trainer: Reinforcement Learning via Q-Learning**

A high-performance **Reinforcement Learning (RL)** project featuring a **Q-Learning** agent that learns to play Blackjack from scratch. Built with **Python** and **Pygame**, this project demonstrates how an agent can transition from total randomness to a mathematically optimal strategy through trial and error.
---

## 🚀 **Key Features**
* **Self-Learning Agent**: Implements the **Bellman Equation** to update state-action values ($Q$-values).
* **Dynamic Visualization**: Generates real-time **Heatmaps** using `Seaborn` and `Matplotlib` to visualize the AI's "Brain."
* **Epsilon-Greedy Strategy**: Features an adjustable **Exploration vs. Exploitation** balance with a decaying epsilon rate.
* **Headless Fast-Training**: Optimized loop architecture that bypasses rendering to simulate thousands of episodes per second.
* **Custom Game Engine**: A robust Blackjack environment built in **Pygame** with automated dealer logic and deck management.
---

## 🧠 **The Machine Learning Logic**
The agent views the game as a **State** consisting of:
1.  **Player’s Total** (4–21)
2.  **Dealer’s Visible Card** (2–Ace)
3.  **Usable Ace** (True/False)

### **The Reward System**
The AI's behavior is shaped by a simple feedback loop:
* **Win**: `+1.0` Reward
* **Loss / Bust**: `-1.0` Reward
* **Draw (Push)**: `0.0` Reward
---

## 📊 **Policy Visualization**
By pressing the **'V'** key, you can view the AI's learned policy. This heatmap shows the most profitable move for every possible hand combination.
* **Green (1)**: The AI has learned to **HIT**.
* **Red (0)**: The AI has learned to **STAND**.
---

## 🛠️ **Installation & Usage**

### **1. Prerequisites**
Ensure you have the following libraries installed:
```bash
pip install pygame numpy pandas seaborn matplotlib