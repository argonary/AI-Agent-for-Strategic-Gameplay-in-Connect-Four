# Introduction
This project investigates the use of deep learning techniques to build an artificial intelligence (AI) agent capable of playing the board game Connect Four at a high level. Our goal was to create a model that, when presented with any board state, can predict the optimal move. To accomplish this, we explored three different neural network architectures:

1. A Convolutional Neural Network (CNN)
2. A Transformer-based model
3. A Hybrid model combining CNN and Transformer components

Each model was evaluated on its ability to accurately predict the best move using training data generated through simulation. In addition to evaluating accuracy on validation datasets, we also tested our models in real gameplay scenarios to better understand their practical performance.

# Data Generation
Generating high-quality, representative data was critical to the success of this project. Our primary method for data generation was Monte Carlo Tree Search (MCTS). This technique is widely used in game-playing AI, including notable systems like AlphaGo, and is well-suited to games such as Connect Four.

## How MCTS Was Used
The MCTS algorithm worked by simulating many possible game continuations from a given board state. It selected the move that led to the highest win rate after numerous simulated games. Specifically, the process included:

- Randomly simulating the remaining game from a given state
- Evaluating which moves led to the most successful outcomes
- Building a tree of potential moves that increasingly prioritized successful paths
- Selecting the move with the highest average success rate as the "optimal" move

## Enhancements and Challenges
To improve the quality of the data, we increased the number of simulation steps used in each MCTS rollout, extending the range from 1500–2500 to 1500–3000 steps. We also introduced random moves at the start of each game to diversify early board positions.

To augment our dataset, we mirrored boards horizontally and removed duplicate entries. However, inconsistencies arose due to team members using slightly different versions of the data generation script. These inconsistencies, such as differing board encodings or move formats, introduced some noise. Despite this, we assembled a dataset of approximately two million unique samples.

# Convolutional Neural Network (CNN) Model
The CNN model was designed to take advantage of the grid-like structure of the Connect Four board. Since the game board is essentially a 6×7 matrix, CNNs are well-suited to recognize spatial patterns such as horizontal, vertical, and diagonal alignments.

## Architecture Overview
Our CNN architecture was built using Keras and consisted of several key components:
- **Convolutional Layers:** Used various kernel sizes (4×4, 3×3, 2×2) to capture patterns of different sizes and orientations.
- **Batch Normalization:** Stabilized the training process and improved convergence.
- **Max Pooling:** Reduced spatial dimensions while preserving important features.
- **Dense Layers:** Translated the extracted features into a decision over which column to play.
- **Dropout:** Prevented overfitting by randomly deactivating neurons during training.

## Training Details
- Data was split into 80% training and 20% validation.
- Labels (move columns) were one-hot encoded.
- The model was compiled using the Adam optimizer and categorical cross-entropy loss.
- Early stopping and learning rate reduction callbacks were used to prevent overfitting and improve training efficiency.

## Results
The CNN model achieved:
- 77.34% training accuracy
- 68.7% validation accuracy

It outperformed the Transformer-based models, likely due to its effectiveness in recognizing the localized spatial patterns that dominate Connect Four strategy.

# Transformer Model
Transformers, widely known for their success in natural language processing and sequence modeling tasks, were adapted for this project using techniques similar to Vision Transformers (ViTs). The idea was to capture more global relationships on the board and reason across multiple patches of the board state.

## Architecture Overview
The final Transformer model included:
- **Overlapping Patch Extraction:** Divided the board into overlapping 3×4 patches to capture wider patterns.
- **Tokenization and Positional Embeddings:** Encoded spatial information and provided a global summary through a CLS token.
- **Multi-Head Self-Attention and Feedforward Layers:** Enabled the model to combine features from different board regions effectively.

The model had over 21 million total parameters, with a substantial portion dedicated to the optimizer state.

## Results
The final Transformer model achieved:
- 67.29% accuracy on new (unseen) data
- Performance gains were observed in early and mid-phase training, but plateaued at around 50–67% accuracy

Although it showed promise, the model did not outperform the CNN, likely due to the limited size of the Connect Four board, which reduced the need for long-range dependency modeling.

# CNN/Transformer Hybrid Model
Motivated by the strengths of both CNNs and Transformers, we experimented with a hybrid architecture that aimed to combine their benefits.

## Design Highlights
- **CNN Front-End:** Used multiple convolutional layers to extract rich spatial features before feeding them into the Transformer layers.
- **Transformer Back-End:** Processed the output from the CNN using multi-head attention to capture global board context.
- **Warmup + Cosine Decay Learning Rate Schedule:** Applied to ensure smooth and efficient convergence.
- **Hyperparameter Tuning:** Conducted using Keras Tuner (Hyperband) to identify optimal architecture and training parameters.

## Results
The hybrid model achieved:
- 67.90% accuracy on unseen validation data

Although the hybrid approach performed comparably to the Transformer-only model, it did not outperform the simpler CNN model. The added complexity may not have been necessary for a relatively small and structured problem like Connect Four.

# Model Comparison and Gameplay Evaluation
To evaluate the real-world effectiveness of each model, we played games against them.

## Observations
- Models performed significantly better when making the first move, reflecting the natural first-move advantage in Connect Four.
- Occasionally, the models missed obvious win/loss scenarios, such as failing to block an opponent’s four-in-a-row or not recognizing a winning move.
- The CNN model consistently offered the best overall performance, both in validation accuracy and in live gameplay.

# Key Takeaways
1. CNNs are well-suited for grid-based games like Connect Four due to their ability to detect localized spatial features.
2. Transformers showed potential, especially with overlapping patch extraction and careful training, but may be overkill for such a small board.
3. The hybrid model was an insightful experiment, but the increased complexity did not yield superior performance.
4. Data quality and consistency are crucial for training reliable AI models.

# Future Work
Several improvements could enhance this project’s outcomes:
- Endgame Logic Integration: Implement a rule-based system to catch immediate win/loss conditions that the models may overlook.
- Player-Specific Models: Train separate models for playing first vs. second to better adapt strategies.
- Reinforcement Learning: Explore self-play and reinforcement learning methods, similar to AlphaZero, to evolve strategies over time.
- Stricter Data Standardization: Ensure all team members use identical formats during data generation to reduce noise and improve consistency.
