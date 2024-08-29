## c4crow
Hi there, welcome! The end goal of this repo is to replicate AlphaZero (parallel self-play RL training w/ MCTS variation using policy and value nets) for connect 4. On the way to that goal, I'll thoroughly learn and implement other game-playing algorithms, both non-rl and rl-based.

Currently implemented algorithms:

1. Minimax (minimax, will add alpha-beta pruning soon)
2. MCTS (Monte Carlo Tree Search)
3. Q* (TD learning to approx DQN, need to fix, vs random player, no self-play)
4. PolicyNet (REINFORCE, need to fix, vs random player, no self-play)

TODO:

5. A2C
6. PPO
7. AlphaZero

---
### Download and Run
```
git clone https://github.com/calvinh99/c4crow.git
cd c4crow
pip install -r requirements.txt
pip install -e .
python src/c4crow/play.py --player1 human --player2 minimax
```

---
### UI
The current UI is terminal-based, the GUI is coming soon. I'd like to highlight a nice feature I added recently - intuitive visualization into the player's "brain". Here's what it looks like for minimax:

<img src="https://github.com/calvinh99/c4crow/blob/main/resources/readme_minimax_1.png" width="300">

I am playing as yellow O here and minimax is red X. You can see that I am trying to play a sure-win move in connect 4 where if you get 3 in a row horizontally and the opponent does not block the left or right then you are guaranteed to win.

<img src="https://github.com/calvinh99/c4crow/blob/main/resources/readme_minimax_2.png" width="320">

Minimax player was able to detect that, as we can see from the scores calculated, it figured out that playing anywhere other than column 2 or column 5 would result in a loss.

Personally this visualization has helped me find and fix bugs in my implementations and just overall understand the algorithms better. This is also why I'm committed to making the GUI soon.

My current design of the GUI:

<img src="https://github.com/calvinh99/c4crow/blob/main/resources/GUI_design.png" width="550">

---
### Additional Notes
You can put agent vs agent with a simple command:
```
python src/c4crow/play.py --player1 minimax --player2 mcts
```
