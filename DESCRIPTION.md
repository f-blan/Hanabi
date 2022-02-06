# Project Description

This is a simple description of the project highligthing the main characteristics.
This project was developed and submitted exclusively by Francesco Blangiardi, s288265

## Usage

<br>To use the project first run the server.py script, then run the cpu_client.py script to instantiate a cpu player 
<br>that plays according to the solutions proposed in this project. After running the script, you will be asked to
<br>enter any character to signal to the server that the player is ready to start.<br>
<br>The parameter of the cpu_client.py script are the same of the client.py script for instantiating a human player
<br>that can play the game by interacting through the terminal. One additional parameter can be specified to cpu_client
<br>to specify which solution to use (can be "NSolver", "FSolver" or "MCSolver"). The default solver used are FSolver
<br>when there are less than 4 players, MCSolver otherwise

## NSolver
<br>This is the simplest solution developed and is not meant to be used directly (altough it works). This can
<br>be described as a "fixed policy" player: it checks if it can perform a set of move types in a fixed order
<br>(i.e. "safe play","safe discard", "safe hint", "random hint" etc.); then, the first match in the given order
<br>is performed in the real game. The ordering is done so that the player follows a very "safe" strategy 
<br>so it will never play an unknown card an will discard unknown cards only if there are no blue tokens available

## FSolver
<br>This solver is meant to compute the most "meaningful" move the agent can play at each turn without looking at
<br>how future turns will play out. <br>
<br>In brief, it keeps track of what the deck and the cards still in the game are throughout the game, and by
<br>computing statistics on them and the rest of the game state (fireworks, tokens) it evaluates each card in the 
<br>hands of each player by computing their "Playability" and "Discardability" in a Fuzzy fashion. For details on how
<br>this is done refer to the class FDeck (methods evaluate_known_cards and evaluate_unknown_cards).
<br>After the evaluation of cards, a set of possible moves are computed for the agent, and each of these moves
<br>are assigned a further evaluation based on the playability and/or discardability of the related cards and on
<br>a small amount of domain knowledge (for details on how the possible moves are computed see class FPlayer method 
<br>GetMoves; for details on move evaluation see class FMove methods EvaluateX).<br>
<br>Finally, the move with the highest score is selected for play. In brief, the FSolver defines a somewhat safe
<br>strategy (it can do unsafe plays but will never risk losing the gmae) and should be able to play reasonably
<br>good hints and to find good plays/discards even when partial knowledge is given through hints

## MCSolver
<br>This solver adds to the FSolver some consideration on how the turns will play out. As the name suggests this is
<br>done through a MonteCarlo Tree Search algorithm.<br>
<br>In detail, we apply to each MCTS node an FSolver strategy to select the top N moves available in that node, so that
<br>the related child nodes can be computed during the Expansion Step of said node.
<br>The best move to be used in the real game is found in a standard way through the MCTS algorithm:
 - We give the algorithm a certain amount of iterations to expand the tree (MCNode.FindMove())
 - To each iteration corresponds the expansion of one node (MCNode.MC_expand)
 - At each iteration, we select the node to be expanded according to a selection procedure (MCNode.MC_eelect()) based on UTC
 - Node expansion implies computation of possible moves and creation of a child node for each of them
 - Whenever a node is generated, it computes its initial score with the Simulation Step (MCNode.MC_eimulate())
 - The expansion of the parent node ends with the BackPropagation step (MCNode.backprop) of its newly generated children
 - After all iterations are performed, the move to be played is chosen according to the highest mean score of all its children

 <br>For details refer to the functions related to each step. Simulation is not performed 
 <br>through a playout but through an ad hoc, fast evaluation function that takes into account the game state at the
 <br>given node. An alternative, probably more performing approach would be to perform one or more playouts following
 <br>a given policy (i.e. NSolver), but this idea didn't reach the final project due to time constraints.<br>

 <br>Moreover, the random nature of the game of Hanabi is also taken into account in the MCTS algorithm. For details on
 <br>which moves are allowed within the several nodes refer to MCNode.apply_move(), while for details on how draws and
 <br>moves related to unknown or partially known cards refer to functions MCPlayer and NodeDeck 
