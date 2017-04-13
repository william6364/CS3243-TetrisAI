This is a basic Tetris AI simulation.

Files:
	PlayerSkeleton - setup for implementing a player
	JNEAT Network Converter - converts the file output of JNEAT into code that can be copied into the Network constructor directly
	State - tetris simulation
	TFrame - frame that draws the board
	TLabel - drawing library

PlayerSkeleton:
The main function plays a game automatically (with visualization).
The AI implemented uses a neural network to evaluate every possible move with a score and select the best move at each stage.
The neural network is implemented in the Network class, which consists of Nodes and Links.
The optimal network was learnt using JNEAT and hard-coded into the constructor class of the Network for the purposes of this project.
	
State:
This is the tetris simulation.  It keeps track of the state and allows you to make moves.
The board state is stored in field (a double array of integers) and is accessed by getField().
Zeros denote an empty square.  Other values denote the turn on which that square was placed.
NextPiece (accessed by getNextPiece) contains the ID (0-6) of the piece you are about to play.

Moves are defined by two numbers: the SLOT, the leftmost column of the piece and the ORIENT, the orientation of the piece.
legalMoves gives an nx2 int array containing the n legal moves.
A move can be made by specifying the two parameters as either 2 ints, an int array of length 2, or a single int specifying the row in the legalMoves array corresponding to the appropriate move.

It also keeps track of the number of lines cleared - accessed by getRowsCleared().

draw() draws the board.
drawNext() draws the next piece above the board
clearNext() clears the drawing of the next piece so it can be drawn in a different
	slot/orientation

TFrame:
This extends JFrame and is instantiated to draw a state.
It can save the current drawing to a .png file.
The main function allows you to play a game manually using the arrow keys.

TLabel:
This is a drawing library.
