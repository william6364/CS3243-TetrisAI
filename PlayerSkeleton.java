import java.util.*;

public class PlayerSkeleton {
    // Neural network of the AI
    private Network network;

    // Number of inputs to the neural network
    private static final int numInputs = 9;

    // Array of inputs to be passed into the neural network
    private double[] inputs = new double[numInputs];

    // Row at which game is lost
    private static final int LOSING_ROW = State.ROWS - 1;

    // Array to store the board state as an array of integers
    private int[] rows = new int[LOSING_ROW];

    // Array to store original rows array for resetting purposes after each possible move
    private int[] currentRows = new int[LOSING_ROW];

    // Array to store height of each column
    private int[] top = new int[State.COLS];

    // Array to store original top array for resetting purposes after each possible move
    private int[] currentTop = new int[State.COLS];

    // ArrayList to store the coordinates taken up by the new piece
    private ArrayList<Integer> filledCoordinates = new ArrayList<>();

    // Tile specific feature values
    private int wallHuggingCoefficient = 0;
    private int floorHuggingCoefficient = 0;
    private int flatteningCoefficient = 0;

    // Constructor
    public PlayerSkeleton() {
        // Initialise the neural network
        network = new Network();

        // Reset the game
        resetGame();
    }

    // Perform the move by placing piece of type pieceIndex with orientation rotationIndex and left column at leftPosition
    private int performMove(int pieceIndex, int rotationIndex, int leftPosition) {
        // Reset tile specific feature values
        wallHuggingCoefficient = 0;
        floorHuggingCoefficient = 0;
        flatteningCoefficient = 0;

        // Height if the first column makes contact
        int height = top[leftPosition] - State.getpBottom()[pieceIndex][rotationIndex][0];

        // For each column beyond the first in the piece
        for (int c = 0; c < State.getpWidth()[pieceIndex][rotationIndex]; c++) {
            // Height if this column makes contact
            height = Math.max(height, top[leftPosition + c] - State.getpBottom()[pieceIndex][rotationIndex][c]);

            // Floor-Hugging Coefficient
            if (top[leftPosition + c] == -1 && State.getpBottom()[pieceIndex][rotationIndex][c] == 0) {
                floorHuggingCoefficient++;
            }
        }

        // Check if game ends after performing this move
        if (height + State.getpHeight()[pieceIndex][rotationIndex] >= LOSING_ROW) {
            return -1;
        }

        // For each column in the piece - fill in the appropriate blocks
        filledCoordinates.clear();
        for (int i = 0; i < State.getpWidth()[pieceIndex][rotationIndex]; i++) {
            // From bottom to top of brick
            for (int h = height + State.getpBottom()[pieceIndex][rotationIndex][i] + 1; h <= height + State.getpTop()[pieceIndex][rotationIndex][i]; h++) {
                rows[h] |= (1 << (i + leftPosition)); // Fill the square in the board
                filledCoordinates.add(h * State.COLS + i + leftPosition); // Store the coordinates filled by the new piece by converting to integer
            }
        }

        // Wall-Hugging Coefficient
        // If piece hugs left wall
        if (leftPosition == 0) {
            wallHuggingCoefficient += State.getpTop()[pieceIndex][rotationIndex][0] - State.getpBottom()[pieceIndex][rotationIndex][0];
        }
        // If piece hugs right wall
        if (leftPosition + State.getpWidth()[pieceIndex][rotationIndex] - 1 == State.COLS - 1) {
            wallHuggingCoefficient += State.getpTop()[pieceIndex][rotationIndex][State.getpWidth()[pieceIndex][rotationIndex] - 1] -
                    State.getpBottom()[pieceIndex][rotationIndex][State.getpWidth()[pieceIndex][rotationIndex] - 1];
        }

        // Flattening Coefficient
        for (int coordinate : filledCoordinates) {
            // Find original coordinate from integer
            int row = coordinate / State.COLS;
            int col = coordinate % State.COLS;

            // Check if the squares next to each piece square were already filled
            int left = row * State.COLS + col - 1;
            int right = row * State.COLS + col + 1;
            int down = (row - 1) * State.COLS + col;
            // left side
            if (col != 0 && !filledCoordinates.contains(left) && ((rows[row] & (1 << (col - 1))) > 0)) {
                flatteningCoefficient++;
            }
            // right side
            if (col != State.COLS - 1 && !filledCoordinates.contains(right) && ((rows[row] & (1 << (col + 1))) > 0)) {
                flatteningCoefficient++;
            }
            // down side
            if (row != 0 && !filledCoordinates.contains(down) && ((rows[row - 1] & (1 << col)) > 0)) {
                flatteningCoefficient++;
            }
        }

        // Calculate new board after rows are cleared
        int rowsCleared = 0;
        for (int r = height + 1; r < LOSING_ROW; r++) {
            // If row is full
            if (rows[r] + 1 == (1 << State.COLS)) {
                rowsCleared++;
            }
            // Otherwise, shift row down by number of lines cleared
            else if (rowsCleared > 0) {
                rows[r - rowsCleared] = rows[r];
            }
        }

        // Reset top rows based on number of rows cleared
        for (int r = LOSING_ROW - 1; r >= LOSING_ROW - rowsCleared; r--) {
            rows[r] = 0;
        }

        // Reset top array then calculate new values of top array
        int hasBlocked = 0;
        for (int c = 0; c < State.COLS; c++) {
            top[c] = -1;
        }

        // Search downwards
        for (int r = LOSING_ROW - 1; r >= 0; r--) {
            // Find which columns are filled in this row that have not been filled before
            int topSquares = (rows[r] & ~hasBlocked);
            // Update top array for each column that is now filled but was not filled before
            while (topSquares > 0) {
                top[Integer.numberOfTrailingZeros(topSquares)] = r;
                topSquares ^= Integer.lowestOneBit(topSquares);
            }
            // Update which columns have already been filled
            hasBlocked |= rows[r];
        }
        return rowsCleared;
    }

    // Implement this function to have a working system
    public int pickMove(State s, int[][] legalMoves) {
        // Store best move and its score
        int bestMove = -1;
        double bestOutput = 0.0;
        //Save current board
        for (int i = 0; i < State.COLS; i++) {
            currentTop[i] = top[i];
        }
        for (int r = 0; r < LOSING_ROW; r++) {
            currentRows[r] = rows[r];
        }
        // Try all possible moves
        for (int i = 0; i < legalMoves.length; ++i) {
            // Perform each possible move
            int linesCleared = performMove(s.getNextPiece(), legalMoves[i][0], legalMoves[i][1]);
            if (linesCleared < 0) continue; // losing move, do not consider.

            // Reset feature values then proceed to calculate all the feature values
            for (int k = 0; k < numInputs; k++) inputs[k] = 0.0;

            // 0. 2 Power of Lines Cleared.
            inputs[0] = (double) (1 << linesCleared);
            for (int c = 0; c < State.COLS; c++) {
                // 1. Sum of Heights of each column
                inputs[1] += (double) ((top[c] + 1));

                // 2. Bumpiness
                if (c > 0) {
                    inputs[2] += (double) ((top[c] - top[c - 1]) * (top[c] - top[c - 1]));
                }
            }
            int hasBlocked = 0;
            for (int r = LOSING_ROW - 1; r >= 0; r--) {
                // 3. Sum of heights of each block
                inputs[3] += (double) ((r + 1) * Integer.bitCount(rows[r]));

                // 5. Number of Holes
                inputs[5] += (double) (Integer.bitCount(hasBlocked & ~rows[r]));
                hasBlocked |= rows[r];
            }
            int hasHoles = 0;
            for (int r = 0; r < LOSING_ROW; r++) {
                // 4. Blockades
                inputs[4] += (double) (Integer.bitCount(hasHoles & rows[r]));
                hasHoles |= (~rows[r]);
            }

            // Tile Specific Features
            inputs[6] = (double) (wallHuggingCoefficient);
            inputs[7] = (double) (floorHuggingCoefficient);
            inputs[8] = (double) (flatteningCoefficient);

            // Get evaluation of board from neural network and update if move is better or if it is first legal move
            double output = network.calculateOutputs(inputs).get(0).getActivation();
            if (bestMove == -1 || output > bestOutput) {
                bestMove = i;
                bestOutput = output;
            }

            // Reset board for simulation of next move
            for (int r = 0; r < LOSING_ROW; r++) {
                rows[r] = currentRows[r];
            }
            for (int c = 0; c < State.COLS; c++) {
                top[c] = currentTop[c];
            }
        }
        // If all moves are losing moves, play first move
        if (bestMove == -1) bestMove = 0;
        // Perform the best move
        performMove(s.getNextPiece(), legalMoves[bestMove][0], legalMoves[bestMove][1]);
        return bestMove;
    }

    // Reset the board
    private void resetGame() {
        for (int i = 0; i < State.COLS; i++) {
            top[i] = -1;
        }
        for (int i = 0; i < LOSING_ROW; i++) {
            rows[i] = 0;
        }
    }

    public static void main(String[] args) {
        State s = new State();
        new TFrame(s);
        PlayerSkeleton p = new PlayerSkeleton();
        while (!s.hasLost()) {
            s.makeMove(p.pickMove(s, s.legalMoves()));
            s.draw();
            s.drawNext(0, 0);
            try {
                Thread.sleep(300);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        System.out.println("You have completed " + s.getRowsCleared() + " rows.");
    }
}

//Neural network class which contains Nodes and Links between the Nodes
class Network {
    // Input nodes of the network
    ArrayList<Node> inputs = new ArrayList<>();

    // Output nodes of the network
    ArrayList<Node> outputs = new ArrayList<>();

    // All nodes of the network
    ArrayList<Node> allNodes = new ArrayList<>();

    // Maximum number of neurons between an output and an input, to know how many times to activate
    private int maxDepth = 0;

    // Network Constructor using optimal network
    public Network() {
        // Optimal neural network of nodes and links obtained from experiments
        allNodes.add(new Node(1, 1, 1));
        allNodes.add(new Node(2, 1, 1));
        allNodes.add(new Node(3, 1, 1));
        allNodes.add(new Node(4, 1, 1));
        allNodes.add(new Node(5, 1, 1));
        allNodes.add(new Node(6, 1, 1));
        allNodes.add(new Node(7, 1, 1));
        allNodes.add(new Node(8, 1, 1));
        allNodes.add(new Node(9, 1, 1));
        allNodes.add(new Node(13, 0, 2));
        allNodes.add(new Node(19, 0, 0));
        allNodes.add(new Node(33, 0, 0));
        allNodes.add(new Node(60, 0, 0));
        allNodes.add(new Node(65, 0, 0));
        new Link(1, 13, 0.010818541590347397, this);
        new Link(3, 13, -0.27490334814819495, this);
        new Link(5, 13, -0.2660072866644904, this);
        new Link(6, 13, -5.130228583802634, this);
        new Link(7, 13, 2.2011611814049403, this);
        new Link(8, 13, 2.0043225955353448, this);
        new Link(9, 13, 2.062152344710336, this);
        new Link(4, 19, -0.9545223177673527, this);
        new Link(19, 13, 0.4276937558254309, this);
        new Link(9, 19, 0.23710094177667254, this);
        new Link(5, 33, -2.9248016754531756, this);
        new Link(33, 13, 0.6585141220716934, this);
        new Link(1, 19, 0.13455651256123172, this);
        new Link(4, 60, 1.25317240673601, this);
        new Link(60, 13, -0.5736523386046739, this);
        new Link(2, 65, -1.648604179803141, this);
        new Link(65, 13, 0.415524937195853, this);
        new Link(3, 65, 0.34024434247505586, this);
        new Link(4, 65, -0.35572144197754824, this);

        // Classify nodes into input and output nodes
        for (Node _node : allNodes) {
            if (_node.gen_node_label == Node.INPUT || _node.gen_node_label == Node.BIAS)
                inputs.add(_node);
            if (_node.gen_node_label == Node.OUTPUT)
                outputs.add(_node);
        }

        // Calculate maximum depth
        for (Node _node : outputs) {
            maxDepth = Math.max(maxDepth, _node.depth(0, maxDepth));
        }
    }

    // Find outputs based on given inputs
    public ArrayList<Node> calculateOutputs(double[] sensorValues) {
        // Load in sensor values and calculate activation values for each neuron based on its input links
        int counter = 0;
        for (Node _node : inputs) {
            if (_node.type == Node.SENSOR)
                _node.sensor_load(sensorValues[counter++]);
        }

        // Repeat for number of layers in the neural network to ensure output is correct
        for (int relax = 0; relax <= maxDepth + 1; relax++) {
            // For each node, compute the sum of its incoming activation
            for (Node _node : allNodes) {
                if (_node.type != Node.SENSOR) {
                    _node.activesum = 0.0; // reset activation value
                    _node.active_flag = false; // flag node disabled
                    for (Link _link : _node.incoming) {
                        _node.activesum += _link.weight * _link.in_node.get_active_out();
                        if (_link.in_node.active_flag || _link.in_node.type == Node.SENSOR)
                            _node.active_flag = true;
                    }
                }
            }
            // Now activate all the non-sensor nodes off their incoming activation
            for (Node _node : allNodes) {
                if (_node.type != Node.SENSOR) {
                    //Only activate if some active input came in
                    if (_node.active_flag) {
                        _node.activation = (1 / (1 + (Math.exp(-(4.924273 * _node.activesum))))); //SIGMOID function
                        _node.activation_count += 1.0;
                    }
                }
            }
        }
        return outputs;
    }

}

// Node class, the nodes of the neural network
// Each node is either a NEURON or a SENSOR. If it's a sensor, it can be loaded with a value for output
// If it's a neuron, it has a list of its incoming input signals
// Use an activation count to avoid flushing
class Node {
    // NODE CONSTANTS//

    // gen_node_label
    public static final int HIDDEN = 0;
    public static final int INPUT = 1;
    public static final int OUTPUT = 2;
    public static final int BIAS = 3;

    // type
    public static final int NEURON = 0;
    public static final int SENSOR = 1;

    // Numeric identification of node
    int node_id;

    // Type is either NEURON or SENSOR
    int type;

    // Label of node : input, bias, hidden or output based on node constants
    int gen_node_label;

    // The incoming activity before being processed
    double activesum = 0.0;

    // The total activation entering in this Node based on its incoming links
    double activation = 0.0;

    // How many times this node is activated during activation of network
    double activation_count = 0.0;

    // When there are signal(s) to node, this switches from FALSE to TRUE
    boolean active_flag = false;

    // A list of pointers to incoming weighted signals from other nodes
    ArrayList<Link> incoming = new ArrayList<>();

    // A list of pointers to links carrying this node's signal
    ArrayList<Link> outgoing = new ArrayList<>();

    // Inner level of the node, used for calculating maximum depth
    public int inner_level = 0;

    // Whether node has been traversed, used for calculating maximum depth
    public boolean is_traversed = false;

    // Accessor function for activation
    public double getActivation() {
        return activation;
    }

    // Node Constructor
    public Node(int nodeId, int nType, int genNodeLabel) {
        node_id = nodeId;
        type = nType;
        gen_node_label = genNodeLabel;
    }

    // Find maximum depth of node from any input node recursively
    public int depth(int xlevel, int xmax_level) {
        // Base Case
        if (this.type == Node.SENSOR)
            return xlevel;
        xlevel++;

        // Recursive Step
        for (Link link : incoming) {
            Node _ynode = link.in_node;
            int cur_depth;
            if (!_ynode.is_traversed) {
                _ynode.is_traversed = true;
                cur_depth = _ynode.depth(xlevel, xmax_level);
                _ynode.inner_level = cur_depth - xlevel;
            } else
                cur_depth = xlevel + _ynode.inner_level;
            xmax_level = Math.max(xmax_level, cur_depth);
        }
        return xmax_level;
    }

    // Get activation output if it has been activated
    public double get_active_out() {
        if (activation_count > 0)
            return activation;
        else
            return 0.0;
    }

    // Load in sensor value to neuron if it is a sensor neuron
    public void sensor_load(double value) {
        if (type == Node.SENSOR) {
            //Time delay memory, puts sensor into next time-step
            activation_count++;
            activation = value;
        }
    }
}

// Link class where each link connects an input to an output node with a given weight
class Link {
    // Input node
    Node in_node = null;

    // Output node
    Node out_node = null;

    // Weight of connection
    double weight = 0.0;

    // Link Constructor
    public Link(int inode_num, int onode_num, double w, Network net) {
        // Search through nodes to find input node and output node based on id and add the links to the nodes
        for (Node _node : net.allNodes) {
            if (_node.node_id == inode_num) {
                in_node = _node;
                in_node.outgoing.add(this);
            }
            if (_node.node_id == onode_num) {
                out_node = _node;
                out_node.incoming.add(this);
            }
        }
        weight = w;
    }
}