package tracks.singlePlayer.advanced.sampleMCTS;

import java.util.Random;

import core.game.StateObservation;
import ontology.Types;
import tools.ElapsedCpuTimer;
import tools.Utils;

public class SingleTreeNode
{
    private final double HUGE_NEGATIVE = -10000000.0;
    private final double HUGE_POSITIVE =  10000000.0;
    public double epsilon = 1e-6;
    public double egreedyEpsilon = 0.05;
    public SingleTreeNode parent;
    public SingleTreeNode[] children;
    public double totValue;
    public int nVisits;
    public Random m_rnd;
    public int m_depth;
    protected double[] bounds = new double[]{Double.MAX_VALUE, -Double.MAX_VALUE};
    public int childIdx;

    public int num_actions;
    Types.ACTIONS[] actions;
    public int ROLLOUT_DEPTH = 10;
    public double K = Math.sqrt(2);
    public int MAX_ITERATIONS = -1;

    public StateObservation rootState;

    public SingleTreeNode(Random rnd, int num_actions, Types.ACTIONS[] actions) {
        this(null, -1, rnd, num_actions, actions);
    }

    public SingleTreeNode(SingleTreeNode parent, int childIdx, Random rnd, int num_actions, Types.ACTIONS[] actions) {
        this.parent = parent;
        this.m_rnd = rnd;
        this.num_actions = num_actions;
        this.actions = actions;
        children = new SingleTreeNode[num_actions];
        totValue = 0.0;
        this.childIdx = childIdx;
        if(parent != null) {
            m_depth = parent.m_depth+1;
            ROLLOUT_DEPTH = parent.ROLLOUT_DEPTH;
            K = parent.K;
            MAX_ITERATIONS = parent.MAX_ITERATIONS;
        } else {
            m_depth = 0;
            ROLLOUT_DEPTH = intProperty("mcts.rolloutDepth", ROLLOUT_DEPTH);
            K = doubleProperty("mcts.k", K);
            MAX_ITERATIONS = intProperty("mcts.maxIterations", MAX_ITERATIONS);
        }
    }


    public void mctsSearch(ElapsedCpuTimer elapsedTimer) {

        double avgTimeTaken = 0;
        double acumTimeTaken = 0;
        long remaining = elapsedTimer.remainingTimeMillis();
        int numIters = 0;

        int remainingLimit = 5;
        while(remaining > 2*avgTimeTaken && remaining > remainingLimit &&
                (MAX_ITERATIONS <= 0 || numIters < MAX_ITERATIONS)){

            StateObservation state = rootState.copy();

            ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();
            SingleTreeNode selected = treePolicy(state);
            double delta = selected.rollOut(state);
            backUp(selected, delta);

            numIters++;
            acumTimeTaken += (elapsedTimerIteration.elapsedMillis()) ;
            //System.out.println(elapsedTimerIteration.elapsedMillis() + " --> " + acumTimeTaken + " (" + remaining + ")");
            avgTimeTaken  = acumTimeTaken/numIters;
            remaining = elapsedTimer.remainingTimeMillis();
        }
    }

    public SingleTreeNode treePolicy(StateObservation state) {

        SingleTreeNode cur = this;

        while (!state.isGameOver() && cur.m_depth < ROLLOUT_DEPTH)
        {
            if (cur.notFullyExpanded()) {
                return cur.expand(state);

            } else {
                SingleTreeNode next = cur.uct(state);
                cur = next;
            }
        }

        return cur;
    }


    public SingleTreeNode expand(StateObservation state) {

        int bestAction = 0;
        double bestValue = -1;

        for (int i = 0; i < children.length; i++) {
            double x = m_rnd.nextDouble();
            if (x > bestValue && children[i] == null) {
                bestAction = i;
                bestValue = x;
            }
        }

        //Roll the state
        state.advance(actions[bestAction]);

        SingleTreeNode tn = new SingleTreeNode(this,bestAction,this.m_rnd,num_actions, actions);
        children[bestAction] = tn;
        return tn;
    }

    public SingleTreeNode uct(StateObservation state) {

        SingleTreeNode selected = null;
        double bestValue = -Double.MAX_VALUE;
        for (SingleTreeNode child : this.children)
        {
            double hvVal = child.totValue;
            double childValue =  hvVal / (child.nVisits + this.epsilon);

            childValue = Utils.normalise(childValue, bounds[0], bounds[1]);
            //System.out.println("norm child value: " + childValue);

            double uctValue = childValue +
                    K * Math.sqrt(Math.log(this.nVisits + 1) / (child.nVisits + this.epsilon));

            uctValue = Utils.noise(uctValue, this.epsilon, this.m_rnd.nextDouble());     //break ties randomly

            // small sampleRandom numbers: break ties in unexpanded nodes
            if (uctValue > bestValue) {
                selected = child;
                bestValue = uctValue;
            }
        }
        if (selected == null)
        {
            throw new RuntimeException("Warning! returning null: " + bestValue + " : " + this.children.length + " " +
            + bounds[0] + " " + bounds[1]);
        }

        //Roll the state:
        state.advance(actions[selected.childIdx]);

        return selected;
    }


    public double rollOut(StateObservation state)
    {
        int thisDepth = this.m_depth;

        while (!finishRollout(state,thisDepth)) {

            int action = m_rnd.nextInt(num_actions);
            state.advance(actions[action]);
            thisDepth++;
        }


        double delta = value(state);

        if(delta < bounds[0])
            bounds[0] = delta;
        if(delta > bounds[1])
            bounds[1] = delta;

        //double normDelta = utils.normalise(delta ,lastBounds[0], lastBounds[1]);

        return delta;
    }

    public double value(StateObservation a_gameState) {

        boolean gameOver = a_gameState.isGameOver();
        Types.WINNER win = a_gameState.getGameWinner();
        double rawScore = a_gameState.getGameScore();

        if(gameOver && win == Types.WINNER.PLAYER_LOSES)
            rawScore += HUGE_NEGATIVE;

        if(gameOver && win == Types.WINNER.PLAYER_WINS)
            rawScore += HUGE_POSITIVE;

        return rawScore;
    }

    public boolean finishRollout(StateObservation rollerState, int depth)
    {
        if(depth >= ROLLOUT_DEPTH)      //rollout end condition.
            return true;

        if(rollerState.isGameOver())               //end of game
            return true;

        return false;
    }

    public void backUp(SingleTreeNode node, double result)
    {
        SingleTreeNode n = node;
        while(n != null)
        {
            n.nVisits++;
            n.totValue += result;
            if (result < n.bounds[0]) {
                n.bounds[0] = result;
            }
            if (result > n.bounds[1]) {
                n.bounds[1] = result;
            }
            n = n.parent;
        }
    }


    public int mostVisitedAction() {
        int selected = -1;
        double bestValue = -Double.MAX_VALUE;
        boolean allEqual = true;
        double first = -1;

        for (int i=0; i<children.length; i++) {

            if(children[i] != null)
            {
                if(first == -1)
                    first = children[i].nVisits;
                else if(first != children[i].nVisits)
                {
                    allEqual = false;
                }

                double childValue = children[i].nVisits;
                childValue = Utils.noise(childValue, this.epsilon, this.m_rnd.nextDouble());     //break ties randomly
                if (childValue > bestValue) {
                    bestValue = childValue;
                    selected = i;
                }
            }
        }

        if (selected == -1)
        {
            System.out.println("Unexpected selection!");
            selected = 0;
        }else if(allEqual)
        {
            //If all are equal, we opt to choose for the one with the best Q.
            selected = bestAction();
        }
        return selected;
    }

    public int bestAction()
    {
        int selected = -1;
        double bestValue = -Double.MAX_VALUE;

        for (int i=0; i<children.length; i++) {

            if(children[i] != null) {
                //double tieBreaker = m_rnd.nextDouble() * epsilon;
                double childValue = children[i].totValue / (children[i].nVisits + this.epsilon);
                childValue = Utils.noise(childValue, this.epsilon, this.m_rnd.nextDouble());     //break ties randomly
                if (childValue > bestValue) {
                    bestValue = childValue;
                    selected = i;
                }
            }
        }

        if (selected == -1)
        {
            System.out.println("Unexpected selection!");
            selected = 0;
        }

        return selected;
    }

    public int visitSoftmaxAction(double temperature) {
        double[] scores = new double[children.length];
        boolean hasChild = false;
        for (int i = 0; i < children.length; i++) {
            if (children[i] != null) {
                scores[i] = children[i].nVisits;
                hasChild = true;
            } else {
                scores[i] = Double.NEGATIVE_INFINITY;
            }
        }
        if (!hasChild)
            return 0;
        if (temperature <= 0)
            return mostVisitedAction();
        return sampleSoftmax(scores, temperature);
    }

    public int valueSoftmaxAction(double temperature) {
        double[] scores = new double[children.length];
        boolean hasChild = false;
        for (int i = 0; i < children.length; i++) {
            if (children[i] != null) {
                scores[i] = children[i].totValue / (children[i].nVisits + this.epsilon);
                hasChild = true;
            } else {
                scores[i] = Double.NEGATIVE_INFINITY;
            }
        }
        if (!hasChild)
            return 0;
        if (temperature <= 0)
            return bestAction();
        return sampleSoftmax(scores, temperature);
    }

    private int sampleSoftmax(double[] scores, double temperature) {
        double max = -Double.MAX_VALUE;
        for (double score : scores) {
            if (!Double.isInfinite(score) && !Double.isNaN(score) && score > max)
                max = score;
        }
        if (max == -Double.MAX_VALUE)
            return 0;

        double sum = 0.0;
        double[] weights = new double[scores.length];
        for (int i = 0; i < scores.length; i++) {
            double score = scores[i];
            if (Double.isInfinite(score) || Double.isNaN(score)) {
                weights[i] = 0.0;
            } else {
                weights[i] = Math.exp((score - max) / Math.max(temperature, 1e-9));
                sum += weights[i];
            }
        }
        if (sum <= 0.0)
            return 0;

        double roll = m_rnd.nextDouble() * sum;
        double acc = 0.0;
        for (int i = 0; i < weights.length; i++) {
            acc += weights[i];
            if (roll <= acc)
                return i;
        }
        return weights.length - 1;
    }

    private static int intProperty(String key, int def) {
        try {
            String value = System.getProperty(key);
            return value == null ? def : Integer.parseInt(value.trim());
        } catch (Exception ignored) {
            return def;
        }
    }

    private static double doubleProperty(String key, double def) {
        try {
            String value = System.getProperty(key);
            return value == null ? def : Double.parseDouble(value.trim());
        } catch (Exception ignored) {
            return def;
        }
    }

    public boolean notFullyExpanded() {
        for (SingleTreeNode tn : children) {
            if (tn == null) {
                return true;
            }
        }

        return false;
    }
}
