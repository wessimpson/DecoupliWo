package tracks.singlePlayer.advanced.sampleMCTS;

import java.util.Random;

import core.game.StateObservation;
import ontology.Types;
import tools.ElapsedCpuTimer;

/**
 * Created with IntelliJ IDEA.
 * User: Diego
 * Date: 07/11/13
 * Time: 17:13
 */
public class SingleMCTSPlayer
{


    /**
     * Root of the tree.
     */
    public SingleTreeNode m_root;

    /**
     * Random generator.
     */
    public Random m_rnd;

    public int num_actions;
    public Types.ACTIONS[] actions;
    public String finalSelection;
    public double temperature;
    public double actionEpsilon;

    public SingleMCTSPlayer(Random a_rnd, int num_actions, Types.ACTIONS[] actions)
    {
        this.num_actions = num_actions;
        this.actions = actions;
        m_rnd = a_rnd;
        finalSelection = stringProperty("mcts.finalSelection", "most_visited");
        temperature = doubleProperty("mcts.temperature", 1.0);
        actionEpsilon = doubleProperty("mcts.actionEpsilon", 0.0);
    }

    /**
     * Inits the tree with the new observation state in the root.
     * @param a_gameState current state of the game.
     */
    public void init(StateObservation a_gameState)
    {
        //Set the game observation to a newly root node.
        //System.out.println("learning_style = " + learning_style);
        m_root = new SingleTreeNode(m_rnd, num_actions, actions);
        m_root.rootState = a_gameState;
    }

    /**
     * Runs MCTS to decide the action to take. It does not reset the tree.
     * @param elapsedTimer Timer when the action returned is due.
     * @return the action to execute in the game.
     */
    public int run(ElapsedCpuTimer elapsedTimer)
    {
        //Do the search within the available time.
        m_root.mctsSearch(elapsedTimer);

        if (actionEpsilon > 0.0 && m_rnd.nextDouble() < actionEpsilon)
            return m_rnd.nextInt(num_actions);

        if ("best_value".equalsIgnoreCase(finalSelection) || "best".equalsIgnoreCase(finalSelection))
            return m_root.bestAction();
        if ("visit_softmax".equalsIgnoreCase(finalSelection) || "softmax".equalsIgnoreCase(finalSelection))
            return m_root.visitSoftmaxAction(temperature);
        if ("value_softmax".equalsIgnoreCase(finalSelection))
            return m_root.valueSoftmaxAction(temperature);

        return m_root.mostVisitedAction();
    }

    private static String stringProperty(String key, String def) {
        String value = System.getProperty(key);
        return value == null ? def : value.trim();
    }

    private static double doubleProperty(String key, double def) {
        try {
            String value = System.getProperty(key);
            return value == null ? def : Double.parseDouble(value.trim());
        } catch (Exception ignored) {
            return def;
        }
    }

}
