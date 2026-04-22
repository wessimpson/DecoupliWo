package tracks.singlePlayer.ascii;

import core.game.StateObservation;
import ontology.Types;
import tools.ElapsedCpuTimer;
import tools.Vector2d;
import tracks.singlePlayer.advanced.sampleMCTS.Agent;

import java.util.ArrayList;

/**
 * Records per-tick ASCII frames + actions + rewards while delegating the
 * actual move selection to {@link Agent} (stock ``sampleMCTS``).
 *
 * <p>Agents are instantiated by {@code ArcadeMachine.createPlayer} via
 * reflection, so config and output plumbing are passed via static fields set
 * by {@link RunAsciiCollectionMCTS} before each episode:
 * <ul>
 *   <li>{@link #extractorForNextEpisode}: the ASCII extractor to use.</li>
 *   <li>{@link #mctsMillisForNextEpisode}: per-tick MCTS budget.</li>
 * </ul>
 * After an episode ends, the runner reads {@link #lastInstance} to drain the
 * captured buffer.
 *
 * <p>Single-threaded; safe as long as no more than one runOneGame() call is
 * live in the same JVM at a time (which matches {@code ArcadeMachine}'s own
 * usage).
 */
public class RecordingMCTSAgent extends Agent {

	public static AsciiGridExtractor extractorForNextEpisode;
	public static long mctsMillisForNextEpisode = 40L;
	public static RecordingMCTSAgent lastInstance;

	/** One step of captured data. */
	public static final class Frame {
		public final byte[] asciiGrid;
		public final int action;
		public final float reward;
		public final boolean done;
		public final float playerX;
		public final float playerY;

		public Frame(byte[] asciiGrid, int action, float reward, boolean done, float playerX, float playerY) {
			this.asciiGrid = asciiGrid;
			this.action = action;
			this.reward = reward;
			this.done = done;
			this.playerX = playerX;
			this.playerY = playerY;
		}
	}

	private final AsciiGridExtractor extractor;
	private final long mctsBudgetMs;
	private final Types.ACTIONS[] actionOrder;
	private final ArrayList<Frame> episode;
	private final double[] scoreHolder;

	/**
	 * Constructor signature required by {@code ArcadeMachine.createPlayer}
	 * (reflection finds ``public Agent(StateObservation, ElapsedCpuTimer)``).
	 */
	public RecordingMCTSAgent(StateObservation so, ElapsedCpuTimer elapsedTimer) {
		super(so, elapsedTimer);
		this.extractor = extractorForNextEpisode;
		this.mctsBudgetMs = mctsMillisForNextEpisode;
		this.actionOrder = Types.ACTIONS.values();
		this.episode = new ArrayList<>(2048);
		this.scoreHolder = new double[] { so.getGameScore() };
		if (this.extractor == null)
			throw new IllegalStateException(
					"RecordingMCTSAgent.extractorForNextEpisode must be set before runOneGame()");
		lastInstance = this;
	}

	/** Override to use our own per-tick time budget rather than the competition default. */
	@Override
	public Types.ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {
		byte[] ascii = extractor.extract(stateObs);
		Vector2d pos = stateObs.getAvatarPosition();
		float px = pos != null ? (float) pos.x : Float.NaN;
		float py = pos != null ? (float) pos.y : Float.NaN;
		double score = stateObs.getGameScore();
		float reward = (float) (score - scoreHolder[0]);
		scoreHolder[0] = score;

		ElapsedCpuTimer budgetTimer = new ElapsedCpuTimer();
		budgetTimer.setMaxTimeMillis(mctsBudgetMs);
		Types.ACTIONS action = super.act(stateObs, budgetTimer);

		int actionIndex = indexOfAction(action);
		episode.add(new Frame(ascii, actionIndex, reward, false, px, py));
		return action;
	}

	/** Mark the final captured frame's done flag so the shard layout records termination. */
	public void markEpisodeFinished() {
		int n = episode.size();
		if (n == 0) return;
		Frame last = episode.get(n - 1);
		episode.set(n - 1, new Frame(last.asciiGrid, last.action, last.reward, true, last.playerX, last.playerY));
	}

	public ArrayList<Frame> drainEpisode() {
		ArrayList<Frame> out = new ArrayList<>(episode);
		episode.clear();
		return out;
	}

	public int numAvailableActions() { return num_actions; }

	private int indexOfAction(Types.ACTIONS a) {
		for (int i = 0; i < actionOrder.length; i++) {
			if (actionOrder[i] == a) return i;
		}
		return 0;
	}
}
