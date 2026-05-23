package tracks.singlePlayer;

import java.awt.Graphics2D;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicLong;

import core.game.StateObservation;
import core.game.StateObservationMulti;
import core.player.AbstractPlayer;
import ontology.Types;
import tools.ElapsedCpuTimer;

/**
 * Wraps a single-player {@link AbstractPlayer} and records (state grid, action, player position)
 * into NumPy shards.  Increments the global frame counter exactly once per game tick.
 */
public final class TransitionRecordingPlayer extends AbstractPlayer {

	private final AbstractPlayer inner;
	private final GvgaiTransitionShardRecorder[] recorders;
	private final AtomicLong globalFrames;
	private final long maxGlobalFrames;
	private final long agentBudgetMs;
	private boolean firstStepInEpisode = true;

	public TransitionRecordingPlayer(AbstractPlayer inner, AtomicLong globalFrames,
			GvgaiTransitionShardRecorder... recorders) {
		this(inner, globalFrames, -1, recorders);
	}

	public TransitionRecordingPlayer(AbstractPlayer inner, AtomicLong globalFrames, long maxGlobalFrames,
			GvgaiTransitionShardRecorder... recorders) {
		this(inner, globalFrames, maxGlobalFrames, -1, recorders);
	}

	public TransitionRecordingPlayer(AbstractPlayer inner, AtomicLong globalFrames, long maxGlobalFrames,
			long agentBudgetMs, GvgaiTransitionShardRecorder... recorders) {
		this.inner = inner;
		this.globalFrames = globalFrames;
		this.maxGlobalFrames = maxGlobalFrames;
		this.agentBudgetMs = agentBudgetMs;
		this.recorders = recorders.clone();
	}

	@Override
	public Types.ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {
		for (GvgaiTransitionShardRecorder r : recorders)
			r.captureFrame();

		ElapsedCpuTimer agentTimer = elapsedTimer;
		if (agentBudgetMs > 0) {
			agentTimer = new ElapsedCpuTimer();
			agentTimer.setMaxTimeMillis(agentBudgetMs);
		}
		Types.ACTIONS a = inner.act(stateObs, agentTimer);

		try {
			boolean restartedStep = firstStepInEpisode;
			for (GvgaiTransitionShardRecorder r : recorders)
				r.commitFrame(stateObs, a, restartedStep);
			firstStepInEpisode = false;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		long currentFrames = globalFrames.incrementAndGet();
		if (maxGlobalFrames > 0 && currentFrames >= maxGlobalFrames)
			throw new FrameBudgetReachedException();
		return a;
	}

	@Override
	public Types.ACTIONS act(StateObservationMulti stateObs, ElapsedCpuTimer elapsedTimer) {
		return Types.ACTIONS.ACTION_NIL;
	}

	@Override
	public void result(StateObservation stateObs, ElapsedCpuTimer elapsedCpuTimer) {
		inner.result(stateObs, elapsedCpuTimer);
	}

	@Override
	public void resultMulti(StateObservationMulti stateObs, ElapsedCpuTimer elapsedCpuTimer) {
		inner.resultMulti(stateObs, elapsedCpuTimer);
	}

	@Override
	public void draw(Graphics2D g) {
		inner.draw(g);
	}

	public static final class FrameBudgetReachedException extends RuntimeException {
		private static final long serialVersionUID = 1L;
	}
}
