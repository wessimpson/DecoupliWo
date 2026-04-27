package tracks.singlePlayer;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import core.competition.CompetitionParameters;
import core.game.Game;
import core.player.AbstractPlayer;
import core.player.Player;
import core.vgdl.VGDLFactory;
import core.vgdl.VGDLParser;
import core.vgdl.VGDLRegistry;
import tracks.ArcadeMachine;

/**
 * Collects pixel-based transition data by running parallel GVGAI game environments.
 * Each parallel worker runs its own game+agent loop, writing to globally-unique shards.
 * Replays with different seeds until {@code --total-timesteps} frames are collected.
 */
public class RunDataCollectionAgent {

	public static final String DEFAULT_MCTS_AGENT = "tracks.singlePlayer.advanced.sampleMCTS.Agent";

	/** GVGAI's VGDLParser / VGDLRegistry mutate global singletons — serialize setup. */
	private static final Object SETUP_LOCK = new Object();

	public static void main(String[] args) {
		if (args == null || args.length == 0) {
			printUsage();
			System.exit(1);
		}
		String game = null;
		String level = null;
		String agent = DEFAULT_MCTS_AGENT;
		boolean visuals = false;
		Integer seed = null;
		Path outputRoot = null;
		int chunkSize = 1_000;
		double scale = 0.5;
		long totalTimesteps = -1;
		int numEnvs = 1;

		for (int i = 0; i < args.length; i++) {
			String a = args[i];
			if ("-h".equals(a) || "--help".equals(a)) {
				printUsage();
				System.exit(0);
			} else if ("--list".equals(a)) {
				listDataCollectionGames();
				System.exit(0);
			} else if ("--game".equals(a) && i + 1 < args.length) {
				game = normalizeGamePath(args[++i]);
			} else if ("--level".equals(a) && i + 1 < args.length) {
				level = args[++i];
			} else if ("--agent".equals(a) && i + 1 < args.length) {
				agent = args[++i];
			} else if ("--visuals".equals(a)) {
				visuals = true;
			} else if ("--no-visuals".equals(a)) {
				visuals = false;
			} else if ("--seed".equals(a) && i + 1 < args.length) {
				seed = Integer.parseInt(args[++i]);
			} else if ("--output-root".equals(a) && i + 1 < args.length) {
				outputRoot = Paths.get(args[++i]).toAbsolutePath().normalize();
			} else if ("--chunk-size".equals(a) && i + 1 < args.length) {
				chunkSize = Integer.parseInt(args[++i]);
			} else if ("--scale".equals(a) && i + 1 < args.length) {
				scale = Double.parseDouble(args[++i]);
			} else if ("--total-timesteps".equals(a) && i + 1 < args.length) {
				totalTimesteps = Long.parseLong(args[++i]);
			} else if ("--num-envs".equals(a) && i + 1 < args.length) {
				numEnvs = Integer.parseInt(args[++i]);
			} else if (!a.startsWith("-") && game == null) {
				game = normalizeGamePath(a);
			} else {
				System.err.println("Unknown or misplaced argument: " + a);
				printUsage();
				System.exit(1);
			}
		}

		if (game == null) {
			System.err.println("Missing game.");
			printUsage();
			System.exit(1);
		}
		if (level == null)
			level = Test.defaultLevelForOneArgGame(game);
		numEnvs = Math.max(1, numEnvs);

		String gameStem = gameStem(game);
		Path[] transitionRoots;
		if (outputRoot != null) {
			transitionRoots = new Path[] { outputRoot };
		} else {
			Path repo = repoRoot();
			transitionRoots = new Path[] { repo.resolve("data").resolve("transitions").resolve("train") };
		}

		System.out.println("Game:  " + game);
		System.out.println("Level: " + level);
		System.out.println("Agent: " + agent);
		System.out.println("Output: " + transitionRoots[0].resolve(gameStem));

		try {
			if (totalTimesteps > 0) {
				System.out.printf("Target: %d frames, %d parallel env(s), chunk_size=%d%n",
						totalTimesteps, numEnvs, chunkSize);
				collectParallel(game, level, visuals, agent, seed, transitionRoots,
						gameStem, chunkSize, scale, totalTimesteps, numEnvs);
			} else {
				System.out.println("Mode: single episode");
				int randomSeed = seed != null ? seed : new Random().nextInt();
				AtomicLong gf = new AtomicLong();
				runOneEpisode(game, level, visuals, agent, randomSeed, transitionRoots,
						gameStem, chunkSize, scale, null, gf, true);
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
	}

	// -----------------------------------------------------------------------
	// Parallel collection
	// -----------------------------------------------------------------------

	static void collectParallel(String gameFile, String levelFile, boolean visuals, String agentName,
			Integer baseSeed, Path[] transitionRoots, String envStem, int chunkSize, double scale,
			long totalTimesteps, int numEnvs) throws InterruptedException, IOException {

		// Init singletons once on the main thread before spawning workers.
		VGDLFactory.GetInstance().init();
		VGDLRegistry.GetInstance().init();

		AtomicLong globalFrames = new AtomicLong();
		AtomicInteger episodeCounter = new AtomicInteger();

		// Print obs dimensions from a quick probe game.
		{
			Game probe = new VGDLParser().parseGame(gameFile);
			probe.buildLevel(levelFile, 0);
			GvgaiTransitionShardRecorder tmp = new GvgaiTransitionShardRecorder(
					transitionRoots[0], envStem, chunkSize, probe, scale, globalFrames);
			System.out.println("Obs: " + tmp.getNativeWidth() + "x" + tmp.getNativeHeight()
					+ " -> " + tmp.getImageWidth() + "x" + tmp.getImageHeight()
					+ " (scale=" + scale + ")");
		}

		long startMs = System.currentTimeMillis();
		ExecutorService pool = Executors.newFixedThreadPool(numEnvs);

		for (int w = 0; w < numEnvs; w++) {
			final int workerId = w;
			final long workerSeed = (baseSeed != null ? baseSeed : System.nanoTime()) + workerId * 999983L;
			pool.submit(() -> {
				Random seedRng = new Random(workerSeed);
				try {
					GvgaiTransitionShardRecorder[] recorders = new GvgaiTransitionShardRecorder[transitionRoots.length];
					for (int i = 0; i < transitionRoots.length; i++)
						recorders[i] = new GvgaiTransitionShardRecorder(
								transitionRoots[i], envStem, chunkSize, null, scale, globalFrames);

					while (globalFrames.get() < totalTimesteps) {
						int epSeed = seedRng.nextInt();
						runOneEpisode(gameFile, levelFile, false, agentName, epSeed,
								transitionRoots, envStem, chunkSize, scale, recorders, globalFrames, false);
						episodeCounter.incrementAndGet();
					}

					for (GvgaiTransitionShardRecorder r : recorders)
						r.close();
				} catch (IOException e) {
					throw new RuntimeException("Worker " + workerId + " failed", e);
				}
			});
		}

		pool.shutdown();
		// Progress monitor: workers exit when frame budget is met, then close() flushes tail shards.
		while (!pool.awaitTermination(1, TimeUnit.SECONDS)) {
			long frames = globalFrames.get();
			int eps = episodeCounter.get();
			printProgress(frames, totalTimesteps, eps, numEnvs, startMs);
		}

		long frames = globalFrames.get();
		int eps = episodeCounter.get();
		printProgress(frames, totalTimesteps, eps, numEnvs, startMs);
		System.out.println();
		System.out.printf("Done: %d frames, %d episodes, %d envs, %s%n",
				frames, eps, numEnvs, formatDuration(System.currentTimeMillis() - startMs));
	}

	// -----------------------------------------------------------------------
	// Single episode
	// -----------------------------------------------------------------------

	static void runOneEpisode(String game_file, String level_file, boolean visuals, String agentName,
			int randomSeed, Path[] transitionRoots, String envStem, int chunkSize, double scale,
			GvgaiTransitionShardRecorder[] sharedRecorders,
			AtomicLong globalFrames,
			boolean verbose) throws IOException {

		Game toPlay;
		AbstractPlayer inner;

		// GVGAI's VGDLParser/Registry mutate global singletons — serialize setup.
		synchronized (SETUP_LOCK) {
			if (sharedRecorders == null) {
				VGDLFactory.GetInstance().init();
				VGDLRegistry.GetInstance().init();
			}

			toPlay = new VGDLParser().parseGame(game_file);
			toPlay.buildLevel(level_file, randomSeed);
			ArcadeMachine.warmUp(toPlay, CompetitionParameters.WARMUP_TIME);

			if (toPlay.no_players != 1) {
				System.err.println("Transition recording supports single-player games only.");
				return;
			}

			inner = ArcadeMachine.createPlayer(agentName, null, toPlay.getObservation(), randomSeed, false);
		}

		if (inner == null) {
			toPlay.disqualify();
			if (verbose) { toPlay.handleResult(); toPlay.printResult(); }
			return;
		}

		boolean ownRecorders = (sharedRecorders == null);
		GvgaiTransitionShardRecorder[] recorders;
		if (ownRecorders) {
			recorders = new GvgaiTransitionShardRecorder[transitionRoots.length];
			for (int i = 0; i < transitionRoots.length; i++)
				recorders[i] = new GvgaiTransitionShardRecorder(
						transitionRoots[i], envStem, chunkSize, toPlay, scale, globalFrames);
		} else {
			recorders = sharedRecorders;
			for (GvgaiTransitionShardRecorder r : recorders)
				r.setGame(toPlay);
		}

		if (verbose) {
			System.out.println("Obs: " + recorders[0].getNativeWidth() + "x" + recorders[0].getNativeHeight()
					+ " -> " + recorders[0].getImageWidth() + "x" + recorders[0].getImageHeight()
					+ " (scale=" + scale + ")");
		}

		TransitionRecordingPlayer wrapped = new TransitionRecordingPlayer(inner, globalFrames, recorders);
		wrapped.setPlayerID(0);

		// Actual gameplay runs outside the lock — fully parallel across workers.
		Player[] players = new Player[] { wrapped };
		if (visuals)
			toPlay.playGame(players, randomSeed, false, 0);
		else
			toPlay.runGame(players, randomSeed);

		if (ownRecorders)
			for (GvgaiTransitionShardRecorder r : recorders)
				r.close();

		ArcadeMachine.tearPlayerDown(toPlay, players, null, randomSeed, true);
		if (verbose) { toPlay.handleResult(); toPlay.printResult(); }
	}

	// -----------------------------------------------------------------------
	// Progress bar
	// -----------------------------------------------------------------------

	private static void printProgress(long current, long total, int episodes, int envs, long startMs) {
		double frac = Math.min(1.0, (double) current / total);
		int pct = (int) (frac * 100);
		int barLen = 30;
		int filled = (int) (frac * barLen);
		StringBuilder bar = new StringBuilder();
		for (int i = 0; i < barLen; i++)
			bar.append(i < filled ? '#' : '-');

		long elapsedMs = System.currentTimeMillis() - startMs;
		String eta = "?";
		if (current > 0 && frac < 1.0) {
			long remainMs = (long) (elapsedMs / frac * (1.0 - frac));
			eta = formatDuration(remainMs);
		}
		double fps = elapsedMs > 0 ? current * 1000.0 / elapsedMs : 0;

		System.out.printf("\r[%s] %3d%%  %d/%d  ep %d  %d envs  %.0f fps  %s  eta %s   ",
				bar, pct, current, total, episodes, envs, fps, formatDuration(elapsedMs), eta);
		System.out.flush();
	}

	private static String formatDuration(long ms) {
		long sec = ms / 1000;
		if (sec < 60) return sec + "s";
		if (sec < 3600) return String.format("%dm%02ds", sec / 60, sec % 60);
		return String.format("%dh%02dm%02ds", sec / 3600, (sec % 3600) / 60, sec % 60);
	}

	// -----------------------------------------------------------------------
	// Utilities
	// -----------------------------------------------------------------------

	static Path repoRoot() {
		return Paths.get(System.getProperty("user.dir")).resolve("..").normalize().toAbsolutePath();
	}

	static String gameStem(String gamePath) {
		Path p = Paths.get(gamePath);
		String name = p.getFileName().toString();
		if (name.toLowerCase().endsWith(".txt"))
			return name.substring(0, name.length() - 4);
		return name;
	}

	static String normalizeGamePath(String raw) {
		String g = raw.trim();
		if (g.contains("/") || g.contains("\\"))
			return g;
		if (!g.toLowerCase().endsWith(".txt"))
			g = g + ".txt";
		return "examples/data_collection/" + g;
	}

	static void listDataCollectionGames() {
		System.out.println("VGDL games in examples/data_collection/ (*.txt):");
		Path dir = Paths.get("examples/data_collection");
		List<String> names = new ArrayList<>();
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir, "*.txt")) {
			for (Path p : stream)
				names.add(p.getFileName().toString());
		} catch (IOException e) {
			System.err.println(e.getMessage());
			return;
		}
		Collections.sort(names);
		for (String n : names)
			System.out.println("  " + n.replace(".txt", ""));
	}

	static void printUsage() {
		System.out.println("Collect pixel transition data from GVGAI games using parallel environments.");
		System.out.println("Writes to ../data/transitions/train/<game>/ by default.");
		System.out.println();
		System.out.println("Agent: " + DEFAULT_MCTS_AGENT);
		System.out.println();
		System.out.println("Usage:");
		System.out.println("  java ... tracks.singlePlayer.RunDataCollectionAgent <game> [options]");
		System.out.println();
		System.out.println("Options:");
		System.out.println("  --list                  List available games and exit.");
		System.out.println("  --total-timesteps <n>   Collect n frames across multiple episodes (required for parallel).");
		System.out.println("  --num-envs <int>        Parallel environments (default: 1).");
		System.out.println("  --scale <float>         Render scale 0..1 (default: 1.0, full-resolution RGB).");
		System.out.println("  --chunk-size <int>      Frames per shard (default: 1000).");
		System.out.println("  --seed <int>            Base RNG seed.");
		System.out.println("  --level <path>          Level file (default: inferred).");
		System.out.println("  --agent <class>         Agent class (default: sample MCTS).");
		System.out.println("  --visuals               Open Swing window (single-episode only).");
		System.out.println("  --output-root <dir>     Single output root.");
		System.out.println("  -h, --help              This message.");
		System.out.println();
		System.out.println("Examples:");
		System.out.println("  # 100k frames with 8 parallel envs");
		System.out.println("  java -cp \"out;gson-2.6.2.jar\" tracks.singlePlayer.RunDataCollectionAgent aliens \\");
		System.out.println("        --total-timesteps 100000 --num-envs 8");
	}
}
