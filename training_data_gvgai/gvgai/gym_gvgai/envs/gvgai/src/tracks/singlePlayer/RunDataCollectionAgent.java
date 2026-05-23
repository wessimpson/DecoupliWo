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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
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
	public static final String DEFAULT_RANDOM_AGENT = "tracks.singlePlayer.simple.simpleRandom.Agent";

	/** GVGAI's VGDLParser / VGDLRegistry mutate global singletons — serialize setup. */
	private static final Object SETUP_LOCK = new Object();

	public static void main(String[] args) {
		if (args == null || args.length == 0) {
			printUsage();
			System.exit(1);
		}
		String game = null;
		String level = null;
		String agent = null;
		boolean agentExplicit = false;
		String profile = "mcts_default";
		String split = "train";
		boolean visuals = false;
		Integer seed = null;
		Path outputRoot = null;
		String spriteRoot = null;
		String levelIndex = null;
		String sourceRoot = null;
		String sourceBaseGame = null;
		String sourceRuleTag = null;
		int chunkSize = 1_000;
		double scale = 1.0;
		long totalTimesteps = -1;
		int numEnvs = 1;
		String mctsK = null;
		String mctsRolloutDepth = null;
		String mctsMaxIterations = null;
		String mctsFinalSelection = null;
		String mctsTemperature = null;
		String mctsActionEpsilon = null;

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
				agentExplicit = true;
			} else if ("--profile".equals(a) && i + 1 < args.length) {
				profile = args[++i];
			} else if ("--split".equals(a) && i + 1 < args.length) {
				split = args[++i];
			} else if ("--visuals".equals(a)) {
				visuals = true;
			} else if ("--no-visuals".equals(a)) {
				visuals = false;
			} else if ("--seed".equals(a) && i + 1 < args.length) {
				seed = Integer.parseInt(args[++i]);
			} else if ("--output-root".equals(a) && i + 1 < args.length) {
				outputRoot = Paths.get(args[++i]).toAbsolutePath().normalize();
			} else if ("--sprite-root".equals(a) && i + 1 < args.length) {
				spriteRoot = args[++i];
			} else if ("--level-index".equals(a) && i + 1 < args.length) {
				levelIndex = args[++i];
			} else if ("--source-root".equals(a) && i + 1 < args.length) {
				sourceRoot = args[++i];
			} else if ("--source-base-game".equals(a) && i + 1 < args.length) {
				sourceBaseGame = args[++i];
			} else if ("--source-rule-tag".equals(a) && i + 1 < args.length) {
				sourceRuleTag = args[++i];
			} else if ("--chunk-size".equals(a) && i + 1 < args.length) {
				chunkSize = Integer.parseInt(args[++i]);
			} else if ("--scale".equals(a) && i + 1 < args.length) {
				scale = Double.parseDouble(args[++i]);
			} else if ("--total-timesteps".equals(a) && i + 1 < args.length) {
				totalTimesteps = Long.parseLong(args[++i]);
			} else if ("--num-envs".equals(a) && i + 1 < args.length) {
				numEnvs = Integer.parseInt(args[++i]);
			} else if ("--mcts-k".equals(a) && i + 1 < args.length) {
				mctsK = args[++i];
			} else if ("--mcts-rollout-depth".equals(a) && i + 1 < args.length) {
				mctsRolloutDepth = args[++i];
			} else if ("--mcts-max-iterations".equals(a) && i + 1 < args.length) {
				mctsMaxIterations = args[++i];
			} else if ("--mcts-final-selection".equals(a) && i + 1 < args.length) {
				mctsFinalSelection = args[++i];
			} else if ("--mcts-temperature".equals(a) && i + 1 < args.length) {
				mctsTemperature = args[++i];
			} else if ("--mcts-action-epsilon".equals(a) && i + 1 < args.length) {
				mctsActionEpsilon = args[++i];
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
				level = defaultLevelForGame(game);
			numEnvs = Math.max(1, numEnvs);
			configureSpriteImagePath(spriteRoot);

			MctsProfile mctsProfile = applyMctsProfile(profile);
		if (mctsK != null) setMctsProperty("mcts.k", mctsK);
		if (mctsRolloutDepth != null) setMctsProperty("mcts.rolloutDepth", mctsRolloutDepth);
		if (mctsMaxIterations != null) setMctsProperty("mcts.maxIterations", mctsMaxIterations);
		if (mctsFinalSelection != null) setMctsProperty("mcts.finalSelection", mctsFinalSelection);
		if (mctsTemperature != null) setMctsProperty("mcts.temperature", mctsTemperature);
		if (mctsActionEpsilon != null) setMctsProperty("mcts.actionEpsilon", mctsActionEpsilon);
		if (!agentExplicit)
			agent = "random".equalsIgnoreCase(profile) ? DEFAULT_RANDOM_AGENT : DEFAULT_MCTS_AGENT;

		String gameStem = gameStem(game);
		Path[] transitionRoots;
		if (outputRoot != null) {
			transitionRoots = new Path[] { outputRoot };
		} else {
			transitionRoots = new Path[] { defaultTransitionRoot(split) };
		}
		String metadataJson = buildMetadataJson(profile, mctsProfile, agent, game, level, gameStem,
				split, scale, chunkSize, seed, numEnvs, totalTimesteps,
				levelIndex, sourceRoot, sourceBaseGame, sourceRuleTag);

		System.out.println("Game:  " + game);
		System.out.println("Level: " + level);
			System.out.println("Agent: " + agent);
			System.out.println("Profile: " + profile);
			System.out.println("Sprites: " + CompetitionParameters.IMG_PATH);
			System.out.println("Output: " + transitionRoots[0].resolve(gameStem));

		try {
			if (totalTimesteps > 0) {
				System.out.printf("Target: %d frames, %d parallel env(s), chunk_size=%d%n",
						totalTimesteps, numEnvs, chunkSize);
				collectParallel(game, level, visuals, agent, seed, transitionRoots,
						gameStem, chunkSize, scale, totalTimesteps, numEnvs, metadataJson);
			} else {
				System.out.println("Mode: single episode");
				int randomSeed = seed != null ? seed : new Random().nextInt();
				AtomicLong gf = new AtomicLong();
				runOneEpisode(game, level, visuals, agent, randomSeed, transitionRoots,
						gameStem, chunkSize, scale, null, gf, true, metadataJson, -1);
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
			long totalTimesteps, int numEnvs, String metadataJson) throws InterruptedException, IOException {

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
					transitionRoots[0], envStem, chunkSize, probe, scale, globalFrames, metadataJson);
			System.out.println("Obs: " + tmp.getNativeWidth() + "x" + tmp.getNativeHeight()
					+ " -> " + tmp.getImageWidth() + "x" + tmp.getImageHeight()
					+ " (scale=" + scale + ")");
		}

		long startMs = System.currentTimeMillis();
		ExecutorService pool = Executors.newFixedThreadPool(numEnvs);
		List<Future<?>> futures = new ArrayList<>();

		for (int w = 0; w < numEnvs; w++) {
			final int workerId = w;
			final long workerSeed = (baseSeed != null ? baseSeed : System.nanoTime()) + workerId * 999983L;
			futures.add(pool.submit(() -> {
				Random seedRng = new Random(workerSeed);
				try {
					GvgaiTransitionShardRecorder[] recorders = new GvgaiTransitionShardRecorder[transitionRoots.length];
					for (int i = 0; i < transitionRoots.length; i++)
						recorders[i] = new GvgaiTransitionShardRecorder(
								transitionRoots[i], envStem, chunkSize, null, scale, globalFrames, metadataJson);

					while (globalFrames.get() < totalTimesteps) {
						int epSeed = seedRng.nextInt();
						runOneEpisode(gameFile, levelFile, false, agentName, epSeed,
								transitionRoots, envStem, chunkSize, scale, recorders, globalFrames, false, metadataJson,
								totalTimesteps);
						episodeCounter.incrementAndGet();
					}

					for (GvgaiTransitionShardRecorder r : recorders)
						r.close();
				} catch (IOException e) {
					throw new RuntimeException("Worker " + workerId + " failed", e);
				}
			}));
		}

		pool.shutdown();
		// Progress monitor: workers exit when frame budget is met, then close() flushes tail shards.
		while (!pool.awaitTermination(1, TimeUnit.SECONDS)) {
			long frames = globalFrames.get();
			int eps = episodeCounter.get();
			printProgress(frames, totalTimesteps, eps, numEnvs, startMs);
		}
		for (Future<?> f : futures) {
			try {
				f.get();
			} catch (ExecutionException e) {
				throw new RuntimeException(e.getCause());
			}
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
			boolean verbose, String metadataJson, long maxGlobalFrames) throws IOException {

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
						transitionRoots[i], envStem, chunkSize, toPlay, scale, globalFrames, metadataJson);
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

		TransitionRecordingPlayer wrapped = new TransitionRecordingPlayer(inner, globalFrames, maxGlobalFrames, recorders);
		wrapped.setPlayerID(0);

		// Actual gameplay runs outside the lock — fully parallel across workers.
		Player[] players = new Player[] { wrapped };
		try {
			if (visuals)
				toPlay.playGame(players, randomSeed, false, 0);
			else
				toPlay.runGame(players, randomSeed);
		} catch (TransitionRecordingPlayer.FrameBudgetReachedException done) {
			// Expected in fixed-frame collection mode.
		} finally {
			if (ownRecorders)
				for (GvgaiTransitionShardRecorder r : recorders)
					r.close();

			ArcadeMachine.tearPlayerDown(toPlay, players, null, randomSeed, true);
			if (verbose) { toPlay.handleResult(); toPlay.printResult(); }
		}
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

	static final class MctsProfile {
		final String name;
		final String k;
		final String rolloutDepth;
		final String maxIterations;
		final String finalSelection;
		final String temperature;
		final String actionEpsilon;

		MctsProfile(String name, String k, String rolloutDepth, String maxIterations,
				String finalSelection, String temperature, String actionEpsilon) {
			this.name = name;
			this.k = k;
			this.rolloutDepth = rolloutDepth;
			this.maxIterations = maxIterations;
			this.finalSelection = finalSelection;
			this.temperature = temperature;
			this.actionEpsilon = actionEpsilon;
		}
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
		String stem = g.substring(0, g.length() - 4);
		return Paths.get("gym_gvgai", "envs", "games", stem + "_v0", stem + ".txt").toString();
	}

	static String defaultLevelForGame(String gamePath) {
		Path p = Paths.get(gamePath);
		String stem = gameStem(gamePath);
		return p.resolveSibling(stem + "_lvl0.txt").toString();
	}

	static Path defaultTransitionRoot(String split) {
		Path highCapacity = Paths.get("/hdd2", "soyuj", "transition_data");
		if (Files.isDirectory(highCapacity.getParent()))
			return highCapacity.resolve(split);

		Path cwd = Paths.get(System.getProperty("user.dir")).toAbsolutePath().normalize();
		Path workspace = cwd;
		while (workspace != null && !"game-wm".equals(workspace.getFileName() == null ? "" : workspace.getFileName().toString())) {
			workspace = workspace.getParent();
		}
		if (workspace != null) {
			Path decoupliwo = workspace.resolve("DecoupliWo");
			if (Files.isDirectory(decoupliwo))
				return decoupliwo.resolve("data").resolve("transitions").resolve(split);
		}
		Path sibling = cwd.resolveSibling("DecoupliWo");
		if (Files.isDirectory(sibling))
			return sibling.resolve("data").resolve("transitions").resolve(split);
		return cwd.resolve("data").resolve("transitions").resolve(split);
	}

	static void configureSpriteImagePath(String spriteRoot) {
		Path root = null;
		if (spriteRoot != null && spriteRoot.trim().length() > 0)
			root = Paths.get(spriteRoot.trim()).toAbsolutePath().normalize();

		if (root == null || !Files.isDirectory(root)) {
			Path cwd = Paths.get(System.getProperty("user.dir")).toAbsolutePath().normalize();
			List<Path> candidates = new ArrayList<Path>();
			candidates.add(cwd.resolve("gym_gvgai").resolve("envs").resolve("gvgai").resolve("sprites"));
			candidates.add(cwd.resolve("sprites"));
			Path workspace = cwd;
			while (workspace != null && !"game-wm".equals(workspace.getFileName() == null ? "" : workspace.getFileName().toString()))
				workspace = workspace.getParent();
			if (workspace != null) {
				candidates.add(workspace.resolve("GVGAI_jpype").resolve("gym_gvgai").resolve("envs").resolve("gvgai").resolve("sprites"));
				candidates.add(workspace.resolve("DecoupliWo").resolve("training_data_gvgai").resolve("gvgai").resolve("gym_gvgai").resolve("envs").resolve("gvgai").resolve("sprites"));
			}
			for (Path candidate : candidates) {
				if (Files.isDirectory(candidate)) {
					root = candidate.toAbsolutePath().normalize();
					break;
				}
			}
		}

		if (root != null && Files.isDirectory(root)) {
			String path = root.toString();
			if (!path.endsWith("/") && !path.endsWith("\\"))
				path = path + "/";
			CompetitionParameters.IMG_PATH = path;
		}
	}

	static void listDataCollectionGames() {
		System.out.println("VGDL games in gym_gvgai/envs/games/*_v0:");
		Path dir = Paths.get("gym_gvgai", "envs", "games");
		List<String> names = new ArrayList<>();
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir, "*_v0")) {
			for (Path p : stream) {
				if (!Files.isDirectory(p))
					continue;
				String dname = p.getFileName().toString();
				String stem = dname.substring(0, dname.length() - 3);
				if (Files.isRegularFile(p.resolve(stem + ".txt")))
					names.add(stem);
			}
		} catch (IOException e) {
			System.err.println(e.getMessage());
			return;
		}
		Collections.sort(names);
		for (String n : names)
			System.out.println("  " + n);
	}

	static MctsProfile applyMctsProfile(String profile) {
		String p = profile == null ? "mcts_default" : profile.trim().toLowerCase();
		MctsProfile cfg;
		if ("random".equals(p)) {
			cfg = new MctsProfile(p, null, null, null, null, null, null);
		} else if ("mcts_exploit".equals(p)) {
			cfg = new MctsProfile(p, "0.50", "12", "150", "best_value", "0.25", "0.0");
		} else if ("mcts_balanced".equals(p)) {
			cfg = new MctsProfile(p, "1.41421356237", "10", "100", "visit_softmax", "0.35", "0.02");
		} else if ("mcts_explore".equals(p)) {
			cfg = new MctsProfile(p, "2.50", "12", "80", "visit_softmax", "1.00", "0.05");
		} else if ("mcts_scout".equals(p)) {
			cfg = new MctsProfile(p, "4.00", "6", "40", "visit_softmax", "1.50", "0.15");
		} else if ("mcts_default".equals(p) || "default".equals(p)) {
			cfg = new MctsProfile("mcts_default", null, null, null, null, null, null);
		} else {
			throw new IllegalArgumentException("Unknown profile: " + profile);
		}
		if (cfg.k != null) setMctsProperty("mcts.k", cfg.k);
		if (cfg.rolloutDepth != null) setMctsProperty("mcts.rolloutDepth", cfg.rolloutDepth);
		if (cfg.maxIterations != null) setMctsProperty("mcts.maxIterations", cfg.maxIterations);
		if (cfg.finalSelection != null) setMctsProperty("mcts.finalSelection", cfg.finalSelection);
		if (cfg.temperature != null) setMctsProperty("mcts.temperature", cfg.temperature);
		if (cfg.actionEpsilon != null) setMctsProperty("mcts.actionEpsilon", cfg.actionEpsilon);
		return cfg;
	}

	static void setMctsProperty(String key, String value) {
		System.setProperty(key, value);
	}

	static String buildMetadataJson(String profile, MctsProfile cfg, String agent, String game,
			String level, String envStem, String split, double scale, int chunkSize, Integer seed,
			int numEnvs, long totalTimesteps, String levelIndex, String sourceRoot,
			String sourceBaseGame, String sourceRuleTag) {
		StringBuilder sb = new StringBuilder();
		sb.append("{\n");
		appendJson(sb, "profile", profile).append(",\n");
		appendJson(sb, "policy", "random".equalsIgnoreCase(profile) ? "random" : "mcts").append(",\n");
		appendJson(sb, "agent", agent).append(",\n");
		appendJson(sb, "game_file", game).append(",\n");
		appendJson(sb, "level_file", level).append(",\n");
		appendJson(sb, "env_stem", envStem).append(",\n");
		appendJson(sb, "render_mode", "sprite_images").append(",\n");
		appendJson(sb, "sprite_root", CompetitionParameters.IMG_PATH).append(",\n");
		appendJson(sb, "level_index", levelIndex != null ? levelIndex : inferLevelIndex(level)).append(",\n");
		appendJson(sb, "source_root", sourceRoot).append(",\n");
		appendJson(sb, "source_base_game", sourceBaseGame).append(",\n");
		appendJson(sb, "source_rule_tag", sourceRuleTag).append(",\n");
		appendJson(sb, "split", split).append(",\n");
		sb.append("  \"scale\": ").append(scale).append(",\n");
		sb.append("  \"chunk_size\": ").append(chunkSize).append(",\n");
		sb.append("  \"base_seed\": ").append(seed == null ? "null" : seed.toString()).append(",\n");
		sb.append("  \"num_envs\": ").append(numEnvs).append(",\n");
		sb.append("  \"total_timesteps\": ").append(totalTimesteps).append(",\n");
		sb.append("  \"mcts\": {\n");
		appendJson(sb, "k", System.getProperty("mcts.k")).append(",\n");
		appendJson(sb, "rollout_depth", System.getProperty("mcts.rolloutDepth")).append(",\n");
		appendJson(sb, "max_iterations", System.getProperty("mcts.maxIterations")).append(",\n");
		appendJson(sb, "final_selection", System.getProperty("mcts.finalSelection")).append(",\n");
		appendJson(sb, "temperature", System.getProperty("mcts.temperature")).append(",\n");
		appendJson(sb, "action_epsilon", System.getProperty("mcts.actionEpsilon")).append("\n");
		sb.append("  }\n");
		sb.append("}\n");
		return sb.toString();
	}

	static String inferLevelIndex(String levelPath) {
		if (levelPath == null)
			return null;
		String name = Paths.get(levelPath).getFileName().toString();
		int start = name.indexOf("lvl");
		if (start < 0)
			return null;
		start += 3;
		int end = start;
		while (end < name.length() && Character.isDigit(name.charAt(end)))
			end++;
		if (end == start)
			return null;
		return name.substring(start, end);
	}

	static StringBuilder appendJson(StringBuilder sb, String key, String value) {
		sb.append("  \"").append(jsonEscape(key)).append("\": ");
		if (value == null) {
			sb.append("null");
		} else {
			sb.append("\"").append(jsonEscape(value)).append("\"");
		}
		return sb;
	}

	static String jsonEscape(String value) {
		return value.replace("\\", "\\\\").replace("\"", "\\\"");
	}

	static void printUsage() {
		System.out.println("Collect pixel transition data from GVGAI games using parallel environments.");
		System.out.println("Writes to /hdd2/soyuj/transition_data/<split>/<game>/ by default when available.");
		System.out.println();
		System.out.println("Agent: " + DEFAULT_MCTS_AGENT);
		System.out.println();
		System.out.println("Usage:");
		System.out.println("  java ... tracks.singlePlayer.RunDataCollectionAgent <game> [options]");
		System.out.println();
		System.out.println("Options:");
		System.out.println("  --list                  List available games and exit.");
		System.out.println("  --profile <name>        random, mcts_default, mcts_exploit, mcts_balanced, mcts_explore, mcts_scout.");
		System.out.println("  --split <name>          Output split when --output-root is omitted (default: train).");
		System.out.println("  --total-timesteps <n>   Collect n frames across multiple episodes (required for parallel).");
		System.out.println("  --num-envs <int>        Parallel environments (default: 1).");
		System.out.println("  --scale <float>         Render scale 0..1 (default: 1.0, full-resolution RGB).");
		System.out.println("  --chunk-size <int>      Frames per shard (default: 1000).");
		System.out.println("  --seed <int>            Base RNG seed.");
		System.out.println("  --level <path>          Level file (default: inferred).");
		System.out.println("  --level-index <int>     Metadata-only level index when --level is an explicit file.");
		System.out.println("  --source-root <dir>     Metadata-only source game root.");
		System.out.println("  --source-base-game <s>  Metadata-only base game name.");
		System.out.println("  --source-rule-tag <s>   Metadata-only rule tag, empty for base games.");
		System.out.println("  --agent <class>         Agent class (default: sample MCTS).");
		System.out.println("  --mcts-k <float>        Override UCT exploration constant.");
		System.out.println("  --mcts-rollout-depth <int>");
		System.out.println("  --mcts-max-iterations <int>");
		System.out.println("  --mcts-final-selection <most_visited|best_value|visit_softmax|value_softmax>");
		System.out.println("  --mcts-temperature <float>");
		System.out.println("  --mcts-action-epsilon <float>");
		System.out.println("  --visuals               Open Swing window (single-episode only).");
		System.out.println("  --sprite-root <dir>     Directory containing sprite image assets.");
		System.out.println("  --output-root <dir>     Single output root.");
		System.out.println("  -h, --help              This message.");
		System.out.println();
		System.out.println("Examples:");
		System.out.println("  # 100k frames with 8 parallel envs");
		System.out.println("  java -cp gym_gvgai/envs/gvgai/GVGAI_Build tracks.singlePlayer.RunDataCollectionAgent aliens \\");
		System.out.println("        --profile mcts_explore --total-timesteps 100000 --num-envs 8");
	}
}
