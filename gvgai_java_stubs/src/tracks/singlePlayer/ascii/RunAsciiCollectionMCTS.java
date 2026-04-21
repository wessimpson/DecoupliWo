package tracks.singlePlayer.ascii;

import ontology.Types;
import tracks.ArcadeMachine;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

/**
 * Headless data-collection driver for the ASCII VAE training pipeline.
 *
 * <p>For each requested game / level / episode, runs stock
 * ``tracks.singlePlayer.advanced.sampleMCTS.Agent`` via
 * {@link RecordingMCTSAgent} (a thin subclass that records the per-tick ASCII
 * grid + chosen action + reward + player position), then flushes to shard
 * directories laid out exactly like the output of
 * ``data/collect_transitions.py`` so both pipelines share a single loader:
 *
 * <pre>
 *   &lt;out-root&gt;/&lt;game&gt;/shard_XXXXX/
 *       obs.npy        uint8[N, H, W]   ASCII bytes
 *       action.npy     int64[N]         Types.ACTIONS ordinal (0..6)
 *       n_actions.npy  int64 scalar     always 7 (full Types.ACTIONS space)
 *       reward.npy     float32[N]       per-tick score delta
 *       done.npy       uint8[N]         1 only on the last frame of an episode
 *       player_x.npy   float32[N]       avatar x in pixels (NaN if absent)
 *       player_y.npy   float32[N]       avatar y in pixels (NaN if absent)
 * </pre>
 *
 * <p>Frames from different episodes within the same invocation accumulate into
 * the same shard until the {@code --chunk-size} threshold is hit; episode
 * boundaries within a shard are recoverable via {@code done.npy}.
 *
 * <p>Usage:
 * <pre>
 *   java -cp &lt;gvgai-build&gt; tracks.singlePlayer.ascii.RunAsciiCollectionMCTS \
 *     --gvgai-root &lt;path&gt; \
 *     --repo-root  &lt;path&gt; \
 *     --game       aliens \
 *     --out        data/transitions/train/aliens \
 *     --frames     500000 \
 *     --levels     0,1,2,3,4 \
 *     --mcts-ms    40 \
 *     --chunk-size 5000 \
 *     --seed       42
 * </pre>
 */
public final class RunAsciiCollectionMCTS {

	public static final int NUM_GVGAI_ACTIONS = Types.ACTIONS.values().length;
	private static final String CONTROLLER_NAME = "tracks.singlePlayer.ascii.RecordingMCTSAgent";

	public static void main(String[] args) throws Exception {
		Args a = Args.parse(args);
		AsciiGridExtractor extractor = AsciiGridExtractor.fromJson(a.mappingPath);
		RecordingMCTSAgent.extractorForNextEpisode = extractor;
		RecordingMCTSAgent.mctsMillisForNextEpisode = a.mctsMs;

		Files.createDirectories(a.outDir);
		ShardAccumulator acc = new ShardAccumulator(a.outDir, a.chunkSize);
		Random rng = new Random(a.seed);

		int episode = 0;
		long totalFrames = 0L;
		long startNs = System.nanoTime();
		while (totalFrames < a.totalFrames) {
			int levelIdx = a.levels.get(episode % a.levels.size());
			Path levelPath = a.gvgaiRoot.resolve("examples/gridphysics/" + a.game + "_lvl" + levelIdx + ".txt");
			Path gamePath = a.gvgaiRoot.resolve("examples/gridphysics/" + a.game + ".txt");
			int episodeSeed = rng.nextInt(Integer.MAX_VALUE);

			ArcadeMachine.runOneGame(
					gamePath.toString(),
					levelPath.toString(),
					/* visuals */ false,
					CONTROLLER_NAME,
					/* actionFile */ null,
					episodeSeed,
					/* playerID */ 0);

			RecordingMCTSAgent agent = RecordingMCTSAgent.lastInstance;
			if (agent == null) {
				System.err.println("warning: no agent instance after episode " + episode + "; skipping");
				episode++;
				continue;
			}
			agent.markEpisodeFinished();
			ArrayList<RecordingMCTSAgent.Frame> frames = agent.drainEpisode();
			acc.addEpisode(a.game, frames, extractor.getNativeH(), extractor.getNativeW());
			totalFrames += frames.size();
			episode++;

			double elapsedS = (System.nanoTime() - startNs) / 1e9;
			System.out.printf(Locale.ROOT,
					"[%s lvl%d] episode=%d frames=%d total=%d / %d (%.1f fps, %.1fs)%n",
					a.game, levelIdx, episode, frames.size(), totalFrames, a.totalFrames,
					totalFrames / Math.max(elapsedS, 1e-9), elapsedS);
		}
		acc.flush(a.game, extractor.getNativeH(), extractor.getNativeW());
		System.out.printf(Locale.ROOT,
				"done: %d frames across %d episodes -> %s%n",
				totalFrames, episode, a.outDir);
	}

	/** Parsed command-line arguments. */
	private static final class Args {
		Path gvgaiRoot;
		Path repoRoot;
		String game;
		Path mappingPath;
		Path outDir;
		List<Integer> levels;
		long totalFrames;
		long mctsMs;
		int chunkSize;
		long seed;

		static Args parse(String[] argv) {
			Map<String, String> flags = new LinkedHashMap<>();
			for (int i = 0; i < argv.length; i++) {
				String k = argv[i];
				if (!k.startsWith("--"))
					throw new IllegalArgumentException("bad arg: " + k);
				if (i + 1 >= argv.length)
					throw new IllegalArgumentException("missing value for " + k);
				flags.put(k, argv[++i]);
			}
			Args a = new Args();
			a.gvgaiRoot = Paths.get(require(flags, "--gvgai-root")).toAbsolutePath().normalize();
			a.repoRoot = Paths.get(require(flags, "--repo-root")).toAbsolutePath().normalize();
			a.game = require(flags, "--game");
			a.outDir = Paths.get(require(flags, "--out")).toAbsolutePath().normalize();
			a.totalFrames = Long.parseLong(flags.getOrDefault("--frames", "100000"));
			a.mctsMs = Long.parseLong(flags.getOrDefault("--mcts-ms", "40"));
			a.chunkSize = Integer.parseInt(flags.getOrDefault("--chunk-size", "5000"));
			a.seed = Long.parseLong(flags.getOrDefault("--seed", "42"));
			a.levels = parseLevels(flags.getOrDefault("--levels", "0,1,2,3,4"));
			String mapping = flags.get("--mapping");
			a.mappingPath = mapping != null
					? Paths.get(mapping).toAbsolutePath().normalize()
					: a.repoRoot.resolve("world_model/ascii/mappings/" + a.game + ".json");
			if (!Files.isRegularFile(a.mappingPath))
				throw new IllegalArgumentException("mapping not found: " + a.mappingPath);
			return a;
		}

		private static String require(Map<String, String> flags, String key) {
			String v = flags.get(key);
			if (v == null) throw new IllegalArgumentException("required: " + key);
			return v;
		}

		private static List<Integer> parseLevels(String csv) {
			List<Integer> out = new ArrayList<>();
			for (String tok : csv.split(",")) {
				String t = tok.trim();
				if (!t.isEmpty()) out.add(Integer.parseInt(t));
			}
			if (out.isEmpty()) throw new IllegalArgumentException("--levels is empty");
			return out;
		}
	}

	/**
	 * Buffers frames across episodes and flushes a shard directory every
	 * {@code chunkSize} frames, matching the layout {@link NpyWriter} and the
	 * Python {@code AllAsciiFramesDataset} expect.
	 */
	private static final class ShardAccumulator {
		private final Path outDir;
		private final int chunkSize;
		private final ArrayList<RecordingMCTSAgent.Frame> buf = new ArrayList<>();
		private int nextShardIdx;

		ShardAccumulator(Path outDir, int chunkSize) throws IOException {
			this.outDir = outDir;
			this.chunkSize = chunkSize;
			this.nextShardIdx = nextFreeShardIndex(outDir);
		}

		void addEpisode(String game, ArrayList<RecordingMCTSAgent.Frame> frames, int h, int w) throws IOException {
			buf.addAll(frames);
			while (buf.size() >= chunkSize) {
				List<RecordingMCTSAgent.Frame> chunk = new ArrayList<>(buf.subList(0, chunkSize));
				buf.subList(0, chunkSize).clear();
				writeShard(game, chunk, h, w);
			}
		}

		void flush(String game, int h, int w) throws IOException {
			if (buf.isEmpty()) return;
			List<RecordingMCTSAgent.Frame> chunk = new ArrayList<>(buf);
			buf.clear();
			writeShard(game, chunk, h, w);
		}

		private void writeShard(String game, List<RecordingMCTSAgent.Frame> chunk, int h, int w) throws IOException {
			int n = chunk.size();
			byte[] obs = new byte[n * h * w];
			long[] action = new long[n];
			float[] reward = new float[n];
			byte[] done = new byte[n];
			float[] px = new float[n];
			float[] py = new float[n];
			for (int i = 0; i < n; i++) {
				RecordingMCTSAgent.Frame f = chunk.get(i);
				System.arraycopy(f.asciiGrid, 0, obs, i * h * w, h * w);
				action[i] = f.action;
				reward[i] = f.reward;
				done[i] = f.done ? (byte) 1 : (byte) 0;
				px[i] = f.playerX;
				py[i] = f.playerY;
			}
			Path shardDir = outDir.resolve(String.format(Locale.ROOT, "shard_%05d", nextShardIdx++));
			Files.createDirectories(shardDir);
			NpyWriter.writeUint8(shardDir.resolve("obs.npy"), obs, n, h, w);
			NpyWriter.writeInt64(shardDir.resolve("action.npy"), action);
			NpyWriter.writeInt64Scalar(shardDir.resolve("n_actions.npy"), NUM_GVGAI_ACTIONS);
			NpyWriter.writeFloat32(shardDir.resolve("reward.npy"), reward);
			NpyWriter.writeUint8Flat(shardDir.resolve("done.npy"), done);
			NpyWriter.writeFloat32(shardDir.resolve("player_x.npy"), px);
			NpyWriter.writeFloat32(shardDir.resolve("player_y.npy"), py);
		}

		private static int nextFreeShardIndex(Path outDir) throws IOException {
			if (!Files.isDirectory(outDir)) return 0;
			int max = -1;
			try (java.util.stream.Stream<Path> s = Files.list(outDir)) {
				for (Path p : (Iterable<Path>) s::iterator) {
					String name = p.getFileName().toString();
					if (name.startsWith("shard_")) {
						try { max = Math.max(max, Integer.parseInt(name.substring(6))); }
						catch (NumberFormatException ignored) {}
					}
				}
			}
			return max + 1;
		}
	}

	private RunAsciiCollectionMCTS() {}
}
