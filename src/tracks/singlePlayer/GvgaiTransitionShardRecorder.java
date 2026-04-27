package tracks.singlePlayer;

import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicLong;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import ontology.Types;
import tools.Vector2d;
import core.game.Game;
import core.game.StateObservation;
import core.vgdl.SpriteGroup;
import core.vgdl.VGDLSprite;

/**
 * Buffers (obs, action, player_xy) per worker thread and writes shards with globally
 * unique indices.  Shard indices are allocated under a per-game-directory file lock so
 * multiple JVMs or threads can append to the same output folder without colliding.
 */
final class GvgaiTransitionShardRecorder {
	private static final int RGB_CHANNELS = 3;
	private static final Pattern SHARD_DIR_PATTERN = Pattern.compile("^shard_(\\d+)$");

	private final Path envDir;
	private final int chunkSize;
	private final int nActionsEnum;
	private final double scale;

	private final AtomicLong globalFrames;

	private Game game;

	private final ArrayList<byte[]> obsRows = new ArrayList<>();
	private final ArrayList<Long> actions = new ArrayList<>();
	private final ArrayList<Long> restarted = new ArrayList<>();
	private final ArrayList<Float> playerX = new ArrayList<>();
	private final ArrayList<Float> playerY = new ArrayList<>();

	private int nativeW = -1;
	private int nativeH = -1;
	private int outW = -1;
	private int outH = -1;
	private BufferedImage offscreen;
	/** Reused final-size buffer when {@code scale < 1}; avoids allocating every frame. */
	private BufferedImage scaled;

	GvgaiTransitionShardRecorder(Path outputRoot, String envStem, int chunkSize,
			Game game, double scale, AtomicLong globalFrames) {
		this.envDir = outputRoot.resolve(envStem);
		this.chunkSize = Math.max(1, chunkSize);
		this.nActionsEnum = Types.ACTIONS.values().length;
		this.game = game;
		this.scale = Math.min(1.0, Math.max(0.01, scale));
		this.globalFrames = globalFrames;
	}

	void setGame(Game game) {
		this.game = game;
	}

	private byte[] pendingFrame;

	void captureFrame() {
		pendingFrame = renderPixels();
	}

	void commitFrame(StateObservation state, Types.ACTIONS action, boolean isRestartedStep) throws IOException {
		if (pendingFrame == null)
			return;
		if (outW > 0 && outH > 0) {
			int expected = outW * outH * RGB_CHANNELS;
			if (pendingFrame.length != expected) {
				throw new IOException("Non-RGB frame payload size: got " + pendingFrame.length
						+ ", expected " + expected + " (" + outW + "x" + outH + "x" + RGB_CHANNELS + ")");
			}
		}
		obsRows.add(pendingFrame);
		pendingFrame = null;
		actions.add((long) action.ordinal());
		restarted.add(isRestartedStep ? 1L : 0L);

		Vector2d pos = state.getAvatarPosition();
		if (pos == null || pos.equals(Types.NIL) || state.isGameOver()) {
			playerX.add(Float.NaN);
			playerY.add(Float.NaN);
		} else {
			playerX.add((float) pos.x);
			playerY.add((float) pos.y);
		}

		if (obsRows.size() >= chunkSize)
			flushShard();
	}

	void close() throws IOException {
		flushShard();
	}

	long getGlobalFrames() {
		return globalFrames.get();
	}

	int getImageWidth() {
		ensureOffscreen();
		return outW;
	}

	int getImageHeight() {
		ensureOffscreen();
		return outH;
	}

	int getNativeWidth() {
		ensureOffscreen();
		return nativeW;
	}

	int getNativeHeight() {
		ensureOffscreen();
		return nativeH;
	}

	private void ensureOffscreen() {
		if (offscreen != null)
			return;
		Dimension screen = game.getScreenSize();
		nativeW = screen.width;
		nativeH = screen.height;
		outW = Math.max(1, (int) Math.round(nativeW * scale));
		outH = Math.max(1, (int) Math.round(nativeH * scale));
		offscreen = new BufferedImage(nativeW, nativeH, BufferedImage.TYPE_INT_RGB);
		if (outW != nativeW || outH != nativeH)
			scaled = new BufferedImage(outW, outH, BufferedImage.TYPE_INT_RGB);
	}

	/** Hints for scaling down: bicubic + quality color path; dither off reduces noisy speckle on flat fills. */
	private static void applyHighQualityScaleHints(Graphics2D g) {
		g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
		g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
		g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		g.setRenderingHint(RenderingHints.KEY_ALPHA_INTERPOLATION,
				RenderingHints.VALUE_ALPHA_INTERPOLATION_QUALITY);
		g.setRenderingHint(RenderingHints.KEY_COLOR_RENDERING, RenderingHints.VALUE_COLOR_RENDER_QUALITY);
		g.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_PURE);
		g.setRenderingHint(RenderingHints.KEY_DITHERING, RenderingHints.VALUE_DITHER_DISABLE);
		g.setRenderingHint(RenderingHints.KEY_FRACTIONALMETRICS, RenderingHints.VALUE_FRACTIONALMETRICS_OFF);
	}

	/**
	 * Downscale into {@code dest} (size = target) with clearer results than one huge shrink: when shrinking
	 * by more than ~2×, halve repeatedly with bicubic, then bicubic into {@code dest}. {@code full} is not
	 * flushed; temporary chain images are flushed.
	 */
	private static void downscaleDrawToDest(BufferedImage full, BufferedImage dest) {
		final int tw = dest.getWidth();
		final int th = dest.getHeight();
		int w = full.getWidth();
		int h = full.getHeight();
		if (w == tw && h == th) {
			Graphics2D g0 = dest.createGraphics();
			applyHighQualityScaleHints(g0);
			g0.drawImage(full, 0, 0, null);
			g0.dispose();
			return;
		}
		BufferedImage cur = full;
		boolean curIsOwned = false;
		while (w > tw * 2 || h > th * 2) {
			int nw = Math.max(tw, (w + 1) / 2);
			int nh = Math.max(th, (h + 1) / 2);
			BufferedImage step = new BufferedImage(nw, nh, BufferedImage.TYPE_INT_RGB);
			Graphics2D g = step.createGraphics();
			applyHighQualityScaleHints(g);
			g.drawImage(cur, 0, 0, nw, nh, null);
			g.dispose();
			if (curIsOwned)
				cur.flush();
			cur = step;
			curIsOwned = true;
			w = nw;
			h = nh;
		}
		Graphics2D g2 = dest.createGraphics();
		applyHighQualityScaleHints(g2);
		g2.drawImage(cur, 0, 0, tw, th, null);
		g2.dispose();
		if (curIsOwned)
			cur.flush();
	}

	private byte[] renderPixels() {
		ensureOffscreen();

		Graphics2D g = offscreen.createGraphics();
		g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		g.setColor(Types.BLACK);
		g.fillRect(0, 0, nativeW, nativeH);
		try {
			int[] spriteOrder = game.getSpriteOrder();
			SpriteGroup[] groups = game.getSpriteGroups();
			if (groups != null) {
				for (int typeInt : spriteOrder) {
					if (groups[typeInt] != null) {
						ArrayList<VGDLSprite> sprites = groups[typeInt].getSprites();
						for (VGDLSprite sp : sprites) {
							if (sp != null)
								sp.draw(g, game);
						}
					}
				}
			}
		} catch (Exception ignored) {
		}
		g.dispose();

		final BufferedImage src;
		if (outW == nativeW && outH == nativeH)
			src = offscreen;
		else {
			downscaleDrawToDest(offscreen, scaled);
			src = scaled;
		}

		int[] argb = ((DataBufferInt) src.getRaster().getDataBuffer()).getData();
		byte[] rgb = new byte[outH * outW * RGB_CHANNELS];
		int p = 0;
		for (int pixel : argb) {
			rgb[p++] = (byte) ((pixel >> 16) & 0xff);
			rgb[p++] = (byte) ((pixel >> 8) & 0xff);
			rgb[p++] = (byte) (pixel & 0xff);
		}
		return rgb;
	}

	/**
	 * Reserves the next shard directory name under {@code envDir} using an exclusive
	 * file lock so separate processes never pick the same {@code shard_%05d}.
	 */
	private static Path reserveNextShardDir(Path envDir) throws IOException {
		Files.createDirectories(envDir);
		Path lockPath = envDir.resolve(".shard_alloc.lock");
		try (FileChannel ch = FileChannel.open(lockPath, StandardOpenOption.CREATE, StandardOpenOption.READ,
				StandardOpenOption.WRITE); FileLock ignored = ch.lock()) {
			int maxIdx = -1;
			try (DirectoryStream<Path> stream = Files.newDirectoryStream(envDir, "shard_*")) {
				for (Path p : stream) {
					String name = p.getFileName().toString();
					Matcher m = SHARD_DIR_PATTERN.matcher(name);
					if (!m.matches())
						continue;
					int v = Integer.parseInt(m.group(1));
					if (v > maxIdx)
						maxIdx = v;
				}
			}
			int idx = maxIdx + 1;
			Path shard = envDir.resolve(String.format("shard_%05d", idx));
			Files.createDirectories(shard);
			return shard;
		}
	}

	private void flushShard() throws IOException {
		if (obsRows.isEmpty())
			return;
		if (outW < 0)
			return;

		int n = obsRows.size();
		int plane = outH * outW * RGB_CHANNELS;
		byte[] obsFlat = new byte[n * plane];
		for (int i = 0; i < n; i++) {
			if (obsRows.get(i).length != plane) {
				throw new IOException("Shard contains non-RGB frame at row " + i + ": got "
						+ obsRows.get(i).length + ", expected " + plane + " (" + outW + "x" + outH + "x" + RGB_CHANNELS + ")");
			}
			System.arraycopy(obsRows.get(i), 0, obsFlat, i * plane, plane);
		}
		long[] actArr = new long[actions.size()];
		for (int i = 0; i < actArr.length; i++)
			actArr[i] = actions.get(i);
		long[] restartedArr = new long[restarted.size()];
		for (int i = 0; i < restartedArr.length; i++)
			restartedArr[i] = restarted.get(i);
		float[] px = new float[playerX.size()];
		float[] py = new float[playerY.size()];
		for (int i = 0; i < px.length; i++) {
			px[i] = playerX.get(i);
			py[i] = playerY.get(i);
		}

		Path shard = reserveNextShardDir(envDir);

		GvgaiNpyWriter.saveUint8Array4D(shard.resolve("obs.npy"), obsFlat, n, outH, outW, RGB_CHANNELS);
		GvgaiNpyWriter.saveInt64Vector(shard.resolve("action.npy"), actArr);
		GvgaiNpyWriter.saveInt64Vector(shard.resolve("restarted.npy"), restartedArr);
		GvgaiNpyWriter.saveInt64Scalar(shard.resolve("n_actions.npy"), (long) nActionsEnum);
		GvgaiNpyWriter.saveFloat32Vector(shard.resolve("player_x.npy"), px);
		GvgaiNpyWriter.saveFloat32Vector(shard.resolve("player_y.npy"), py);

		obsRows.clear();
		actions.clear();
		restarted.clear();
		playerX.clear();
		playerY.clear();
	}
}
