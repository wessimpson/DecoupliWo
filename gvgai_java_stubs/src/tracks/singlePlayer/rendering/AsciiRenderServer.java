package tracks.singlePlayer.rendering;

import core.game.Game;
import core.vgdl.VGDLFactory;
import core.vgdl.VGDLParser;
import core.vgdl.VGDLRegistry;
import core.vgdl.VGDLSprite;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * Tiny line-protocol TCP server that renders ASCII grids to PNG bytes using
 * GVGAI's own sprite draw pipeline.
 *
 * Protocol (ASCII, newline-delimited; PNG bytes are raw on the socket after
 * the length header):
 * <pre>
 *   Client -> "INIT &lt;vgdl_file_path&gt;\n"
 *   Server -> "OK &lt;screen_w&gt; &lt;screen_h&gt; &lt;block_size&gt;\n"
 *
 *   Client -> "RENDER &lt;seed&gt; &lt;num_rows&gt;\n" followed by that many raw ASCII rows,
 *             each terminated by "\n".
 *   Server -> "OK &lt;png_bytes_len&gt;\n" followed by that many PNG bytes.
 *
 *   Client -> "QUIT\n"
 *   Server -> "BYE\n" (then closes)
 * </pre>
 *
 * One connection at a time (single-threaded is plenty; rendering is fast).
 * The parsed {@link Game} is cached per VGDL file path across RENDER calls on
 * a connection; re-parsing only happens if the client re-issues INIT with a
 * different path.
 *
 * Usage:
 * <pre>
 *   java -cp out/production/gvgai tracks.singlePlayer.rendering.AsciiRenderServer &lt;port&gt;
 * </pre>
 */
public class AsciiRenderServer {

	private static final String OK = "OK";
	private static final String ERR = "ERR";
	private static final String BYE = "BYE";

	public static void main(String[] args) throws Exception {
		int port = args.length > 0 ? Integer.parseInt(args[0]) : 0;
		try (ServerSocket server = new ServerSocket(port)) {
			int boundPort = server.getLocalPort();
			System.out.println("AsciiRenderServer listening on " + boundPort);
			System.out.flush();
			while (true) {
				Socket client = server.accept();
				try {
					handleConnection(client);
				} catch (Exception e) {
					e.printStackTrace();
				} finally {
					try { client.close(); } catch (Exception ignored) {}
				}
			}
		}
	}

	private static void handleConnection(Socket client) throws Exception {
		BufferedReader in = new BufferedReader(new InputStreamReader(client.getInputStream()));
		DataOutputStream out = new DataOutputStream(client.getOutputStream());

		Map<String, Game> gameCache = new HashMap<>();
		Game currentGame = null;

		String line;
		while ((line = in.readLine()) != null) {
			if (line.startsWith("INIT ")) {
				String path = line.substring(5).trim();
				currentGame = gameCache.get(path);
				if (currentGame == null) {
					VGDLFactory.GetInstance().init();
					VGDLRegistry.GetInstance().init();
					currentGame = new VGDLParser().parseGame(path);
					gameCache.put(path, currentGame);
				}
				Dimension screen = currentGame.getScreenSize();
				writeLine(out, OK + " "
						+ (screen == null ? 0 : (int) screen.getWidth()) + " "
						+ (screen == null ? 0 : (int) screen.getHeight()) + " "
						+ currentGame.getBlockSize());
			} else if (line.startsWith("RENDER ")) {
				if (currentGame == null) {
					writeLine(out, ERR + " RENDER_BEFORE_INIT");
					continue;
				}
				String[] parts = line.split("\\s+");
				int seed = Integer.parseInt(parts[1]);
				int rows = Integer.parseInt(parts[2]);
				String[] levelLines = new String[rows];
				for (int i = 0; i < rows; i++) {
					String rowLine = in.readLine();
					if (rowLine == null) {
						writeLine(out, ERR + " UNEXPECTED_EOF");
						return;
					}
					levelLines[i] = rowLine;
				}
				try {
					byte[] png = renderLevel(currentGame, levelLines, seed);
					writeLine(out, OK + " " + png.length);
					out.write(png);
					out.flush();
				} catch (Exception e) {
					writeLine(out, ERR + " " + e.getClass().getSimpleName() + ": " + e.getMessage());
				}
			} else if (line.equals("QUIT")) {
				writeLine(out, BYE);
				return;
			} else if (line.length() > 0) {
				writeLine(out, ERR + " UNKNOWN_COMMAND");
			}
		}
	}

	private static void writeLine(DataOutputStream out, String s) throws Exception {
		out.write((s + "\n").getBytes("UTF-8"));
		out.flush();
	}

	private static byte[] renderLevel(Game game, String[] levelLines, int seed) throws Exception {
		game.buildStringLevel(levelLines, seed);

		Dimension screen = game.getScreenSize();
		int w = (int) screen.getWidth();
		int h = (int) screen.getHeight();
		BufferedImage bi = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
		Graphics2D g = bi.createGraphics();
		try {
			g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			g.setColor(Color.BLACK);
			g.fillRect(0, 0, w, h);
			int[] order = game.getSpriteOrder();
			for (int itype : order) {
				ArrayList<VGDLSprite> sprites = game.getSprites(itype);
				if (sprites == null) continue;
				for (VGDLSprite sp : sprites) {
					if (sp != null) sp.draw(g, game);
				}
			}
		} finally {
			g.dispose();
		}

		ByteArrayOutputStream buf = new ByteArrayOutputStream();
		ImageIO.write(bi, "png", buf);
		return buf.toByteArray();
	}
}
