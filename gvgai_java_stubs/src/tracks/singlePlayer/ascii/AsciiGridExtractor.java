package tracks.singlePlayer.ascii;

import core.game.Observation;
import core.game.StateObservation;
import core.vgdl.VGDLRegistry;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Turns a {@link StateObservation} into a ``uint8[h][w]`` ASCII grid matching
 * the per-game JSON mapping in
 * ``<repo>/world_model/ascii/mappings/<game>.json``.
 *
 * <p>The JSON schema (kept in sync with ``world_model/ascii/tokenizer.py`` and
 * the VAE training path) is:
 * <pre>
 * {
 *   "game": "aliens",
 *   "native_h": 11,
 *   "native_w": 30,
 *   "top_sprite_priority": ["avatar", "sam", "bomb", ..., "background"],
 *   "sprite_to_char": { "avatar": "A", "sam": "|", ..., "background": "." }
 * }
 * </pre>
 *
 * <p>Sprite names in {@code top_sprite_priority} / {@code sprite_to_char} are
 * VGDL stype identifiers, which this class resolves to itype ints via
 * {@link VGDLRegistry#getRegisteredSpriteValue(String)} the first time a grid
 * is extracted. Keys with no registered itype are silently skipped (typical
 * for abstract VGDL parent classes like {@code cloud} or {@code layers}).
 *
 * <p>Parsing is done with a tiny handwritten JSON reader (no Gson dep) because
 * the schema is flat and the file is checked in.
 */
public final class AsciiGridExtractor {

	public static final byte BACKGROUND_BYTE = (byte) '.';
	public static final String BACKGROUND_KEY = "background";

	private final int nativeH;
	private final int nativeW;
	private final List<String> priorityOrder;
	private final Map<String, Byte> nameToChar;
	/** Lazily populated on first extract() once {@link VGDLRegistry} is initialised. */
	private int[] priorityItypes;
	private byte[] priorityChars;
	private byte backgroundChar;

	private AsciiGridExtractor(
			int nativeH, int nativeW,
			List<String> priorityOrder, Map<String, Byte> nameToChar) {
		this.nativeH = nativeH;
		this.nativeW = nativeW;
		this.priorityOrder = priorityOrder;
		this.nameToChar = nameToChar;
		Byte bg = nameToChar.get(BACKGROUND_KEY);
		this.backgroundChar = bg != null ? bg : BACKGROUND_BYTE;
	}

	public int getNativeH() { return nativeH; }
	public int getNativeW() { return nativeW; }

	/**
	 * Parse a mapping JSON at {@code jsonPath} and build an extractor.
	 * @throws IOException if the file is missing or the JSON is malformed.
	 */
	public static AsciiGridExtractor fromJson(Path jsonPath) throws IOException {
		String src = new String(Files.readAllBytes(jsonPath), StandardCharsets.UTF_8);
		Map<String, Object> root = parseObject(src);
		int h = toInt(root.get("native_h"));
		int w = toInt(root.get("native_w"));
		@SuppressWarnings("unchecked")
		List<String> prio = (List<String>) root.get("top_sprite_priority");
		@SuppressWarnings("unchecked")
		Map<String, Object> mapObj = (Map<String, Object>) root.get("sprite_to_char");
		Map<String, Byte> nameToChar = new LinkedHashMap<>();
		for (Map.Entry<String, Object> e : mapObj.entrySet()) {
			String v = (String) e.getValue();
			if (v == null || v.isEmpty())
				throw new IOException("empty char for sprite '" + e.getKey() + "' in " + jsonPath);
			nameToChar.put(e.getKey(), (byte) v.charAt(0));
		}
		return new AsciiGridExtractor(h, w, prio, nameToChar);
	}

	/**
	 * Render {@code stateObs} into an {@code h * w} byte buffer (row-major).
	 *
	 * <p>Each cell is filled with the character of the highest-priority sprite
	 * whose itype is present at that grid cell; cells with no recognised
	 * sprite get {@link #BACKGROUND_KEY}'s char (fallback: ``.``).
	 */
	public byte[] extract(StateObservation stateObs) {
		resolveItypesOnce();
		ArrayList<Observation>[][] grid = stateObs.getObservationGrid();

		int cols = grid.length;
		int rows = cols > 0 ? grid[0].length : 0;
		int h = Math.min(rows, nativeH);
		int w = Math.min(cols, nativeW);

		byte[] out = new byte[nativeH * nativeW];
		java.util.Arrays.fill(out, backgroundChar);

		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				ArrayList<Observation> cell = grid[x][y];
				if (cell == null || cell.isEmpty()) continue;
				byte ch = pickCharForCell(cell);
				out[y * nativeW + x] = ch;
			}
		}
		return out;
	}

	private byte pickCharForCell(ArrayList<Observation> cell) {
		for (int i = 0; i < priorityItypes.length; i++) {
			int itype = priorityItypes[i];
			if (itype < 0) continue;
			for (int k = 0, n = cell.size(); k < n; k++) {
				if (cell.get(k).itype == itype) return priorityChars[i];
			}
		}
		return backgroundChar;
	}

	private void resolveItypesOnce() {
		if (priorityItypes != null) return;
		VGDLRegistry reg = VGDLRegistry.GetInstance();
		int n = priorityOrder.size();
		int[] itypes = new int[n];
		byte[] chars = new byte[n];
		for (int i = 0; i < n; i++) {
			String name = priorityOrder.get(i);
			Byte ch = nameToChar.get(name);
			itypes[i] = reg.getRegisteredSpriteValue(name);
			chars[i] = ch != null ? ch : backgroundChar;
		}
		this.priorityItypes = itypes;
		this.priorityChars = chars;
	}

	private static int toInt(Object o) {
		if (o instanceof Number) return ((Number) o).intValue();
		throw new IllegalArgumentException("expected number, got " + o);
	}

	/* ----------------------------- Tiny JSON reader -----------------------------
	 * Supports objects, arrays, strings, numbers, true/false/null. Good enough
	 * for our small mapping files and avoids a Gson build-classpath dependency.
	 */
	private static Map<String, Object> parseObject(String src) {
		JsonReader r = new JsonReader(src);
		r.skipWs();
		Object v = r.readValue();
		r.skipWs();
		if (r.pos != r.s.length())
			throw new IllegalArgumentException("trailing content at " + r.pos);
		if (!(v instanceof Map))
			throw new IllegalArgumentException("top-level JSON must be an object");
		@SuppressWarnings("unchecked")
		Map<String, Object> out = (Map<String, Object>) v;
		return out;
	}

	private static final class JsonReader {
		final String s;
		int pos;

		JsonReader(String s) { this.s = s; this.pos = 0; }

		Object readValue() {
			skipWs();
			char c = s.charAt(pos);
			if (c == '{') return readObject();
			if (c == '[') return readArray();
			if (c == '"') return readString();
			if (c == 't' || c == 'f') return readBool();
			if (c == 'n') { expect("null"); return null; }
			return readNumber();
		}

		Map<String, Object> readObject() {
			expect('{');
			Map<String, Object> m = new LinkedHashMap<>();
			skipWs();
			if (peek() == '}') { pos++; return m; }
			while (true) {
				skipWs();
				String k = readString();
				skipWs();
				expect(':');
				Object v = readValue();
				m.put(k, v);
				skipWs();
				char c = s.charAt(pos++);
				if (c == ',') continue;
				if (c == '}') return m;
				throw new IllegalArgumentException("expected , or } at " + (pos - 1));
			}
		}

		List<Object> readArray() {
			expect('[');
			List<Object> a = new ArrayList<>();
			skipWs();
			if (peek() == ']') { pos++; return a; }
			while (true) {
				a.add(readValue());
				skipWs();
				char c = s.charAt(pos++);
				if (c == ',') continue;
				if (c == ']') return a;
				throw new IllegalArgumentException("expected , or ] at " + (pos - 1));
			}
		}

		String readString() {
			expect('"');
			StringBuilder sb = new StringBuilder();
			while (true) {
				char c = s.charAt(pos++);
				if (c == '"') return sb.toString();
				if (c == '\\') {
					char e = s.charAt(pos++);
					switch (e) {
						case '"': sb.append('"'); break;
						case '\\': sb.append('\\'); break;
						case '/': sb.append('/'); break;
						case 'n': sb.append('\n'); break;
						case 't': sb.append('\t'); break;
						case 'r': sb.append('\r'); break;
						case 'b': sb.append('\b'); break;
						case 'f': sb.append('\f'); break;
						case 'u':
							sb.append((char) Integer.parseInt(s.substring(pos, pos + 4), 16));
							pos += 4;
							break;
						default: throw new IllegalArgumentException("bad escape \\" + e);
					}
				} else {
					sb.append(c);
				}
			}
		}

		Number readNumber() {
			int start = pos;
			if (peek() == '-') pos++;
			while (pos < s.length()) {
				char c = s.charAt(pos);
				if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') pos++;
				else break;
			}
			String lit = s.substring(start, pos);
			if (lit.indexOf('.') < 0 && lit.indexOf('e') < 0 && lit.indexOf('E') < 0)
				return Long.parseLong(lit);
			return Double.parseDouble(lit);
		}

		Boolean readBool() {
			if (peek() == 't') { expect("true"); return Boolean.TRUE; }
			expect("false"); return Boolean.FALSE;
		}

		void skipWs() {
			while (pos < s.length()) {
				char c = s.charAt(pos);
				if (c == ' ' || c == '\t' || c == '\n' || c == '\r') pos++; else return;
			}
		}

		char peek() { return s.charAt(pos); }

		void expect(char c) {
			if (s.charAt(pos) != c)
				throw new IllegalArgumentException("expected '" + c + "' at " + pos + " got '" + s.charAt(pos) + "'");
			pos++;
		}

		void expect(String tok) {
			for (int i = 0; i < tok.length(); i++) {
				if (s.charAt(pos + i) != tok.charAt(i))
					throw new IllegalArgumentException("expected " + tok + " at " + pos);
			}
			pos += tok.length();
		}
	}
}
