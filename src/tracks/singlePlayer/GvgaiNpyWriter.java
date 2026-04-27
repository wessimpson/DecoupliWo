package tracks.singlePlayer;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Minimal NumPy 1.0 .npy writer (no external deps) for uint8 and int64/float32 C-order arrays.
 */
final class GvgaiNpyWriter {

	private GvgaiNpyWriter() {
	}

	static void saveUint8Array4D(Path path, byte[] data, int n, int h, int w, int c) throws IOException {
		String dict = "{'descr': '<u1', 'fortran_order': False, 'shape': (" + n + ", " + h + ", " + w + ", " + c
				+ "), }";
		writeNpyV1(path, dict, data);
	}

	static void saveInt64Vector(Path path, long[] data) throws IOException {
		byte[] raw = new byte[data.length * 8];
		int p = 0;
		for (long v : data) {
			for (int b = 0; b < 8; b++)
				raw[p++] = (byte) (v >>> (8 * b));
		}
		String dict = "{'descr': '<i8', 'fortran_order': False, 'shape': (" + data.length + ",), }";
		writeNpyV1(path, dict, raw);
	}

	static void saveInt64Scalar(Path path, long value) throws IOException {
		byte[] raw = new byte[8];
		for (int b = 0; b < 8; b++)
			raw[b] = (byte) (value >>> (8 * b));
		String dict = "{'descr': '<i8', 'fortran_order': False, 'shape': (), }";
		writeNpyV1(path, dict, raw);
	}

	static void saveFloat32Vector(Path path, float[] data) throws IOException {
		byte[] raw = new byte[data.length * 4];
		int p = 0;
		for (float f : data) {
			int bits = Float.floatToRawIntBits(f);
			raw[p++] = (byte) bits;
			raw[p++] = (byte) (bits >>> 8);
			raw[p++] = (byte) (bits >>> 16);
			raw[p++] = (byte) (bits >>> 24);
		}
		String dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + data.length + ",), }";
		writeNpyV1(path, dict, raw);
	}

	/**
	 * NumPy 1.0 format (see {@code numpy.lib.format._wrap_header}): after the length field, bytes are
	 * {@code dict_ascii + (SPACE * padlen) + NEWLINE}. The declared header length is {@code dictLen + 1 + padlen}
	 * (the {@code +1} is the final newline). Padding makes {@code 10 + headerDeclaredLen} divisible by 64.
	 */
	private static final int ARRAY_ALIGN = 64;
	/** Magic (6) + version (2), matching numpy {@code MAGIC_LEN}. */
	private static final int MAGIC_LEN = 8;
	private static final int HEADER_LEN_FIELD = 2;

	private static void writeNpyV1(Path path, String dict, byte[] payload) throws IOException {
		byte[] headerDict = dict.getBytes(StandardCharsets.US_ASCII);
		// hlen counts dict bytes plus the final newline (numpy convention before adding space padding).
		int hlen = headerDict.length + 1;
		int padlen = ARRAY_ALIGN - ((MAGIC_LEN + HEADER_LEN_FIELD + hlen) % ARRAY_ALIGN);
		int declaredLen = hlen + padlen;
		if (declaredLen > 65535)
			throw new IOException("npy header too large");

		byte[] hb = new byte[headerDict.length + padlen + 1];
		System.arraycopy(headerDict, 0, hb, 0, headerDict.length);
		for (int i = 0; i < padlen; i++)
			hb[headerDict.length + i] = ' ';
		hb[hb.length - 1] = '\n';

		try (OutputStream out = Files.newOutputStream(path)) {
			out.write(new byte[] { (byte) 0x93, 'N', 'U', 'M', 'P', 'Y' });
			out.write(1);
			out.write(0);
			out.write(declaredLen & 0xff);
			out.write((declaredLen >>> 8) & 0xff);
			out.write(hb);
			out.write(payload);
		}
	}
}
