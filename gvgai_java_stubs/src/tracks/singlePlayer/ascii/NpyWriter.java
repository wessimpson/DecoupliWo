package tracks.singlePlayer.ascii;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

/**
 * Minimal writer for the <a href="https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html">NumPy
 * .npy v1.0</a> format, sized down to the four dtypes the Python dataset layer
 * reads: {@code uint8}, {@code int64}, {@code float32}, and a scalar
 * {@code int64}.
 *
 * <p>The format is straightforward: a 10-byte prefix ({@code \x93NUMPY} +
 * version {@code \x01\x00} + little-endian {@code uint16} header-length),
 * an ASCII dict header padded with spaces to a 64-byte multiple and terminated
 * by a newline, then the raw data bytes in C order.
 */
public final class NpyWriter {

	private NpyWriter() {}

	/** Write ``uint8[N, H, W]`` from a flat {@code N*H*W} buffer. */
	public static void writeUint8(Path path, byte[] data, int n, int h, int w) throws IOException {
		if (data.length != (long) n * h * w)
			throw new IllegalArgumentException("data.length=" + data.length + " != n*h*w=" + (long) n * h * w);
		String header = "{'descr': '|u1', 'fortran_order': False, 'shape': (" + n + ", " + h + ", " + w + "), }";
		try (DataOutputStream out = open(path)) {
			writeHeader(out, header);
			out.write(data);
		}
	}

	/** Write ``uint8[N]`` from a flat buffer. */
	public static void writeUint8Flat(Path path, byte[] data) throws IOException {
		String header = "{'descr': '|u1', 'fortran_order': False, 'shape': (" + data.length + ",), }";
		try (DataOutputStream out = open(path)) {
			writeHeader(out, header);
			out.write(data);
		}
	}

	/** Write ``int64[N]`` (little-endian). */
	public static void writeInt64(Path path, long[] data) throws IOException {
		String header = "{'descr': '<i8', 'fortran_order': False, 'shape': (" + data.length + ",), }";
		try (DataOutputStream out = open(path)) {
			writeHeader(out, header);
			ByteBuffer bb = ByteBuffer.allocate(data.length * 8).order(ByteOrder.LITTLE_ENDIAN);
			for (long v : data) bb.putLong(v);
			out.write(bb.array());
		}
	}

	/** Write ``float32[N]`` (little-endian). */
	public static void writeFloat32(Path path, float[] data) throws IOException {
		String header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + data.length + ",), }";
		try (DataOutputStream out = open(path)) {
			writeHeader(out, header);
			ByteBuffer bb = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
			for (float v : data) bb.putFloat(v);
			out.write(bb.array());
		}
	}

	/** Write a 0-d ``int64`` scalar (matches ``np.save(path, np.array(K, dtype=np.int64))``). */
	public static void writeInt64Scalar(Path path, long value) throws IOException {
		String header = "{'descr': '<i8', 'fortran_order': False, 'shape': (), }";
		try (DataOutputStream out = open(path)) {
			writeHeader(out, header);
			ByteBuffer bb = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
			bb.putLong(value);
			out.write(bb.array());
		}
	}

	private static DataOutputStream open(Path path) throws IOException {
		return new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path.toFile())));
	}

	private static void writeHeader(DataOutputStream out, String headerText) throws IOException {
		byte[] magic = new byte[] { (byte) 0x93, 'N', 'U', 'M', 'P', 'Y' };
		out.write(magic);
		out.writeByte(1);
		out.writeByte(0);
		// Prefix so far = 10 bytes (6 magic + 2 version + 2 header-len).
		int base = 10 + headerText.length();
		int padded = (base + 1 + 63) & ~63;
		int padding = padded - base - 1;
		StringBuilder sb = new StringBuilder(headerText);
		for (int i = 0; i < padding; i++) sb.append(' ');
		sb.append('\n');
		byte[] headerBytes = sb.toString().getBytes(StandardCharsets.US_ASCII);
		int headerLen = headerBytes.length;
		if (headerLen > 0xFFFF)
			throw new IOException("header too long for v1 .npy: " + headerLen);
		out.writeByte(headerLen & 0xFF);
		out.writeByte((headerLen >> 8) & 0xFF);
		out.write(headerBytes);
	}
}
