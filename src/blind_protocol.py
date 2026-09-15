import struct

MAGIC = b"AE"
VERSION = 1
BLOCK_SIZE = 8
HEADER_BITS = 64
HEADER_BYTES = 8


def text_to_bits(text):
    data = text.encode("utf-8")
    bits = []

    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)

    return bits


def bits_to_bytes(bits):
    if len(bits) % 8 != 0:
        raise ValueError("Bit stream length must be divisible by 8")

    data = bytearray()

    for i in range(0, len(bits), 8):
        value = 0

        for bit in bits[i:i + 8]:
            value = (value << 1) | int(bit)

        data.append(value)

    return bytes(data)


def bits_to_text(bits):
    data = bits_to_bytes(bits)
    return data.decode("utf-8", errors="replace")


def create_header(payload_bits):
    """
    8-byte header:

    Bytes 0-1 : Magic = AE
    Byte  2   : Version
    Byte  3   : Block size
    Bytes 4-7 : Payload bit length
    """

    if payload_bits < 0 or payload_bits > 0xFFFFFFFF:
        raise ValueError("Payload too large")

    return struct.pack(
        ">2sBBI",
        MAGIC,
        VERSION,
        BLOCK_SIZE,
        payload_bits
    )


def header_to_bits(header):
    bits = []

    for byte in header:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)

    return bits


def bits_to_header(bits):
    if len(bits) != HEADER_BITS:
        raise ValueError("Invalid header length")

    data = bits_to_bytes(bits)

    if len(data) != HEADER_BYTES:
        raise ValueError("Invalid header")

    magic, version, block_size, payload_bits = struct.unpack(
        ">2sBBI",
        data
    )

    if magic != MAGIC:
        raise ValueError(
            "Invalid stego image: AI-EPVD header not found"
        )

    if version != VERSION:
        raise ValueError(
            f"Unsupported AI-EPVD protocol version: {version}"
        )

    if block_size != BLOCK_SIZE:
        raise ValueError(
            f"Unsupported block size: {block_size}"
        )

    return {
        "version": version,
        "block_size": block_size,
        "payload_bits": payload_bits
    }


def set_lsb(value, bit):
    return (int(value) & 0xFE) | int(bit)


def get_lsb(value):
    return int(value) & 1


def write_header_to_block(block, payload_bits):
    header = create_header(payload_bits)
    bits = header_to_bits(header)

    flat = block.reshape(-1).copy()

    for i, bit in enumerate(bits):
        flat[i] = set_lsb(flat[i], bit)

    return flat.reshape(block.shape)


def read_header_from_block(block):
    flat = block.reshape(-1)

    bits = [
        get_lsb(flat[i])
        for i in range(HEADER_BITS)
    ]

    return bits_to_header(bits)


def write_allocation_tag(block, requested_bits):
    """
    requested_bits:
        5 -> tag 00
        6 -> tag 01
        7 -> tag 10
    """

    if requested_bits not in (5, 6, 7):
        raise ValueError("requested_bits must be 5, 6 or 7")

    tag = requested_bits - 5

    bit0 = (tag >> 1) & 1
    bit1 = tag & 1

    block = block.copy()

    block[0, 0] = set_lsb(block[0, 0], bit0)
    block[0, 1] = set_lsb(block[0, 1], bit1)

    return block


def read_allocation_tag(block):
    bit0 = get_lsb(block[0, 0])
    bit1 = get_lsb(block[0, 1])

    tag = (bit0 << 1) | bit1

    if tag > 2:
        raise ValueError("Invalid allocation tag")

    return tag + 5
