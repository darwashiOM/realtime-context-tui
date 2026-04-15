import Foundation

enum StreamTag: UInt8 {
    case them = 0x00
    case me   = 0x01
}

enum FrameDecodeError: Error {
    case truncated
    case unknownStreamTag(UInt8)
}

struct Frame {
    let streamTag: StreamTag
    let timestampMs: UInt32
    let pcm: Data
}

enum FrameEncoder {
    /// Encode one frame: [tag:1][timestamp_be:4][len_be:4][pcm:N]
    static func encode(streamTag: StreamTag, timestampMs: UInt32, pcm: Data) -> Data {
        var out = Data(capacity: 9 + pcm.count)
        out.append(streamTag.rawValue)
        out.append(timestampMs.bigEndianBytes)
        out.append(UInt32(pcm.count).bigEndianBytes)
        out.append(pcm)
        return out
    }

    /// Decode one frame from the start of `data`. Trailing bytes beyond the frame are
    /// ignored (a streaming reader can peel one frame at a time). Returns `.truncated`
    /// if `data` is shorter than the header or the declared payload length.
    /// Works on `Data` slices with non-zero `startIndex`.
    static func decode(_ data: Data) throws -> Frame {
        guard data.count >= 9 else { throw FrameDecodeError.truncated }
        let base = data.startIndex
        let tagByte = data[base]
        guard let tag = StreamTag(rawValue: tagByte) else {
            throw FrameDecodeError.unknownStreamTag(tagByte)
        }
        let timestampMs = UInt32(bigEndianBytes: data.subdata(in: (base + 1)..<(base + 5)))
        let payloadLen = UInt32(bigEndianBytes: data.subdata(in: (base + 5)..<(base + 9)))
        // Bounds-check BEFORE computing payloadEnd to avoid Int overflow on huge payloadLen.
        guard data.count - 9 >= Int(payloadLen) else { throw FrameDecodeError.truncated }
        let payloadStart = base + 9
        let payloadEnd = payloadStart + Int(payloadLen)
        let pcm = data.subdata(in: payloadStart..<payloadEnd)
        return Frame(streamTag: tag, timestampMs: timestampMs, pcm: pcm)
    }
}

private extension UInt32 {
    var bigEndianBytes: Data {
        var be = self.bigEndian
        return Data(bytes: &be, count: 4)
    }

    init(bigEndianBytes data: Data) {
        var value: UInt32 = 0
        for byte in data { value = (value << 8) | UInt32(byte) }
        self = value
    }
}
