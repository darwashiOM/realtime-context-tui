import XCTest
@testable import AudioTap

final class FrameEncoderTests: XCTestCase {
    func testEncodesThemFrameWithCorrectHeader() {
        let pcm = Data([0x01, 0x02, 0x03, 0x04])  // 4 bytes dummy PCM
        let encoded = FrameEncoder.encode(streamTag: .them, timestampMs: 0x01020304, pcm: pcm)

        XCTAssertEqual(encoded.count, 9 + 4)
        XCTAssertEqual(encoded[0], 0x00)                                    // them = 0
        XCTAssertEqual(encoded.subdata(in: 1..<5), Data([0x01, 0x02, 0x03, 0x04]))  // BE timestamp
        XCTAssertEqual(encoded.subdata(in: 5..<9), Data([0x00, 0x00, 0x00, 0x04]))  // BE length
        XCTAssertEqual(encoded.subdata(in: 9..<13), pcm)                    // payload
    }

    func testEncodesMeFrameWithTagOne() {
        let encoded = FrameEncoder.encode(streamTag: .me, timestampMs: 0, pcm: Data())
        XCTAssertEqual(encoded[0], 0x01)
        XCTAssertEqual(encoded.count, 9)  // 9-byte header, zero-length payload
    }

    func testDecodeRoundTrip() throws {
        let pcm = Data((0..<640).map { UInt8($0 & 0xFF) })  // 640-byte synthetic PCM
        let encoded = FrameEncoder.encode(streamTag: .them, timestampMs: 12345, pcm: pcm)
        let decoded = try FrameEncoder.decode(encoded)

        XCTAssertEqual(decoded.streamTag, .them)
        XCTAssertEqual(decoded.timestampMs, 12345)
        XCTAssertEqual(decoded.pcm, pcm)
    }

    func testDecodeRejectsHeaderTooShort() {
        XCTAssertThrowsError(try FrameEncoder.decode(Data([0x00, 0x00, 0x00]))) { err in
            guard case FrameDecodeError.truncated = err else {
                return XCTFail("expected .truncated, got \(err)")
            }
        }
    }

    func testDecodeRejectsTruncatedPayload() {
        // header claims 4-byte payload, only 2 provided
        let bytes: [UInt8] = [0x00, 0,0,0,0, 0,0,0,4, 0xAA, 0xBB]
        XCTAssertThrowsError(try FrameEncoder.decode(Data(bytes))) { err in
            guard case FrameDecodeError.truncated = err else {
                return XCTFail("expected .truncated, got \(err)")
            }
        }
    }

    func testDecodeRejectsUnknownStreamTag() {
        let bytes: [UInt8] = [0xFE, 0,0,0,0, 0,0,0,0]
        XCTAssertThrowsError(try FrameEncoder.decode(Data(bytes))) { err in
            guard case FrameDecodeError.unknownStreamTag(let b) = err else {
                return XCTFail("expected .unknownStreamTag, got \(err)")
            }
            XCTAssertEqual(b, 0xFE)
        }
    }

    func testDecodeWorksOnSlicedDataWithNonZeroStartIndex() throws {
        let pcm = Data([0xAA, 0xBB, 0xCC])
        let encoded = FrameEncoder.encode(streamTag: .me, timestampMs: 7, pcm: pcm)
        // Prepend garbage and slice past it — slice retains startIndex > 0.
        var garbageThenFrame = Data([0x99, 0x99, 0x99, 0x99])
        garbageThenFrame.append(encoded)
        let slice = garbageThenFrame.subdata(in: 4..<garbageThenFrame.count)
        // Round-trip through Data() to keep the slice base intact in some APIs:
        // but subdata returns a fresh Data with startIndex 0. Force a true slice instead.
        let trueSlice = garbageThenFrame[4..<garbageThenFrame.count]
        XCTAssertNotEqual(trueSlice.startIndex, 0, "precondition: slice should have non-zero startIndex")

        let decodedFromSubdata = try FrameEncoder.decode(slice)
        XCTAssertEqual(decodedFromSubdata.pcm, pcm)

        let decodedFromSlice = try FrameEncoder.decode(trueSlice)
        XCTAssertEqual(decodedFromSlice.streamTag, .me)
        XCTAssertEqual(decodedFromSlice.timestampMs, 7)
        XCTAssertEqual(decodedFromSlice.pcm, pcm)
    }

    func testDecodeIgnoresTrailingBytes() throws {
        let pcm = Data([0x01, 0x02])
        var encoded = FrameEncoder.encode(streamTag: .them, timestampMs: 1, pcm: pcm)
        encoded.append(Data([0xDE, 0xAD, 0xBE, 0xEF]))  // trailing junk
        let decoded = try FrameEncoder.decode(encoded)
        XCTAssertEqual(decoded.pcm, pcm)
    }
}
