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
}
