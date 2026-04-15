import XCTest
@testable import AudioTap

final class SocketServerTests: XCTestCase {
    func testClientReceivesBroadcastedFrame() throws {
        let sockPath = NSTemporaryDirectory() + "audio-tap-test-\(UUID().uuidString).sock"
        defer { try? FileManager.default.removeItem(atPath: sockPath) }

        let server = try SocketServer(path: sockPath)
        try server.start()
        defer { server.stop() }

        // Connect a raw BSD socket as a client.
        let client = socket(AF_UNIX, SOCK_STREAM, 0)
        XCTAssertGreaterThanOrEqual(client, 0)
        defer { close(client) }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathCapacity = MemoryLayout.size(ofValue: addr.sun_path)
        _ = sockPath.withCString { src in
            withUnsafeMutablePointer(to: &addr.sun_path) {
                $0.withMemoryRebound(to: CChar.self, capacity: pathCapacity) { dst in
                    strncpy(dst, src, pathCapacity - 1)
                }
            }
        }
        let addrLen = socklen_t(MemoryLayout<sockaddr_un>.size)
        let connectResult = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                Darwin.connect(client, sa, addrLen)
            }
        }
        XCTAssertEqual(connectResult, 0, "connect() failed: errno=\(errno)")

        // Give the server a moment to register us.
        usleep(50_000)

        // Broadcast a known frame.
        let frame = FrameEncoder.encode(streamTag: .me, timestampMs: 42, pcm: Data([0xAA, 0xBB]))
        server.broadcast(frame)

        // Read exactly frame.count bytes from the client.
        var received = Data()
        let deadline = Date().addingTimeInterval(2.0)
        var buffer = [UInt8](repeating: 0, count: 256)
        while received.count < frame.count && Date() < deadline {
            let n = read(client, &buffer, buffer.count)
            if n > 0 { received.append(buffer, count: n) }
            else if n == 0 { break }
            else if errno != EAGAIN { break }
        }
        XCTAssertEqual(received, frame)
    }
}
