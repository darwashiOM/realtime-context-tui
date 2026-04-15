import Foundation
import Darwin

enum SocketServerError: Error {
    case cannotCreateSocket
    case cannotBind(errno: Int32)
    case cannotListen(errno: Int32)
}

final class SocketServer {
    private let path: String
    private var listenFd: Int32 = -1
    private var clients: [Int32] = []
    private let clientsLock = NSLock()
    private let acceptQueue = DispatchQueue(label: "audio-tap.socket.accept")
    private let writeQueue = DispatchQueue(label: "audio-tap.socket.write", qos: .userInitiated)
    private var running = false

    init(path: String) throws {
        self.path = path
    }

    func start() throws {
        _ = unlink(path)  // clean up stale socket file if present

        listenFd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard listenFd >= 0 else { throw SocketServerError.cannotCreateSocket }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathCapacity = MemoryLayout.size(ofValue: addr.sun_path)
        _ = path.withCString { src in
            withUnsafeMutablePointer(to: &addr.sun_path) {
                $0.withMemoryRebound(to: CChar.self, capacity: pathCapacity) { dst in
                    strncpy(dst, src, pathCapacity - 1)
                }
            }
        }
        let addrLen = socklen_t(MemoryLayout<sockaddr_un>.size)

        let bindResult = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                Darwin.bind(listenFd, sa, addrLen)
            }
        }
        guard bindResult == 0 else {
            let e = errno
            close(listenFd)
            listenFd = -1
            throw SocketServerError.cannotBind(errno: e)
        }

        guard Darwin.listen(listenFd, 8) == 0 else {
            let e = errno
            close(listenFd)
            listenFd = -1
            throw SocketServerError.cannotListen(errno: e)
        }

        running = true
        acceptQueue.async { [weak self] in self?.acceptLoop() }
    }

    private func acceptLoop() {
        while running {
            var clientAddr = sockaddr_un()
            var clientAddrLen = socklen_t(MemoryLayout<sockaddr_un>.size)
            let clientFd = withUnsafeMutablePointer(to: &clientAddr) { ptr in
                ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                    Darwin.accept(listenFd, sa, &clientAddrLen)
                }
            }
            if clientFd < 0 {
                if !running { return }
                continue
            }
            // Enable non-blocking writes so a slow client can't stall broadcasts.
            var flags = fcntl(clientFd, F_GETFL, 0)
            flags |= O_NONBLOCK
            _ = fcntl(clientFd, F_SETFL, flags)

            clientsLock.lock()
            clients.append(clientFd)
            clientsLock.unlock()
        }
    }

    /// Broadcast frame bytes to every connected client. Dead clients are dropped.
    func broadcast(_ data: Data) {
        writeQueue.async { [weak self] in
            guard let self = self else { return }
            self.clientsLock.lock()
            let snapshot = self.clients
            self.clientsLock.unlock()

            var dead: [Int32] = []
            for fd in snapshot {
                let written = data.withUnsafeBytes { raw -> ssize_t in
                    guard let base = raw.baseAddress else { return 0 }
                    return Darwin.write(fd, base, data.count)
                }
                if written < 0 {
                    // EPIPE, EBADF, etc. — drop this client.
                    dead.append(fd)
                }
            }
            if !dead.isEmpty {
                self.clientsLock.lock()
                self.clients.removeAll { dead.contains($0) }
                self.clientsLock.unlock()
                for fd in dead { close(fd) }
            }
        }
    }

    func stop() {
        running = false
        if listenFd >= 0 {
            close(listenFd)
            listenFd = -1
        }
        clientsLock.lock()
        for fd in clients { close(fd) }
        clients.removeAll()
        clientsLock.unlock()
        _ = unlink(path)
    }
}
