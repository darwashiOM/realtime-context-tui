import Foundation

// MARK: - CLI arg parsing

let defaultSocketPath = "/tmp/rctx.sock"
let socketPath: String = {
    let args = CommandLine.arguments
    if let idx = args.firstIndex(of: "--socket-path"), idx + 1 < args.count {
        return args[idx + 1]
    }
    return defaultSocketPath
}()

fputs("audio-tap: starting, socket=\(socketPath)\n", stderr)

// MARK: - Main

do {
    let server = try SocketServer(path: socketPath)
    try server.start()

    // One Resampler per capture: Resampler is not thread-safe and each capture delivers
    // buffers on its own queue (ScreenCaptureKit sampleHandlerQueue for system,
    // AVAudioEngine's internal tap thread for mic).
    let systemResampler = try Resampler(targetSampleRate: 16000)
    let micResampler = try Resampler(targetSampleRate: 16000)

    let systemCapture = SystemAudioCapture(resampler: systemResampler) { pcm, ts in
        let frame = FrameEncoder.encode(streamTag: .them, timestampMs: ts, pcm: pcm)
        server.broadcast(frame)
    }
    let micCapture = MicCapture(resampler: micResampler) { pcm, ts in
        let frame = FrameEncoder.encode(streamTag: .me, timestampMs: ts, pcm: pcm)
        server.broadcast(frame)
    }

    // Start system audio capture (async — ScreenCaptureKit API is async-only).
    Task {
        do {
            try await systemCapture.start()
            fputs("audio-tap: system capture started\n", stderr)
        } catch {
            fputs("audio-tap: system capture FAILED: \(error)\n", stderr)
            fputs("       (grant Screen Recording in System Settings → Privacy & Security)\n", stderr)
        }
    }

    // Start mic capture (synchronous).
    do {
        try micCapture.start()
        fputs("audio-tap: mic capture started\n", stderr)
    } catch {
        fputs("audio-tap: mic capture FAILED: \(error)\n", stderr)
        fputs("       (grant Microphone permission in System Settings → Privacy & Security)\n", stderr)
    }

    // Clean shutdown on SIGINT/SIGTERM.
    //
    // We use DispatchSource signal sources rather than the plain `signal()` handler so
    // the handler body runs on a normal dispatch queue — NOT in async-signal-only
    // context. That lets us safely call `server.stop()` (which takes an NSLock and
    // calls `close`/`unlink`) before exiting, so the socket file is cleaned up.
    //
    // `signal(_:SIG_IGN)` is required first because DispatchSource only observes; the
    // default action (terminate) would otherwise race the source.
    let shutdownQueue = DispatchQueue(label: "audio-tap.shutdown")
    var signalSources: [DispatchSourceSignal] = []
    for sig in [SIGINT, SIGTERM] {
        signal(sig, SIG_IGN)
        let src = DispatchSource.makeSignalSource(signal: sig, queue: shutdownQueue)
        src.setEventHandler {
            fputs("audio-tap: got signal \(sig), exiting\n", stderr)
            Task {
                await systemCapture.stop()
                micCapture.stop()
                server.stop()
                exit(0)
            }
        }
        src.resume()
        signalSources.append(src)
    }

    RunLoop.main.run()
} catch {
    fputs("audio-tap: fatal startup error: \(error)\n", stderr)
    exit(1)
}
