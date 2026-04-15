import Foundation
import AVFoundation

enum MicCaptureError: Error {
    case engineStartFailed(Error)
}

/// Captures the default input device via AVAudioEngine's input tap, resamples to the
/// target rate (16 kHz mono Int16), and invokes `onFrame` per tap callback.
///
/// On macOS there is no `AVAudioSession` to configure (that's iOS-only); the engine uses
/// the system default input directly. Microphone permission must be granted in
/// System Settings → Privacy & Security; if not, `engine.start()` typically still succeeds
/// but the tap delivers silence.
///
/// Thread-safety: the tap callback runs on an AVAudioEngine-internal thread. The `Resampler`
/// is not thread-safe, so do not share one `MicCapture`'s resampler with another capture.
final class MicCapture {
    private let resampler: Resampler
    private let onFrame: (Data, UInt32) -> Void
    private let startTime = Date()
    private let engine = AVAudioEngine()

    init(resampler: Resampler, onFrame: @escaping (Data, UInt32) -> Void) {
        self.resampler = resampler
        self.onFrame = onFrame
    }

    func start() throws {
        let input = engine.inputNode
        let inputFormat = input.inputFormat(forBus: 0)

        // 20 ms at input's native rate. This is a *hint* to the tap — the actual buffer
        // size CoreAudio delivers can differ (often larger, e.g. 512/1024 frames at 44.1/48 kHz).
        // The Resampler handles variable input sizes fine.
        let tapFrames = AVAudioFrameCount(inputFormat.sampleRate * 0.020)

        input.installTap(onBus: 0, bufferSize: tapFrames, format: inputFormat) { [weak self] buffer, _ in
            guard let self = self else { return }
            do {
                let bytes = try self.resampler.resample(buffer)
                // ms since start, wrapped to 32 bits (see note in SystemAudioCapture).
                let ts = UInt32(Date().timeIntervalSince(self.startTime) * 1000) & 0xFFFFFFFF
                self.onFrame(bytes, ts)
            } catch {
                fputs("audio-tap: mic resample failed: \(error)\n", stderr)
            }
        }

        engine.prepare()
        do {
            try engine.start()
        } catch {
            throw MicCaptureError.engineStartFailed(error)
        }
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
    }
}
