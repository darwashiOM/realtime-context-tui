import Foundation
import AVFoundation
import ScreenCaptureKit
import CoreMedia

enum SystemAudioCaptureError: Error {
    case noDisplay
    case screenCaptureKitUnavailable
}

/// Captures system audio via ScreenCaptureKit. Video capture is a no-op (2×2 pixel stream
/// required by SCStream but discarded). Audio callbacks are resampled to 16 kHz mono Int16
/// and delivered as framed bytes via `onFrame`.
///
/// Thread-safety: `Resampler` is not thread-safe. The SCStream delivers sample buffers on
/// the `sampleHandlerQueue` we provide in `addStreamOutput` — a dedicated serial queue —
/// and `onFrame` is invoked synchronously on that queue, so the caller must tolerate that.
final class SystemAudioCapture: NSObject, SCStreamOutput, SCStreamDelegate {
    private let resampler: Resampler
    private let onFrame: (Data, UInt32) -> Void  // (pcm bytes, timestampMs)
    private let startTime = Date()
    private var stream: SCStream?

    init(resampler: Resampler, onFrame: @escaping (Data, UInt32) -> Void) {
        self.resampler = resampler
        self.onFrame = onFrame
    }

    func start() async throws {
        let content = try await SCShareableContent.current
        guard let display = content.displays.first else { throw SystemAudioCaptureError.noDisplay }
        let filter = SCContentFilter(display: display, excludingWindows: [])

        let config = SCStreamConfiguration()
        config.capturesAudio = true
        // Request 48 kHz mono. SCStreamConfiguration delivers Float32 (non-interleaved)
        // to the audio output callback; the Resampler handles any format via AVAudioConverter.
        config.sampleRate = 48000
        config.channelCount = 1
        // Minimal video config — required by SCStream even though we discard frames.
        config.width = 2
        config.height = 2
        config.pixelFormat = kCVPixelFormatType_32BGRA
        config.queueDepth = 5

        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream.addStreamOutput(self,
                                   type: .audio,
                                   sampleHandlerQueue: DispatchQueue(label: "audio-tap.system", qos: .userInitiated))
        try await stream.startCapture()
        self.stream = stream
    }

    func stop() async {
        if let stream = stream { try? await stream.stopCapture() }
        stream = nil
    }

    // MARK: - SCStreamOutput

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of outputType: SCStreamOutputType) {
        guard outputType == .audio else { return }
        guard let pcmBuf = pcmBuffer(from: sampleBuffer) else { return }
        do {
            let bytes = try resampler.resample(pcmBuf)
            // Timestamp is ms since process start, wrapped to 32 bits. At 1 kHz tick this
            // wraps every ~49.7 days; fine for hour-scale capture sessions. Consumers
            // must handle wraparound if they ever run longer.
            let ts = UInt32(Date().timeIntervalSince(startTime) * 1000) & 0xFFFFFFFF
            onFrame(bytes, ts)
        } catch {
            fputs("audio-tap: resample failed: \(error)\n", stderr)
        }
    }

    // MARK: - SCStreamDelegate

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("audio-tap: system capture stopped with error: \(error)\n", stderr)
    }

    // MARK: - Helpers

    /// Convert a ScreenCaptureKit audio CMSampleBuffer to an AVAudioPCMBuffer.
    ///
    /// ScreenCaptureKit delivers audio as non-interleaved Float32. For mono (channelCount
    /// = 1), the `AudioBufferList` has a single buffer whose `mData` points to the samples —
    /// which aligns with `AVAudioPCMBuffer.floatChannelData[0]`. For multi-channel inputs the
    /// copy-via-first-buffer approach below is incorrect, but since we configure
    /// `channelCount = 1` we only ever see the mono path here.
    private func pcmBuffer(from sampleBuffer: CMSampleBuffer) -> AVAudioPCMBuffer? {
        guard let formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(formatDesc) else {
            return nil
        }
        var mutableAsbd = asbd.pointee
        guard let avFormat = AVAudioFormat(streamDescription: &mutableAsbd) else { return nil }

        let frameCount = AVAudioFrameCount(CMSampleBufferGetNumSamples(sampleBuffer))
        guard frameCount > 0 else { return nil }
        guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: avFormat, frameCapacity: frameCount) else { return nil }
        pcmBuffer.frameLength = frameCount

        guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return nil }
        var length = 0
        var dataPointer: UnsafeMutablePointer<Int8>?
        let status = CMBlockBufferGetDataPointer(blockBuffer,
                                                 atOffset: 0,
                                                 lengthAtOffsetOut: nil,
                                                 totalLengthOut: &length,
                                                 dataPointerOut: &dataPointer)
        guard status == kCMBlockBufferNoErr, let src = dataPointer else { return nil }

        // Copy raw bytes into the AVAudioPCMBuffer's backing store. For mono non-interleaved
        // Float32, the ABL's first buffer is the only channel; `mData` is the same memory
        // backing `floatChannelData[0]`, so this writes where `frameLength` expects samples.
        let abl = pcmBuffer.mutableAudioBufferList
        guard let dst = abl.pointee.mBuffers.mData else { return nil }
        let capacityBytes = Int(abl.pointee.mBuffers.mDataByteSize)
        let copyBytes = min(length, capacityBytes)
        memcpy(dst, src, copyBytes)
        return pcmBuffer
    }
}
