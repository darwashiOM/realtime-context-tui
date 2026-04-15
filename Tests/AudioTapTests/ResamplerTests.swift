import XCTest
import AVFoundation
@testable import AudioTap

final class ResamplerTests: XCTestCase {
    /// Build a PCM buffer of `frameCount` Float32 samples at `sampleRate` Hz, mono.
    private func synthBuffer(sampleRate: Double, frameCount: AVAudioFrameCount, fill: Float) -> AVAudioPCMBuffer {
        let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                sampleRate: sampleRate,
                                channels: 1,
                                interleaved: false)!
        let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: frameCount)!
        buf.frameLength = frameCount
        let ptr = buf.floatChannelData![0]
        for i in 0..<Int(frameCount) { ptr[i] = fill }
        return buf
    }

    func testResamples48kHzFloat32To16kHzInt16WithExpectedByteCount() throws {
        let sut = try Resampler(targetSampleRate: 16000)
        let inputBuf = synthBuffer(sampleRate: 48000, frameCount: 4800, fill: 0.5)  // 100 ms at 48 kHz
        let out = try sut.resample(inputBuf)
        // 100 ms at 16 kHz mono Int16 = 1600 samples × 2 bytes = 3200 bytes
        XCTAssertEqual(out.count, 3200)
    }

    /// Regression test: AVAudioConverter is stateful — once its input callback returns
    /// `.endOfStream` the converter is in a terminal state and emits 0 frames forever.
    /// The Resampler must reset the converter between calls so back-to-back invocations
    /// (which is the normal streaming case) all produce non-empty output.
    func testRepeatedCallsAllProduceNonEmptyOutput() throws {
        let sut = try Resampler(targetSampleRate: 16000)
        for i in 0..<5 {
            let inputBuf = synthBuffer(sampleRate: 48000, frameCount: 4800, fill: 0.5)
            let out = try sut.resample(inputBuf)
            XCTAssertEqual(out.count, 3200, "call #\(i) produced \(out.count) bytes (expected 3200)")
        }
    }

    func testSilenceInputProducesNearZeroOutput() throws {
        let sut = try Resampler(targetSampleRate: 16000)
        let inputBuf = synthBuffer(sampleRate: 48000, frameCount: 4800, fill: 0.0)
        let out = try sut.resample(inputBuf)
        let nonZero = out.withUnsafeBytes { raw -> Int in
            let samples = raw.bindMemory(to: Int16.self)
            return samples.filter { abs(Int($0)) > 100 }.count  // tolerate tiny numerical noise
        }
        XCTAssertLessThan(nonZero, 10)
    }
}
