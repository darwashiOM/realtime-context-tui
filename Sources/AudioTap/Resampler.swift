import Foundation
import AVFoundation

enum ResamplerError: Error {
    case cannotCreateConverter
    case conversionFailed(Error?)
    case cannotCreateOutputBuffer
}

/// Resamples any AVAudioPCMBuffer to the target sample rate, mono Int16 PCM bytes.
/// One instance is reusable but not thread-safe — call from a single queue.
final class Resampler {
    let targetSampleRate: Double
    private let targetFormat: AVAudioFormat

    /// Lazily created on first buffer, keyed by input format. We include
    /// `interleaved` in the key so two formats that differ only in interleaving
    /// don't share a (wrong) converter.
    private var converterCache: [String: AVAudioConverter] = [:]

    init(targetSampleRate: Double) throws {
        self.targetSampleRate = targetSampleRate
        guard let fmt = AVAudioFormat(commonFormat: .pcmFormatInt16,
                                      sampleRate: targetSampleRate,
                                      channels: 1,
                                      interleaved: true) else {
            throw ResamplerError.cannotCreateConverter
        }
        self.targetFormat = fmt
    }

    func resample(_ input: AVAudioPCMBuffer) throws -> Data {
        let inputFormat = input.format
        let key = "\(inputFormat.sampleRate)-\(inputFormat.channelCount)-\(inputFormat.commonFormat.rawValue)-\(inputFormat.isInterleaved ? 1 : 0)"
        let converter: AVAudioConverter
        if let cached = converterCache[key] {
            converter = cached
        } else {
            guard let c = AVAudioConverter(from: inputFormat, to: targetFormat) else {
                throw ResamplerError.cannotCreateConverter
            }
            converterCache[key] = c
            converter = c
        }

        let ratio = targetSampleRate / inputFormat.sampleRate
        // Slack accounts for sample-rate-converter filter latency (a few hundred frames)
        // and rounding when ratio isn't exact.
        let outCapacity = AVAudioFrameCount(Double(input.frameLength) * ratio) + 1024
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outCapacity) else {
            throw ResamplerError.cannotCreateOutputBuffer
        }

        // Drive the conversion: first call supplies the input buffer; subsequent
        // calls signal end-of-stream so the SRC drains its internal filter state.
        // Without this drain step, downsampling drops the trailing ~240 output frames.
        var inputConsumed = false
        var conversionError: NSError?
        let status = converter.convert(to: outBuf, error: &conversionError) { _, inputStatus in
            if inputConsumed {
                inputStatus.pointee = .endOfStream
                return nil
            }
            inputConsumed = true
            inputStatus.pointee = .haveData
            return input
        }

        if status == .error {
            throw ResamplerError.conversionFailed(conversionError)
        }

        // Int16 interleaved mono — bytes are contiguous in channel 0's int16ChannelData.
        guard let channelData = outBuf.int16ChannelData else { return Data() }
        let byteCount = Int(outBuf.frameLength) * MemoryLayout<Int16>.size
        return Data(bytes: channelData[0], count: byteCount)
    }
}
