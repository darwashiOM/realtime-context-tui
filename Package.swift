// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "audio-tap",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "AudioTap",
            path: "Sources/AudioTap"
        ),
        .testTarget(
            name: "AudioTapTests",
            dependencies: ["AudioTap"],
            path: "Tests/AudioTapTests"
        ),
    ]
)
