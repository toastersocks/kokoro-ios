import OSLog

/// Stage-level points of interest for profiling Kokoro independently of its host app.
enum KokoroPerformance {
  static let signposter = OSSignposter(
    subsystem: "com.mlalma.KokoroSwift",
    category: .pointsOfInterest
  )

  static func measured<Result>(
    _ name: StaticString,
    operation: () throws -> Result
  ) rethrows -> Result {
    let id = signposter.makeSignpostID()
    let interval = signposter.beginInterval(name, id: id)
    defer { signposter.endInterval(name, interval) }
    return try operation()
  }
}
