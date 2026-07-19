import MLX

/// Experimental inference choices that can be benchmarked without changing Kokoro's text pipeline.
public struct KokoroConfiguration: Hashable, Sendable {
  public enum WeightPrecision: String, CaseIterable, Sendable {
    case float32
    case float16
  }

  public var compilesModelStages: Bool
  public var weightPrecision: WeightPrecision

  public init(
    compilesModelStages: Bool = false,
    weightPrecision: WeightPrecision = .float32
  ) {
    self.compilesModelStages = compilesModelStages
    self.weightPrecision = weightPrecision
  }

  public static let baseline = KokoroConfiguration()

  var computationDType: DType {
    switch weightPrecision {
    case .float32: .float32
    case .float16: .float16
    }
  }
}
