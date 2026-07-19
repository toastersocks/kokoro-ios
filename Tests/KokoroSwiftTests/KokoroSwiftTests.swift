import Testing

@testable import KokoroSwift

@Test func baselineConfigurationPreservesFloat32UncompiledInference() {
  #expect(KokoroConfiguration.baseline.weightPrecision == .float32)
  #expect(KokoroConfiguration.baseline.compilesModelStages == false)
  #expect(KokoroConfiguration.baseline.computationDType == .float32)
}

@Test func float16ConfigurationUsesFloat16Computation() {
  let configuration = KokoroConfiguration(
    compilesModelStages: true,
    weightPrecision: .float16
  )

  #expect(configuration.compilesModelStages)
  #expect(configuration.computationDType == .float16)
}

@Test func alignmentArrayExpandsEachDurationIntoItsPhonemeRow() {
  let result = KokoroTTS.makeAlignmentArray(
    frameCounts: [2, 1, 3],
    batchSize: 3
  )

  #expect(result.totalFrames == 6)
  #expect(
    result.values == [
      1, 1, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0,
      0, 0, 0, 1, 1, 1,
    ])
}

@Test func alignmentArraySupportsZeroDurationRows() {
  let result = KokoroTTS.makeAlignmentArray(
    frameCounts: [1, 0, 2],
    batchSize: 3
  )

  #expect(result.totalFrames == 3)
  #expect(
    result.values == [
      1, 0, 0,
      0, 0, 0,
      0, 1, 1,
    ])
}
