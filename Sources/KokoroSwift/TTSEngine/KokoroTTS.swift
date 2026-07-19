//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN
import MLXUtilsLibrary

/// Main class that encapsulates the complete Kokoro text-to-speech pipeline.
///
/// KokoroTTS converts text input into audio output by:
/// 1. Processing text through grapheme-to-phoneme (G2P) conversion
/// 2. Encoding the phonemes using BERT-based embeddings
/// 3. Predicting duration and prosody for natural speech
/// 4. Generating audio through a decoder network
///
/// Example usage:
/// ```swift
/// let tts = KokoroTTS(modelPath: modelURL, g2p: .misaki)
/// let audioData = try tts.generateAudio(voice: voiceEmbedding,
///                                       language: .english,
///                                       text: "Hello world",
///                                       speed: 1.0)
/// ```
public final class KokoroTTS {
  /// Errors from the TTS side
  public enum KokoroTTSError: Error {
    /// Thrown when input text exceeds maximum token count
    case tooManyTokens
  }

  /// BERT model for encoding phoneme sequences
  private let bert: CustomAlbert!

  /// Linear layer to project BERT embeddings
  private let bertEncoder: Linear!

  /// Encoder for duration prediction features
  private let durationEncoder: DurationEncoder!

  /// Bidirectional LSTM for duration prediction
  private let predictorLSTM: LSTM!

  /// Projection layer for final duration values
  private let durationProj: Linear!

  /// Predictor for prosodic features (F0, pitch)
  private let prosodyPredictor: ProsodyPredictor!

  /// Text encoder that processes phoneme sequences
  private let textEncoder: TextEncoder!

  /// Decoder that generates audio from encoded features
  private let decoder: Decoder!

  /// Grapheme-to-phoneme processor for text conversion
  private let g2pProcessor: G2PProcessor?

  /// Currently active language (cached to avoid reinitializing G2P)
  private var chosenLanguage: Language = .none

  private let configuration: KokoroConfiguration
  private var compiledBERTAndDuration: (@Sendable ([MLXArray]) -> [MLXArray])?
  private var compiledDurationPrediction: (@Sendable ([MLXArray]) -> [MLXArray])?
  private var compiledProsodyPrediction: (@Sendable ([MLXArray]) -> [MLXArray])?
  private var compiledTextEncoding: (@Sendable ([MLXArray]) -> [MLXArray])?
  private var compiledDecoder: (@Sendable ([MLXArray]) -> [MLXArray])?

  /// Initializes the Kokoro TTS engine with model weights and G2P processor.
  /// - Parameters:
  ///   - modelPath: URL to the directory containing model weights
  ///   - g2p: Grapheme-to-phoneme processor type (default: Misaki)
  public init(
    modelPath: URL,
    g2p: G2P = .misaki,
    configuration: KokoroConfiguration = .baseline
  ) {
    self.configuration = configuration
    // Load and sanitize model weights
    let sanitizedWeights = WeightLoader.loadWeights(
      modelPath: modelPath,
      precision: configuration.weightPrecision
    )
    let config = KokoroConfig.loadConfig()

    // Initialize BERT model for phoneme encoding
    bert = CustomAlbert(
      weights: sanitizedWeights,
      config: AlbertModelArgs(
        numHiddenLayers: config.plbert.numHiddenLayers,
        numAttentionHeads: config.plbert.numAttentionHeads,
        hiddenSize: config.plbert.hiddenSize,
        intermediateSize: config.plbert.intermediateSize,
        vocabSize: config.nToken
      )
    )

    // Initialize BERT output encoder
    bertEncoder = Linear(
      weight: sanitizedWeights["bert_encoder.weight"]!,
      bias: sanitizedWeights["bert_encoder.bias"]!
    )

    // Initialize duration prediction components
    durationEncoder = DurationEncoder(
      weights: sanitizedWeights,
      dModel: config.hiddenDim,
      styDim: config.styleDim,
      nlayers: config.nLayer
    )

    // Initialize bidirectional LSTM for duration prediction
    predictorLSTM = LSTM(
      inputSize: config.hiddenDim + config.styleDim,
      hiddenSize: config.hiddenDim / 2,
      wxForward: sanitizedWeights["predictor.lstm.weight_ih_l0"]!,
      whForward: sanitizedWeights["predictor.lstm.weight_hh_l0"]!,
      biasIhForward: sanitizedWeights["predictor.lstm.bias_ih_l0"]!,
      biasHhForward: sanitizedWeights["predictor.lstm.bias_hh_l0"]!,
      wxBackward: sanitizedWeights["predictor.lstm.weight_ih_l0_reverse"]!,
      whBackward: sanitizedWeights["predictor.lstm.weight_hh_l0_reverse"]!,
      biasIhBackward: sanitizedWeights["predictor.lstm.bias_ih_l0_reverse"]!,
      biasHhBackward: sanitizedWeights["predictor.lstm.bias_hh_l0_reverse"]!
    )

    // Initialize duration projection layer
    durationProj = Linear(
      weight: sanitizedWeights["predictor.duration_proj.linear_layer.weight"]!,
      bias: sanitizedWeights["predictor.duration_proj.linear_layer.bias"]!
    )

    // Initialize prosody predictor (F0, pitch, etc.)
    prosodyPredictor = ProsodyPredictor(
      weights: sanitizedWeights,
      styleDim: config.styleDim,
      dHid: config.hiddenDim
    )

    // Initialize text encoder
    textEncoder = TextEncoder(
      weights: sanitizedWeights,
      channels: config.hiddenDim,
      kernelSize: config.textEncoderKernelSize,
      depth: config.nLayer,
      nSymbols: config.nToken
    )

    // Initialize audio decoder
    decoder = Decoder(
      weights: sanitizedWeights,
      dimIn: config.hiddenDim,
      styleDim: config.styleDim,
      dimOut: config.nMels,
      resblockKernelSizes: config.istftNet.resblockKernelSizes,
      upsampleRates: config.istftNet.upsampleRates,
      upsampleInitialChannel: config.istftNet.upsampleInitialChannel,
      resblockDilationSizes: config.istftNet.resblockDilationSizes,
      upsampleKernelSizes: config.istftNet.upsampleKernelSizes,
      genIstftNFft: config.istftNet.genIstftNFFT,
      genIstftHopSize: config.istftNet.genIstftHopSize
    )

    // Initialize G2P processor for text-to-phoneme conversion
    g2pProcessor = try? G2PFactory.createG2PProcessor(engine: g2p)

    if configuration.compilesModelStages {
      configureCompiledStages()
    }
  }

  /// Generates audio from text using the specified voice and parameters.
  ///
  /// This method performs the complete TTS pipeline:
  /// 1. Converts text to phonemes (G2P)
  /// 2. Tokenizes and encodes phonemes
  /// 3. Predicts duration and prosody
  /// 4. Generates audio waveform
  ///
  /// - Parameters:
  ///   - voice: Voice embedding array (contains speaker characteristics)
  ///   - language: Target language for pronunciation
  ///   - text: Input text to synthesize
  ///   - speed: Speech speed multiplier (1.0 = normal, >1.0 = faster, <1.0 = slower)
  /// - Returns: Array of audio samples as Float values
  /// - Throws: `KokoroTTSError.tooManyTokens` if text is too long,
  ///           or `G2PProcessorError` if G2P processing fails
  public func generateAudio(voice: MLXArray, language: Language, text: String, speed: Float = 1.0)
    throws -> ([Float], [MToken]?)
  {
    // Update language if it has changed
    try updateLanguageIfNeeded(language)

    // Start performance timing
    BenchmarkTimer.reset()
    BenchmarkTimer.startTimer(Constants.bm_TTS)

    // Step 1: Convert text to phonemes
    let (phonemizedText, tokenArray) = try KokoroPerformance.measured("Kokoro G2P") {
      try phonemizeText(text)
    }

    // Step 2: Tokenize and prepare input
    let (paddedInputIds, attentionMask, inputLengths, textMask, inputIds) =
      try KokoroPerformance.measured("Kokoro Prepare Input") {
        try prepareInputTensors(phonemizedText)
      }

    // Step 3: Extract style embeddings from voice
    let configuredVoice =
      configuration.weightPrecision == .float16
      ? voice.asType(.float16)
      : voice
    let (globalStyle, acousticStyle) = KokoroPerformance.measured("Kokoro Style") {
      extractStyleEmbeddings(from: configuredVoice, tokenCount: inputIds.count)
    }

    // Step 4: Encode text with BERT and predict duration
    let durationFeatures = KokoroPerformance.measured("Kokoro BERT Duration") {
      encodeBERTAndDuration(
        inputIds: paddedInputIds,
        attentionMask: attentionMask,
        inputLengths: inputLengths,
        textMask: textMask,
        style: globalStyle
      )
    }

    // Step 5: Predict phoneme durations
    let predictedDurations = KokoroPerformance.measured("Kokoro Predict Durations") {
      predictDurations(features: durationFeatures, speed: speed)
    }
    let alignmentTarget = KokoroPerformance.measured("Kokoro Build Alignment") {
      createAlignmentTarget(
        durations: predictedDurations,
        batchSize: paddedInputIds.shape[1]
      )
    }

    // Step 6: Generate aligned encodings
    let alignedEncoding = durationFeatures.transposed(0, 2, 1).matmul(alignmentTarget)

    // Step 7: Predict prosody (F0, pitch)
    let (f0Prediction, nPrediction) = KokoroPerformance.measured("Kokoro Prosody") {
      predictProsody(alignedEncoding: alignedEncoding, style: globalStyle)
    }

    // Step 8: Encode text for decoder
    let asrFeatures = KokoroPerformance.measured("Kokoro Text Encoder") {
      encodeText(
        inputIds: paddedInputIds,
        inputLengths: inputLengths,
        textMask: textMask,
        alignmentTarget: alignmentTarget
      )
    }

    // Step 9: Generate audio
    let audio = KokoroPerformance.measured("Kokoro Decoder") {
      decode(
        asrFeatures: asrFeatures,
        f0Prediction: f0Prediction,
        nPrediction: nPrediction,
        acousticStyle: acousticStyle
      )
    }

    // Try to predict timestamp of each token if G2P processor returns tokens
    if let tokenArray {
      KokoroPerformance.measured("Kokoro Timestamps") {
        TimestampPredictor.preditTimestamps(
          tokens: tokenArray, predictionDuration: predictedDurations)
      }
    }

    // Stop performance timing
    BenchmarkTimer.stopTimer(Constants.bm_TTS)

    let samples = KokoroPerformance.measured("Kokoro Materialize Audio") {
      audio[0].asArray(Float.self)
    }
    return (samples, tokenArray)
  }

  /// Updates the G2P language if it differs from the current language.
  private func updateLanguageIfNeeded(_ language: Language) throws {
    guard chosenLanguage != language else { return }

    guard let g2pProcessor else {
      throw G2PProcessorError.processorNotInitialized
    }

    try g2pProcessor.setLanguage(language)
    chosenLanguage = language
  }

  /// Converts input text to phonemes using the G2P processor.
  private func phonemizeText(_ text: String) throws -> (String, [MToken]?) {
    let phonemizedOutput = try g2pProcessor?.process(input: text)
    guard let phonemizedOutput else {
      throw G2PProcessorError.processorNotInitialized
    }
    return phonemizedOutput
  }

  /// Prepares input tensors for the model from phonemized text.
  /// - Returns: Tuple containing:
  ///   - paddedInputIds: Tokenized and padded input sequence
  ///   - attentionMask: Mask for attention mechanism
  ///   - inputLengths: Length of input sequence
  ///   - textMask: Mask for text padding
  ///   - inputIds: Original token IDs before padding
  private func prepareInputTensors(_ phonemizedText: String) throws -> (
    MLXArray, MLXArray, MLXArray, MLXArray, [Int]
  ) {
    // Tokenize phonemized text
    let inputIds = Tokenizer.tokenize(phonemizedText: phonemizedText)

    // Check token count limit
    guard inputIds.count <= Constants.maxTokenCount else {
      throw KokoroTTSError.tooManyTokens
    }

    // Add padding tokens at start and end
    let paddedInputIdsArray = [0] + inputIds + [0]
    let paddedInputIds = MLXArray(paddedInputIdsArray).expandedDimensions(axes: [0])

    // Create input length tensor
    let inputLengths = MLXArray(paddedInputIds.dim(-1))
    let inputLengthMax: Int = inputLengths.max().item()

    // Create text mask for padding positions
    var textMask = MLXArray(0..<inputLengthMax)
    textMask = textMask + 1 .> inputLengths
    textMask = textMask.expandedDimensions(axes: [0])

    // Create attention mask (1 for valid positions, 0 for padding)
    let swiftTextMask: [Bool] = textMask.asArray(Bool.self)
    let swiftTextMaskInt = swiftTextMask.map { !$0 ? 1 : 0 }
    let attentionMask = MLXArray(swiftTextMaskInt).reshaped(textMask.shape)

    return (paddedInputIds, attentionMask, inputLengths, textMask, inputIds)
  }

  /// Extracts style embeddings from the voice array.
  /// - Parameters:
  ///   - voice: Voice embedding array
  ///   - tokenCount: Number of tokens in the input
  /// - Returns: Tuple of (globalStyle, acousticStyle)
  ///   - globalStyle: Style embedding for prosody/duration (indices 128+)
  ///   - acousticStyle: Style embedding for acoustic features (indices 0-127)
  private func extractStyleEmbeddings(from voice: MLXArray, tokenCount: Int) -> (MLXArray, MLXArray)
  {
    // Extract reference style from voice embedding
    let referenceStyle = voice[tokenCount - 1, 0...1, 0...]

    // Split into global style (for prosody/duration) and acoustic style
    let globalStyle = referenceStyle[0...1, 128...]
    let acousticStyle = referenceStyle[0...1, 0...127]

    return (globalStyle, acousticStyle)
  }

  /// Encodes text with BERT and generates duration prediction features.
  private func encodeBERTAndDuration(
    inputIds: MLXArray,
    attentionMask: MLXArray,
    inputLengths: MLXArray,
    textMask: MLXArray,
    style: MLXArray
  ) -> MLXArray {
    if let compiledBERTAndDuration {
      return compiledBERTAndDuration([
        inputIds,
        attentionMask,
        inputLengths,
        textMask,
        style,
      ])[0]
    }

    // Pass through BERT model
    let (bertOutput, _) = bert(inputIds, attentionMask: attentionMask)

    // Project BERT output and transpose for duration encoder
    let bertEncoded = bertEncoder(bertOutput).transposed(0, 2, 1)

    // Generate duration features with style conditioning
    let durationFeatures = durationEncoder(
      bertEncoded,
      style: style,
      textLengths: inputLengths,
      m: textMask
    )

    return durationFeatures
  }

  /// Predicts phoneme durations and creates alignment target matrix.
  /// - Parameters:
  ///   - features: Duration prediction features from encoder
  ///   - batchSize: Size of the input batch
  ///   - speed: Speech speed multiplier
  /// - Returns: Predicted durations and alignment target matrix for duration expansion
  private func predictDurations(features: MLXArray, speed: Float) -> MLXArray {
    if let compiledDurationPrediction {
      return compiledDurationPrediction([features, MLXArray(speed)])[0]
    }

    return durationPrediction(features: features, speed: MLXArray(speed))
  }

  private func durationPrediction(features: MLXArray, speed: MLXArray) -> MLXArray {
    // Pass through LSTM
    let (lstmOutput, _) = predictorLSTM(features)

    // Project to duration values
    let durationLogits = durationProj(lstmOutput)

    // Convert to actual durations (clamped to minimum of 1 frame)
    let durationSigmoid = MLX.sigmoid(durationLogits).sum(axis: -1) / speed
    return MLX.clip(durationSigmoid.round(), min: 1).asType(.int32)[0]
  }

  /// Creates an alignment target matrix from predicted durations. Maps each phoneme to multiple frames based on duration.
  /// Each row corresponds to a phoneme, and columns represent frames.
  /// - Parameters:
  ///   - durations: Predicted duration for each phoneme
  ///   - batchSize: Size of the input batch
  /// - Returns: Alignment matrix [batchSize × totalFrames]
  private func createAlignmentTarget(durations: MLXArray, batchSize: Int) -> MLXArray {
    // Materialize once. Scalar item() calls here serialize GPU work for every
    // token and generated frame, which dominates duration expansion on device.
    let frameCounts = durations.asArray(Int32.self).map(Int.init)
    let (alignmentArray, totalFrames) = Self.makeAlignmentArray(
      frameCounts: frameCounts,
      batchSize: batchSize
    )

    let alignmentTarget = MLXArray(alignmentArray)
      .asType(configuration.computationDType)
      .reshaped([batchSize, totalFrames])
    return alignmentTarget.expandedDimensions(axis: 0)
  }

  private func predictProsody(
    alignedEncoding: MLXArray,
    style: MLXArray
  ) -> (MLXArray, MLXArray) {
    if let compiledProsodyPrediction {
      let output = compiledProsodyPrediction([alignedEncoding, style])
      return (output[0], output[1])
    }
    return prosodyPredictor.F0NTrain(x: alignedEncoding, s: style)
  }

  private func encodeText(
    inputIds: MLXArray,
    inputLengths: MLXArray,
    textMask: MLXArray,
    alignmentTarget: MLXArray
  ) -> MLXArray {
    if let compiledTextEncoding {
      return compiledTextEncoding([inputIds, inputLengths, textMask, alignmentTarget])[0]
    }
    return MLX.matmul(
      textEncoder(inputIds, inputLengths: inputLengths, m: textMask),
      alignmentTarget
    )
  }

  private func decode(
    asrFeatures: MLXArray,
    f0Prediction: MLXArray,
    nPrediction: MLXArray,
    acousticStyle: MLXArray
  ) -> MLXArray {
    if let compiledDecoder {
      return compiledDecoder([asrFeatures, f0Prediction, nPrediction, acousticStyle])[0]
    }
    return decoder(
      asr: asrFeatures,
      F0Curve: f0Prediction,
      N: nPrediction,
      s: acousticStyle
    )[0]
  }

  private func configureCompiledStages() {
    let bert = bert!
    let bertEncoder = bertEncoder!
    let durationEncoder = durationEncoder!
    let predictorLSTM = predictorLSTM!
    let durationProj = durationProj!
    let prosodyPredictor = prosodyPredictor!
    let textEncoder = textEncoder!
    let decoder = decoder!

    // Kokoro's graphs contain shape-dependent operations. Shape-aware compilation
    // avoids backend aborts from unsupported shapeless traces. Model weights are
    // immutable during inference, so capturing them as constants is intentional.
    compiledBERTAndDuration = MLX.compile {
      [bert, bertEncoder, durationEncoder] arrays in
      let (bertOutput, _) = bert(arrays[0], attentionMask: arrays[1])
      let bertEncoded = bertEncoder(bertOutput).transposed(0, 2, 1)
      return [
        durationEncoder(
          bertEncoded,
          style: arrays[4],
          textLengths: arrays[2],
          m: arrays[3]
        )
      ]
    }
    compiledDurationPrediction = MLX.compile {
      [predictorLSTM, durationProj] arrays in
      let (lstmOutput, _) = predictorLSTM(arrays[0])
      let logits = durationProj(lstmOutput)
      let duration = MLX.sigmoid(logits).sum(axis: -1) / arrays[1]
      return [MLX.clip(duration.round(), min: 1).asType(.int32)[0]]
    }
    compiledProsodyPrediction = MLX.compile { [prosodyPredictor] arrays in
      let (f0, noise) = prosodyPredictor.F0NTrain(x: arrays[0], s: arrays[1])
      return [f0, noise]
    }
    compiledTextEncoding = MLX.compile { [textEncoder] arrays in
      let encoded = textEncoder(arrays[0], inputLengths: arrays[1], m: arrays[2])
      return [MLX.matmul(encoded, arrays[3])]
    }
    compiledDecoder = MLX.compile { [decoder] arrays in
      [decoder(asr: arrays[0], F0Curve: arrays[1], N: arrays[2], s: arrays[3])[0]]
    }
  }

  /// Builds the row-major one-hot duration expansion used by the decoder.
  static func makeAlignmentArray(
    frameCounts: [Int],
    batchSize: Int
  ) -> (values: [Float], totalFrames: Int) {
    precondition(frameCounts.count == batchSize)
    precondition(frameCounts.allSatisfy { $0 >= 0 })

    let totalFrames = frameCounts.reduce(0, +)
    var values = [Float](repeating: 0, count: batchSize * totalFrames)
    var frameOffset = 0

    for (phonemeIndex, frameCount) in frameCounts.enumerated() {
      let rowOffset = phonemeIndex * totalFrames
      for frame in frameOffset..<frameOffset + frameCount {
        values[rowOffset + frame] = 1
      }
      frameOffset += frameCount
    }

    return (values, totalFrames)
  }

  /// Constants used throughout the TTS engine.
  public struct Constants {
    /// Maximum number of tokens allowed in input
    public static let maxTokenCount = 510

    /// Audio sampling rate in Hz
    public static let samplingRate = 24000

    // Benchmark timer identifiers
    public static let bm_TTS = "TTSAudio"
    static let bm_Phonemize = "Phonemize"
    static let bm_bert = "BERT"
    static let bm_duration = "Duration"
    static let bm_prosody = "Prosody"
    static let bm_decoder = "Decoder"
  }
}
