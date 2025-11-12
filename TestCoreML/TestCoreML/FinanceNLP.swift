import CoreML
import Foundation

struct FinanceSlotPrediction: Identifiable {
    let id = UUID()
    let amount: String
    let name: String
    let category: String
    let type: String
    let time: String
}

struct TokenizedText {
    let ids: [Int]
    let attentionMask: [Int]
    let tokens: [String]
}

final class FinanceTokenizer {
    private let vocab: [String: Int]
    private let unkToken = "[UNK]"
    private let clsToken = "[CLS]"
    private let sepToken = "[SEP]"
    private let padToken = "[PAD]"
    private let doLowercase = true
    private let maxInputCharsPerWord = 100

    init?(bundle: Bundle = .main, subdirectory: String = "NLPResources") {
        guard let vocabURL = bundle.resourceURL(for: "vocab", ext: "txt", subdirectory: subdirectory),
              let contents = try? String(contentsOf: vocabURL, encoding: .utf8) else {
            return nil
        }
        var table: [String: Int] = [:]
        for (idx, token) in contents.split(separator: "\n").map(String.init).enumerated() where !token.isEmpty {
            table[token] = idx
        }
        vocab = table
    }

    func encode(text: String, maxLength: Int) -> TokenizedText {
        var pieces = [clsToken]
        pieces.append(contentsOf: tokenize(text: text))
        pieces.append(sepToken)

        if pieces.count > maxLength {
            pieces = Array(pieces.prefix(maxLength))
            if pieces.last != sepToken {
                pieces[maxLength - 1] = sepToken
            }
        }

        var ids = pieces.map { vocab[$0] ?? vocab[unkToken] ?? 100 }
        var mask = Array(repeating: 1, count: ids.count)
        while ids.count < maxLength {
            ids.append(vocab[padToken] ?? 0)
            mask.append(0)
            pieces.append(padToken)
        }
        return TokenizedText(ids: ids, attentionMask: mask, tokens: pieces)
    }

    private func tokenize(text: String) -> [String] {
        basicTokenize(text: text).flatMap(wordPiece)
    }

    private func basicTokenize(text: String) -> [String] {
        let lowered = doLowercase ? text.lowercased() : text
        var buffer = ""
        for scalar in lowered.unicodeScalars {
            if isWhitespace(scalar) {
                buffer.append(" ")
            } else if isControl(scalar) {
                continue
            } else if isChineseChar(scalar) || isPunctuation(scalar) {
                buffer.append(" ")
                buffer.append(String(scalar))
                buffer.append(" ")
            } else {
                buffer.append(String(scalar))
            }
        }
        return buffer.split(whereSeparator: { $0.isWhitespace }).map(String.init)
    }

    private func wordPiece(token: String) -> [String] {
        guard token.count <= maxInputCharsPerWord else { return [unkToken] }
        var start = token.startIndex
        var pieces: [String] = []
        let end = token.endIndex

        while start < end {
            var cursor = end
            var current: String?
            while start < cursor {
                var substr = String(token[start..<cursor])
                if start != token.startIndex { substr = "##" + substr }
                if vocab[substr] != nil {
                    current = substr
                    break
                }
                cursor = token.index(before: cursor)
            }
            if let found = current {
                pieces.append(found)
                start = cursor
            } else {
                pieces = [unkToken]
                break
            }
        }
        return pieces
    }

    private func isWhitespace(_ scalar: UnicodeScalar) -> Bool {
        CharacterSet.whitespacesAndNewlines.contains(scalar)
    }

    private func isControl(_ scalar: UnicodeScalar) -> Bool {
        if scalar == "\t" || scalar == "\n" || scalar == "\r" { return false }
        return CharacterSet.controlCharacters.contains(scalar)
    }

    private func isPunctuation(_ scalar: UnicodeScalar) -> Bool {
        switch scalar.properties.generalCategory {
        case .connectorPunctuation, .dashPunctuation, .openPunctuation,
             .closePunctuation, .initialPunctuation, .finalPunctuation, .otherPunctuation:
            return true
        default:
            return false
        }
    }

    private func isChineseChar(_ scalar: UnicodeScalar) -> Bool {
        switch scalar.value {
        case 0x4E00...0x9FFF, 0x3400...0x4DBF, 0x20000...0x2A6DF,
             0x2A700...0x2B73F, 0x2B740...0x2B81F, 0x2B820...0x2CEAF,
             0xF900...0xFAFF, 0x2F800...0x2FA1F:
            return true
        default:
            return false
        }
    }
}

final class FinanceNLPModel {
    static let shared = try? FinanceNLPModel()

    private let model: MLModel
    private let tokenizer: FinanceTokenizer
    private let outputKey: String
    private let maxLength = 64
    private let slotOrder = ["amount", "name", "category", "type", "time"]
    private let id2label: [Int: String] = [
        0: "O", 1: "B-amount", 2: "I-amount", 3: "B-name", 4: "I-name",
        5: "B-category", 6: "I-category", 7: "B-type", 8: "I-type",
        9: "B-time", 10: "I-time"
    ]
    private let specialTokens: Set<String> = ["[CLS]", "[SEP]", "[PAD]"]

    init() throws {
        guard let tokenizer = FinanceTokenizer() else {
            throw NSError(domain: "FinanceNLP", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to load vocab"])
        }
        self.tokenizer = tokenizer

        let bundle = Bundle.main
        if let compiled = bundle.resourceURL(for: "coreml_model", ext: "mlmodelc", subdirectory: "NLPResources") {
            model = try MLModel(contentsOf: compiled)
        } else if let package = bundle.resourceURL(for: "coreml_model", ext: "mlpackage", subdirectory: "NLPResources") {
            let compiled = try MLModel.compileModel(at: package)
            model = try MLModel(contentsOf: compiled)
        } else {
            throw NSError(domain: "FinanceNLP", code: -2, userInfo: [NSLocalizedDescriptionKey: "Missing CoreML model"])
        }

        outputKey = FinanceNLPModel.resolveOutputKey(from: model.modelDescription) ?? "logits"
    }

    func predictSlots(for text: String) throws -> FinanceSlotPrediction {
        let tokenized = tokenizer.encode(text: text, maxLength: maxLength)
        let (provider, sequenceLength) = try makeInput(from: tokenized)
        let output = try model.prediction(from: provider)
        guard let tensor = output.featureValue(for: outputKey)?.multiArrayValue else {
            throw NSError(domain: "FinanceNLP", code: -3, userInfo: [NSLocalizedDescriptionKey: "Missing output tensor"])
        }
        let labels = decodeLabels(from: tensor, sequenceLength: sequenceLength)
        let slots = extractSlots(tokens: tokenized.tokens, labels: labels)
        return slots
    }

    private func makeInput(from tokenized: TokenizedText) throws -> (MLFeatureProvider, Int) {
        let length = tokenized.ids.count
        let shape: [NSNumber] = [1, NSNumber(value: length)]
        let idsArray = try MLMultiArray(shape: shape, dataType: .int32)
        let maskArray = try MLMultiArray(shape: shape, dataType: .int32)
        for (idx, value) in tokenized.ids.enumerated() {
            idsArray[[0, NSNumber(value: idx)]] = NSNumber(value: value)
        }
        for (idx, value) in tokenized.attentionMask.enumerated() {
            maskArray[[0, NSNumber(value: idx)]] = NSNumber(value: value)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: idsArray),
            "attention_mask": MLFeatureValue(multiArray: maskArray)
        ])
        return (provider, length)
    }

    private func decodeLabels(from tensor: MLMultiArray, sequenceLength: Int) -> [Int] {
        switch tensor.shape.count {
        case 3:
            return decodeLogits(tensor, sequenceLength: sequenceLength)
        case 2:
            return decodePredictions(tensor, sequenceLength: sequenceLength)
        default:
            return Array(repeating: 0, count: sequenceLength)
        }
    }

    private func decodeLogits(_ tensor: MLMultiArray, sequenceLength: Int) -> [Int] {
        let numLabels = id2label.count
        var results = Array(repeating: 0, count: sequenceLength)
        for position in 0..<sequenceLength {
            var bestLabel = 0
            var bestScore = -Double.infinity
            for label in 0..<numLabels {
                let score = tensor[[0, NSNumber(value: position), NSNumber(value: label)]].doubleValue
                if score > bestScore {
                    bestScore = score
                    bestLabel = label
                }
            }
            results[position] = bestLabel
        }
        return results
    }

    private func decodePredictions(_ tensor: MLMultiArray, sequenceLength: Int) -> [Int] {
        var results = Array(repeating: 0, count: sequenceLength)
        for position in 0..<sequenceLength {
            let value = tensor[[0, NSNumber(value: position)]].doubleValue
            results[position] = Int(value.rounded())
        }
        return results
    }

    private func extractSlots(tokens: [String], labels: [Int]) -> FinanceSlotPrediction {
        var pieces: [String: [String]] = [:]
        slotOrder.forEach { pieces[$0] = [] }

        var currentSlot: String?
        var currentPieces: [String] = []

        func flush() {
            if let slot = currentSlot, !currentPieces.isEmpty {
                pieces[slot, default: []].append(currentPieces.joined())
            }
            currentSlot = nil
            currentPieces.removeAll(keepingCapacity: true)
        }

        for (token, labelId) in zip(tokens, labels) {
            if specialTokens.contains(token) {
                flush()
                continue
            }
            let label = id2label[labelId] ?? "O"
            let piece = cleanPiece(token)
            if label == "O" || !label.contains("-") || piece.isEmpty {
                flush()
                continue
            }
            let parts = label.split(separator: "-", maxSplits: 1)
            guard parts.count == 2 else {
                flush()
                continue
            }
            let prefix = parts[0]
            let slot = String(parts[1])
            guard slotOrder.contains(slot) else {
                flush()
                continue
            }
            if prefix == "B" || slot != currentSlot {
                flush()
                currentSlot = slot
                currentPieces = [piece]
            } else {
                currentPieces.append(piece)
            }
        }
        flush()

        func value(for slot: String) -> String {
            let joined = pieces[slot]?.joined(separator: " / ") ?? ""
            return joined.isEmpty ? "-" : joined
        }

        return FinanceSlotPrediction(
            amount: value(for: "amount"),
            name: value(for: "name"),
            category: value(for: "category"),
            type: value(for: "type"),
            time: value(for: "time")
        )
    }

    private func cleanPiece(_ piece: String) -> String {
        if piece.hasPrefix("##") { return String(piece.dropFirst(2)) }
        if piece.hasPrefix("▁") { return String(piece.dropFirst()) }
        return piece
    }

    private static func resolveOutputKey(from description: MLModelDescription) -> String? {
        let preferred = ["logits", "Identity", "Identity_1"]
        for key in preferred where description.outputDescriptionsByName[key]?.type == .multiArray {
            return key
        }
        return description.outputDescriptionsByName.first(where: { $0.value.type == .multiArray })?.key
    }
}

private extension Bundle {
    func resourceURL(for name: String, ext: String, subdirectory: String) -> URL? {
        if let direct = url(forResource: name, withExtension: ext, subdirectory: subdirectory) {
            return direct
        }
        return url(forResource: name, withExtension: ext)
    }
}
