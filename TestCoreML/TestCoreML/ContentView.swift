import SwiftUI

struct ContentView: View {
    @State private var input = ""
    @State private var result: FinanceSlotPrediction?
    @State private var errorMessage: String?

    var body: some View {
        NavigationStack {
            Form {
                Section("输入") {
                    TextField("请输入一句记账语句", text: $input, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .lineLimit(3...6)
                    Button("运行 CoreML 模型") {
                        runInference()
                    }
                    .buttonStyle(.borderedProminent)
                }

                if let result {
                    Section("预测槽位") {
                        SlotRow(name: "金额", value: result.amount)
                        SlotRow(name: "商户", value: result.name)
                        SlotRow(name: "类别", value: result.category)
                        SlotRow(name: "类型", value: result.type)
                        SlotRow(name: "时间", value: result.time)
                    }
                }

                if let errorMessage {
                    Section("错误") {
                        Text(errorMessage)
                            .foregroundStyle(.red)
                            .font(.footnote)
                    }
                }
            }
            .navigationTitle("Finance CoreML")
        }
    }

    private func runInference() {
        do {
            guard let model = FinanceNLPModel.shared else {
                errorMessage = "模型尚未初始化"
                return
            }
            let prediction = try model.predictSlots(for: input)
            result = prediction
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
            result = nil
        }
    }
}

private struct SlotRow: View {
    let name: String
    let value: String

    var body: some View {
        HStack {
            Text(name)
                .fontWeight(.medium)
            Spacer()
            Text(value)
                .foregroundStyle(.secondary)
        }
    }
}

#Preview {
    ContentView()
}
