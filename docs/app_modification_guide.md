# DubCamera iOS App 修改指南 — PPG 信号质量优化

> **目标**：通过修改 App 端设置，将 PPG 信噪比提升 10-20 倍，从根本上解决模型训练 r≈0 的问题。
>
> **背景**：当前录制的视频中，自动曝光漂移幅度为 5-10% 像素值，而真正的 PPG 脉搏信号仅占 0.1-0.5%。信号被噪声淹没了 10-100 倍。锁定曝光是解决这个问题最关键的一步。

---

## 修改概览

| 优先级 | 修改项 | 预期收益 | 难度 |
|--------|--------|---------|------|
| **P0 必做** | 锁定曝光/ISO/白平衡 | PPG SNR ↑ 10-20x | 中 |
| **P0 必做** | 手电筒亮度降低至 0.1-0.3 | 避免红色通道饱和 | 低 |
| **P1 强烈建议** | 视频质量改为最高 | 减少 H.264 压缩损失 | 低 |
| **P1 强烈建议** | 实时 PPG 信号质量反馈 | 保证每次采集数据可用 | 中高 |
| **P2 建议** | 同时输出每帧 RGB 均值 CSV | 完全无损 PPG 信号 | 中 |
| **P2 建议** | 前 5 秒预热期标记 | 丢弃自动曝光稳定期 | 低 |

---

## P0：锁定曝光/ISO/白平衡（最关键）

### 为什么必须做

当前代码 `CameraService.swift` 中，摄像头使用默认的自动曝光模式。iOS 会持续调整曝光和增益以维持"好看"的画面，但这对 PPG 信号是灾难性的：

- 自动曝光每次调整会造成 **15-25 个像素值的跳变**（0-255 范围）
- PPG 脉搏信号仅有 **0.3-1.3 个像素值的波动**
- 模型看到的"信号"中，99% 是曝光漂移噪声

### 在哪里改

修改 `CameraService.swift`，在 `configureSession` 函数中，创建 input 之后、commit 之前，对 camera_0（PPG 摄像头）锁定参数。

### 代码修改

在 `CameraService.swift` 中添加新方法：

```swift
// MARK: - PPG Camera Optimization (新增)

extension CameraService {

    /// 锁定摄像头参数以优化 PPG 信号采集
    /// 自动曝光会产生 5-10% 像素值漂移，完全淹没 0.1-0.5% 的 PPG 脉搏信号
    /// 锁定后，像素值变化将仅反映血液容积变化（即 PPG 信号）
    private func lockExposureForPPG(device: AVCaptureDevice) {
        do {
            try device.lockForConfiguration()

            // ========== 1. 锁定曝光（最关键）==========
            // 策略：先让自动曝光稳定 1-2 秒，然后锁定当前值
            // 这样不需要手动猜测合适的 ISO 和曝光时间
            if device.isExposureModeSupported(.locked) {
                // 方案 A（推荐）：先自动后锁定
                // 先设为 autoExpose，等 adjustingExposure 变 false 后切 locked
                // 见下方 observeExposureAndLock() 方法

                // 方案 B（备选，立即锁定固定值）：
                // 适用于测试阶段，快速验证效果
                let targetISO: Float = 50.0  // 低 ISO 减少传感器噪声
                let clampedISO = max(device.activeFormat.minISO,
                                    min(targetISO, device.activeFormat.maxISO))

                // 曝光时间：1/60s，避免与 50Hz 日光灯产生拍频
                let targetDuration = CMTimeMake(value: 1, timescale: 60)
                let clampedDuration = CMTimeClampToRange(
                    targetDuration,
                    range: device.activeFormat.minExposureDuration
                           ... device.activeFormat.maxExposureDuration
                )

                device.setExposureModeCustom(duration: clampedDuration, iso: clampedISO)
            }

            // ========== 2. 锁定白平衡 ==========
            // 白平衡调整会改变 R/G/B 通道的增益比例
            // 我们依赖红色通道的绝对值变化，所以必须锁定
            if device.isWhiteBalanceModeSupported(.locked) {
                device.whiteBalanceMode = .locked
            }

            // ========== 3. 锁定焦距 ==========
            // 自动对焦会微调镜片位置，间接影响进光量
            if device.isFocusModeSupported(.locked) {
                device.focusMode = .locked
            }

            device.unlockForConfiguration()

            print("[PPG] Camera locked - ISO: \(device.iso), "
                + "Exposure: \(CMTimeGetSeconds(device.exposureDuration))s, "
                + "WB: locked")

        } catch {
            print("[PPG] Failed to lock camera: \(error.localizedDescription)")
        }
    }

    /// 推荐方案：等自动曝光稳定后再锁定
    /// 调用时机：开启预览后、开始录制前
    func autoThenLockExposure(device: AVCaptureDevice, completion: @escaping () -> Void) {
        do {
            try device.lockForConfiguration()

            // 先设为连续自动曝光
            if device.isExposureModeSupported(.continuousAutoExposure) {
                device.exposureMode = .continuousAutoExposure
            }
            if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                device.whiteBalanceMode = .continuousAutoWhiteBalance
            }
            device.unlockForConfiguration()

        } catch {
            print("[PPG] Failed to set auto exposure: \(error)")
            completion()
            return
        }

        // 等待 2 秒让自动曝光稳定（手指放上去后需要时间适应）
        DispatchQueue.global().asyncAfter(deadline: .now() + 2.0) { [weak self] in
            guard let _ = self else { return }

            do {
                try device.lockForConfiguration()

                // 锁定当前的曝光值
                if device.isExposureModeSupported(.locked) {
                    device.exposureMode = .locked
                }
                if device.isWhiteBalanceModeSupported(.locked) {
                    device.whiteBalanceMode = .locked
                }
                if device.isFocusModeSupported(.locked) {
                    device.focusMode = .locked
                }

                device.unlockForConfiguration()

                print("[PPG] Auto-then-lock complete - ISO: \(device.iso), "
                    + "Exposure: \(CMTimeGetSeconds(device.exposureDuration))s")

            } catch {
                print("[PPG] Failed to lock after auto: \(error)")
            }

            completion()
        }
    }
}

// CMTime 辅助函数
private func CMTimeClampToRange(_ time: CMTime, range: ClosedRange<CMTime>) -> CMTime {
    if time < range.lowerBound { return range.lowerBound }
    if time > range.upperBound { return range.upperBound }
    return time
}
```

### 集成到录制流程

修改 `CameraViewModel.swift` 中的 `startActualRecording()`：

```swift
// 当前代码（第 135-149 行）：
private func startActualRecording() {
    Task {
        do {
            elapsedTime = 0
            recordingState = .recording
            imuService.startRecording()
            try cameraService.startRecording()
        } catch {
            imuService.stopRecording()
            cancelTimer()
            recordingState = .error(error.localizedDescription)
        }
    }
}

// 修改为：
private func startActualRecording() {
    Task {
        do {
            elapsedTime = 0

            // 新增：在开始录制前，锁定 PPG 摄像头（camera_0）的曝光
            // 倒计时 3 秒期间自动曝光已在运行，此时锁定当前值
            cameraService.lockPPGCameraExposure()

            recordingState = .recording
            imuService.startRecording()
            try cameraService.startRecording()
        } catch {
            imuService.stopRecording()
            cancelTimer()
            recordingState = .error(error.localizedDescription)
        }
    }
}
```

在 `CameraService.swift` 中添加公开方法：

```swift
/// 公开方法：锁定 PPG 摄像头（第一个后置摄像头）
func lockPPGCameraExposure() {
    guard let device = cameraConfigs.first?.device,
          device.position == .back else { return }
    lockExposureForPPG(device: device)
}
```

### 更好的集成方案（推荐）

利用倒计时的 3 秒让自动曝光稳定，然后在倒计时结束时锁定：

```swift
// 修改 CameraViewModel.swift 中 startCountdown() 函数
// 在倒计时开始时，通知 CameraService 准备锁定

private func startCountdown() {
    guard recordingState == .idle || recordingState == .completed else { return }

    cancelTimer()
    countdownValue = 3
    elapsedTime = 0
    recordingState = .countdown(countdownValue)
    hapticFeedback.prepare()

    // 新增：倒计时开始时，让摄像头先自动曝光
    // 3 秒倒计时期间自动曝光会稳定下来
    // 提示用户此时应该已经把手指放上去了

    timer = Timer(timeInterval: 1.0, repeats: true) { [weak self] timer in
        guard let self = self else {
            timer.invalidate()
            return
        }

        DispatchQueue.main.async {
            switch self.recordingState {
            case .countdown(let seconds):
                if seconds > 1 {
                    self.countdownValue = seconds - 1
                    self.recordingState = .countdown(self.countdownValue)
                } else {
                    self.recordingState = .countdown(0)

                    // 新增：倒计时结束时锁定曝光，然后开始录制
                    self.cameraService.lockPPGCameraExposure()

                    self.startActualRecording()
                }
            // ... 其余不变
            }
        }
    }

    RunLoop.main.add(timer!, forMode: .common)
}
```

---

## P0：降低手电筒亮度

### 为什么必须做

当前代码 `CameraService.swift` 第 282 行：
```swift
try device.setTorchModeOn(level: 1.0)  // 最大亮度！
```

手电筒满功率照射手指，红色通道容易饱和到 255（数据中观察到部分帧 R > 245）。
饱和 = 信息丢失，PPG 信号被截断。

### 代码修改

```swift
// 修改 CameraService.swift 中 enableTorch() 方法（第 274-288 行）

private func enableTorch() {
    guard let device = getBackCameraDevice(), device.hasTorch else {
        return
    }

    do {
        try device.lockForConfiguration()
        if device.isTorchModeSupported(.on) {
            // 原代码：try device.setTorchModeOn(level: 1.0)
            // 修改为：降低到 10%-30%，避免红色通道饱和
            // 如果画面太暗可以逐步增加，从 0.1 开始测试
            try device.setTorchModeOn(level: 0.1)
        }
        device.unlockForConfiguration()
    } catch {
        print("Failed to enable torch: \(error.localizedDescription)")
    }
}
```

**调试建议**：
- 从 `level: 0.1` 开始测试
- 如果锁定曝光后画面太暗（红通道均值 < 100），逐步增加到 0.2、0.3
- 理想范围：红通道均值在 **150-220** 之间（既不饱和，又有足够的动态范围）

---

## P1：视频质量设为最高

### 为什么建议做

当前默认 `AVAssetExportPresetMediumQuality`（约 2-3 Mbps），H.264 压缩会引入量化噪声。PPG 信号的像素级微小变化可能被压缩算法抹平。

### 代码修改

方案一：直接改默认值

```swift
// SettingsService.swift 第 18 行
// 原代码：
private let defaultVideoQuality = AVAssetExportPresetMediumQuality

// 改为：
private let defaultVideoQuality = AVAssetExportPresetHighestQuality
```

方案二（更好）：跳过 AVAssetExportSession 的二次压缩，直接保存原始 MOV

```swift
// CameraService.swift 中 convertToMP4() 方法（第 365-379 行）
// 当前流程：录制 MOV → AVAssetExportSession 转 MP4（二次压缩！）
// 问题：AVAssetExportSession 会对视频重新编码，引入额外压缩损失

// 方案：直接重命名 MOV 为 MP4（MOV 和 MP4 容器格式几乎相同）
// 或者使用 AVAssetExportPresetPassthrough 避免重编码
private func convertToMP4(inputURL: URL, outputURL: URL) async {
    let asset = AVAsset(url: inputURL)

    // 使用 Passthrough 避免重新编码（零质量损失）
    guard let exportSession = AVAssetExportSession(
        asset: asset,
        presetName: AVAssetExportPresetPassthrough  // 关键：直通模式，不重编码
    ) else {
        return
    }

    exportSession.outputURL = outputURL
    exportSession.outputFileType = .mp4

    await exportSession.export()
}
```

**推荐使用 `AVAssetExportPresetPassthrough`**，这样：
- 不重新编码，零质量损失
- 导出速度更快
- 文件大小与原始录制相同

---

## P1：实时 PPG 信号质量反馈（强烈建议）

### 为什么建议做

当前数据中有 **10/98 样本为 poor 质量**（HR 误差 > 20 BPM），采集时无法判断数据好坏。如果在录制过程中能实时显示 PPG 波形质量，受试者可以调整手指按压力度，大幅减少废数据。

### 实现思路

需要新增一个 `AVCaptureVideoDataOutput` 来实时读取每帧像素值，同时不影响现有的 `AVCaptureMovieFileOutput` 录制。

```swift
// 新增文件：PPGSignalMonitor.swift

import AVFoundation
import Accelerate

class PPGSignalMonitor: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    // 最近 N 帧的红色通道均值（用于实时 PPG 波形显示）
    private var redChannelHistory: [Float] = []
    private let historySize = 150  // 5 秒 @ 30fps
    private let lock = NSLock()

    // 信号质量指标（0-1，越大越好）
    @Published var signalQuality: Float = 0.0
    @Published var currentRedMean: Float = 0.0
    @Published var qualityMessage: String = ""

    /// 创建视频数据输出（添加到 session）
    func createVideoDataOutput() -> AVCaptureVideoDataOutput {
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        // 设置较低的回调频率以减少 CPU 负担
        output.alwaysDiscardsLateVideoFrames = true

        let queue = DispatchQueue(label: "ppg.monitor", qos: .userInteractive)
        output.setSampleBufferDelegate(self, queue: queue)

        return output
    }

    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }
        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)

        // 只取中心 50% 区域（避免边缘漏光）
        let startX = width / 4
        let endX = width * 3 / 4
        let startY = height / 4
        let endY = height * 3 / 4

        var redSum: Float = 0
        var pixelCount: Float = 0

        for y in startY..<endY {
            for x in startX..<endX {
                let offset = y * bytesPerRow + x * 4
                // BGRA 格式：B=0, G=1, R=2, A=3
                let red = Float(buffer[offset + 2])
                redSum += red
                pixelCount += 1
            }
        }

        let redMean = redSum / pixelCount

        lock.lock()
        redChannelHistory.append(redMean)
        if redChannelHistory.count > historySize {
            redChannelHistory.removeFirst()
        }
        let history = redChannelHistory
        lock.unlock()

        // 更新信号质量指标
        DispatchQueue.main.async { [weak self] in
            self?.currentRedMean = redMean
            self?.updateQuality(history: history, redMean: redMean)
        }
    }

    private func updateQuality(history: [Float], redMean: Float) {
        // 检查 1：饱和度
        if redMean > 245 {
            qualityMessage = "⚠️ 画面过亮，请轻放手指或降低手电筒亮度"
            signalQuality = 0.1
            return
        }
        if redMean < 50 {
            qualityMessage = "⚠️ 画面过暗，请调整手指位置"
            signalQuality = 0.1
            return
        }

        // 检查 2：信号变异性（是否能看到脉搏波动）
        guard history.count >= 90 else {  // 至少 3 秒数据
            qualityMessage = "正在分析信号..."
            signalQuality = 0.5
            return
        }

        // 计算变异系数（CV = std/mean）
        let mean = history.reduce(0, +) / Float(history.count)
        let variance = history.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(history.count)
        let std = sqrt(variance)
        let cv = std / (mean + 1e-8)

        // PPG 信号的 CV 通常在 0.001-0.01 之间
        // 自动曝光漂移的 CV 通常 > 0.05
        if cv < 0.0005 {
            qualityMessage = "⚠️ 信号太弱，请调整手指按压力度"
            signalQuality = 0.2
        } else if cv > 0.05 {
            qualityMessage = "⚠️ 信号不稳定，请保持手指不动"
            signalQuality = 0.3
        } else {
            qualityMessage = "✅ 信号质量良好"
            signalQuality = min(1.0, cv * 100)  // 归一化到 0-1
        }
    }
}
```

**集成到 CameraService**：在 `configureSession` 中，为 camera_0 额外添加一个 `AVCaptureVideoDataOutput`。注意 MultiCamSession 支持同一摄像头同时有 `MovieFileOutput` 和 `VideoDataOutput`。

---

## P2：同时输出每帧 RGB 均值 CSV

### 为什么建议做

PPG 分析只需要每帧的 RGB 通道均值（3 个浮点数），不需要完整的视频帧。直接输出 CSV 可以：
- 完全避免 H.264 压缩损失
- 文件极小（120s × 30fps × 3 channels ≈ 100KB vs 当前 3-5MB）
- 后续处理更快（不需要用 OpenCV 解码视频）

### 实现方式

利用上面 `PPGSignalMonitor` 的帧回调，将每帧的 RGB 均值记录到文件：

```swift
// 在 PPGSignalMonitor 中添加

private var csvLines: [String] = []

func startCSVRecording() {
    lock.lock()
    csvLines = ["timestamp_ms,red_mean,green_mean,blue_mean"]
    lock.unlock()
}

func stopCSVRecording() -> URL? {
    let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let timestamp = Int(Date().timeIntervalSince1970)
    let fileURL = documentsPath.appendingPathComponent("ppg_raw_\(timestamp).csv")

    lock.lock()
    let data = csvLines.joined(separator: "\n")
    lock.unlock()

    try? data.write(to: fileURL, atomically: true, encoding: .utf8)
    return fileURL
}

// 在 captureOutput 回调中，计算完均值后追加：
func captureOutput(...) {
    // ... 计算 redMean, greenMean, blueMean ...

    let timestampMs = Int64(Date().timeIntervalSince1970 * 1000)
    let line = "\(timestampMs),\(redMean),\(greenMean),\(blueMean)"

    lock.lock()
    csvLines.append(line)
    lock.unlock()
}
```

---

## P2：前 5 秒预热期标记

### 为什么建议做

即使锁定了曝光，手指按压后的最初几秒内，手指接触面积、按压力度仍在调整。标记预热期可以在后处理中自动丢弃这些不稳定的数据。

### 代码修改

在录制的元数据中记录一个 `warmup_duration_s` 字段：

```swift
// 在 RecordingDataModel 中添加
var warmupDurationS: Double = 5.0  // 默认 5 秒预热期
```

或者更简单：在 `metadata.json` 导出时添加：
```json
{
    "recording_settings": {
        "exposure_locked": true,
        "torch_level": 0.1,
        "warmup_duration_s": 5.0,
        "export_preset": "passthrough"
    }
}
```

后处理时（Python 端），自动跳过前 5 秒。

---

## 验证方法

修改 App 后，用以下方法验证改进效果：

### 1. 快速验证（App 端）
```
录制 30 秒，查看红通道均值：
- 锁定前：均值会有 15-25 units 的慢漂移
- 锁定后：均值应该稳定，只有 0.3-1.3 units 的周期性波动（PPG 脉搏波）
```

### 2. 定量验证（Python 端）
```python
# 读取新旧视频，对比 PPG 信号质量
import cv2
import numpy as np
from scipy.signal import periodogram

cap = cv2.VideoCapture('new_video.mp4')
red_means = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    red_means.append(frame[:,:,2].mean())  # BGR: red = index 2
cap.release()

signal = np.array(red_means)
freqs, psd = periodogram(signal, fs=30)

# 心率频段能量占比（应该 > 0.1，之前 < 0.01）
cardiac_mask = (freqs >= 0.7) & (freqs <= 3.5)
cardiac_ratio = psd[cardiac_mask].sum() / psd.sum()
print(f"Cardiac energy ratio: {cardiac_ratio:.4f}")
# 旧数据: ~0.005, 新数据目标: > 0.1
```

### 3. 对比实验
```
采集协议：
1. 同一受试者，同一时间，连续录制 2 次
2. 第一次：旧版 App（自动曝光）
3. 第二次：新版 App（锁定曝光）
4. 对比 PPG-ECG 相关性

预期：新版 PPG-ECG 相关性 r 从 ~0.05 提升到 > 0.3
```

---

## 修改文件总结

| 文件 | 修改内容 |
|------|---------|
| `CameraService.swift` | 新增 `lockExposureForPPG()`, `autoThenLockExposure()`, `lockPPGCameraExposure()` 方法；修改 `enableTorch()` 中的亮度从 1.0 → 0.1；修改 `convertToMP4()` 使用 Passthrough |
| `CameraViewModel.swift` | 在 `startActualRecording()` 或 `startCountdown()` 末尾调用 `lockPPGCameraExposure()` |
| `SettingsService.swift` | 默认视频质量改为 `AVAssetExportPresetHighestQuality`（如果不用 Passthrough） |
| `PPGSignalMonitor.swift`（新增） | 实时 PPG 信号质量监测 + RGB 均值 CSV 导出 |

---

## 注意事项

1. **锁定曝光时机**：3 秒倒计时期间用户应该已经把手指放在摄像头上，此时自动曝光正在适应手指的红色透光。倒计时结束时锁定，是最佳时机。

2. **不同 iPhone 的差异**：不同型号的 ISO 范围、曝光时间范围不同。建议用 `autoThenLockExposure()` 方案（先自动后锁定），而不是写死固定值。

3. **手电筒与曝光的交互**：手电筒亮度影响进光量，进而影响自动曝光的锁定值。建议先设置手电筒亮度，再等待自动曝光稳定，最后锁定。

4. **测试建议**：先在一台手机上验证（推荐 iPhone 15/14），录制几段 30 秒的测试视频，用 Python 检查红通道均值的稳定性。
