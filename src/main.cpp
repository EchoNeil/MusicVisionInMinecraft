#define _CRT_SECURE_NO_WARNINGS


#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <filesystem>
#include <cstdint>
#include <algorithm>
#include <typeinfo>

#define MINIMP3_IMPLEMENTATION
#include "minimp3.h"
#include "minimp3_ex.h"
#include "kiss_fft.h"

namespace fs = std::filesystem;

const double M_PI = std::acos(-1.0);


// 配置类，提供自由修改参数
struct VisualizationConfig {
    // 起始坐标（平面左下角）
    int baseX = 0;
    int baseY = 0;
    int baseZ = 0;

    // 平面尺寸（默认10x10）
    int width = 10;   // 水平方向（对应频段）
    int height = 10;  // 竖直方向（对应响度）

    // 方块类型（使用Minecraft命名空间ID）
    std::string blockType = "minecraft:ochre_froglight";
    std::string airType = "minecraft:air";

    // 采样率（自动从MP3获取，但可覆盖）
    int targetSampleRate = 48000;  // 默认，若MP3不同会重采样（简化：不重采样，直接使用原采样率）

    // 帧长：50ms对应的样本数
    int frameSamples(int sampleRate) const {
        return static_cast<int>(sampleRate * 0.05); // 50ms
    }

    // 频段划分（从低到高各频段的起始频率，10个柱子11个分隔 单位Hz

    std::vector<double> freqBands(int sampleRate) const 
    {
        std::vector<double> bands(width + 1);
        bands[0] = 20;
        bands[1] = 40;
        bands[2] = 80;
        bands[3] = 160;
        bands[4] = 315;
        bands[5] = 630;
        bands[6] = 1250;
        bands[7] = 2500;
        bands[8] = 5000;
        bands[9] = 10000;
        bands[10] = 20000;

        /*
        double nyquist = sampleRate / 2.0;
        for (int i = 0; i <= width; ++i) {
            bands[i] = i * nyquist / width;
        }
        */
        
        return bands;
    }
};

// ----------------------------------------------
// 音频解码器，使用minimp3
// ----------------------------------------------
class AudioDecoder {
public:
    bool load(const std::string& filename, std::vector<float>& pcm, int& sampleRate) 
    //filename - 文件路径  pcm - 存储解码后的音频数据  samplerate - 采样率
    {
        std::cout << "start decoder" << std::endl;
        mp3dec_t mp3d;
        mp3dec_init(&mp3d);

        // 打开文件
        FILE* file = fopen(filename.c_str(), "rb");
        if (!file) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }

        mp3dec_file_info_t info = { 0 };
        int result = mp3dec_load(&mp3d, filename.c_str(), &info, NULL, NULL);
        if (result != 0) {
            std::cerr << "解码失败，错误码: " << result << std::endl;
            fclose(file);
            return false;
        }
        if (info.buffer == nullptr) {
            // 注意：这里 info.buffer 应该由外部 free，抛出异常前最好也清理一下
            fclose(file); // 别忘了关闭文件
            throw std::runtime_error("info.buffer is nullptr");
        }

        sampleRate = info.hz;

        // --- 修复开始 ---
        // info.samples 是 buffer 中的总元素个数（所有声道之和）
        if (info.channels > 1) {
            // 计算单声道的样本数量（帧数）
            int frames = info.samples / info.channels;

            pcm.resize(frames);  //如果是立体声，需要将双声道数据拆分为单声道数据
            for (int i = 0; i < frames; ++i) {
                // 立体声数据是交错的：L, R, L, R...
                // 索引 i 对应的一对样本在 buffer 中的位置是 2*i 和 2*i+1
                float left = info.buffer[2 * i] / 32768.0f;
                float right = info.buffer[2 * i + 1] / 32768.0f;
                pcm[i] = (left + right) * 0.5f;
            }
        }
        else {
            // 单声道，直接复制
            pcm.resize(info.samples);
            for (int i = 0; i < info.samples; ++i) {
                pcm[i] = info.buffer[i] / 32768.0f;
            }
        }
        // --- 修复结束 ---

        free(info.buffer); // 由minimp3分配的内存
        fclose(file);
        return true;
    }
};

// ----------------------------------------------
// 频率分析器，使用KissFFT
// ----------------------------------------------
class FrequencyAnalyzer {
public:
    FrequencyAnalyzer(int sampleRate, const VisualizationConfig& config)
        : sampleRate_(sampleRate), config_(config) {
        bands_ = config_.freqBands(sampleRate_);
    }

    // 对一帧PCM数据计算各频段能量，返回长度为width的浮点数向量
    std::vector<double> computeFrameEnergies(const std::vector<float>& frame) {
        int N = frame.size();
        // 确保N是2的幂？KissFFT支持任意长度，但为了效率，可填充到2的幂。这里简化直接使用原始长度。
        // 应用汉宁窗
        std::vector<float> windowed(N);
        for (int i = 0; i < N; ++i) {
            double hanning = 0.5 * (1 - cos(2 * M_PI * i / (N - 1)));
            windowed[i] = frame[i] * hanning;
        }

        // 执行FFT
        kiss_fft_cfg cfg = kiss_fft_alloc(N, 0, nullptr, nullptr);
        std::vector<kiss_fft_cpx> in(N), out(N);
        for (int i = 0; i < N; ++i) {
            in[i].r = windowed[i];
            in[i].i = 0;
        }
        kiss_fft(cfg, in.data(), out.data());
        kiss_fft_free(cfg);

        // 计算幅度谱（只取正频率，0~N/2）
        int nfft = N / 2 + 1;
        std::vector<double> magnitude(nfft);
        for (int i = 0; i < nfft; ++i) {
            double real = out[i].r;
            double imag = out[i].i;
            magnitude[i] = sqrt(real * real + imag * imag);
        }

        // 将幅度谱映射到频段能量
        std::vector<double> energies(config_.width, 0.0);
        double freqPerBin = sampleRate_ / (double)N; // 每个FFT bin对应的频率宽度

        for (int bin = 0; bin < nfft; ++bin) {
            double freq = bin * freqPerBin;
            // 找到该bin属于哪个频段
            for (int b = 0; b < config_.width; ++b) {
                if (freq >= bands_[b] && freq < bands_[b + 1]) {
                    energies[b] += magnitude[bin];
                    break;
                }
            }
        }
        return energies;
    }

private:
    int sampleRate_;
    VisualizationConfig config_;
    std::vector<double> bands_;
};

// ----------------------------------------------
// 归一化器，将能量转换为0~10高度
// ----------------------------------------------
class Normalizer {
public:
    Normalizer(int width) : width_(width) {}

    // 传入所有帧的频段能量（vector<vector<double>>），计算每个频段的最大值，并归一化到0~10
    void computeGlobalMax(const std::vector<std::vector<double>>& allEnergies) {
        maxPerBand_.assign(width_, 0.0);
        for (const auto& frame : allEnergies) {
            for (int i = 0; i < width_; ++i) {
                if (frame[i] > maxPerBand_[i]) {
                    maxPerBand_[i] = frame[i];
                }
            }
        }
        // 避免除零
        for (int i = 0; i < width_; ++i) {
            if (maxPerBand_[i] == 0) maxPerBand_[i] = 1e-6;
        }
    }

    // 将一帧能量归一化为整数高度
    std::vector<int> normalizeFrame(const std::vector<double>& energies) {
        std::vector<int> heights(width_);
        for (int i = 0; i < width_; ++i) {
            double normalized = energies[i] / maxPerBand_[i] * 10.0; // 0~10
            heights[i] = static_cast<int>(std::round(normalized));
            if (heights[i] > 10) heights[i] = 10;
            if (heights[i] < 0) heights[i] = 0;
        }
        return heights;
    }

private:
    int width_;
    std::vector<double> maxPerBand_;
};

// ----------------------------------------------
// Minecraft数据包生成器
// ----------------------------------------------
class MinecraftDataPackGenerator {
public:
    MinecraftDataPackGenerator(const VisualizationConfig& config)
        : config_(config) {}

    // 生成数据包到指定目录
    bool generate(const std::string& outputDir,
        const std::vector<std::vector<int>>& frameHeights) {
        // 创建数据包目录结构
        fs::path packDir = outputDir;
        fs::path functionsDir = packDir / "data" / "music_vis" / "function";
        fs::create_directories(functionsDir);

        // 创建pack.mcmeta
        std::ofstream meta(packDir / "pack.mcmeta");
        if (!meta) {
            std::cerr << "无法创建pack.mcmeta" << std::endl;
            return false;
        }
        meta << "{\n"
            "  \"pack\": {\n"
            "    \"pack_format\": 15,\n"
            "    \"description\": \"Music visualization from MP3\"\n"
            "  }\n"
            "}\n";
        meta.close();

        // 生成每个tick的函数文件
        int tickCount = frameHeights.size();
        for (int tick = 0; tick < tickCount; ++tick) {
            std::string funcName = "tick_" + std::to_string(tick) + ".mcfunction";
            
            std::ofstream func(functionsDir / funcName);
            if (!func) {
                std::cerr << "无法创建函数文件: " << funcName << std::endl;
                return false;
            }

            const auto& heights = frameHeights[tick];

            // 对每一列（频段）每一行（高度）生成setblock
            for (int x = 0; x < config_.width; ++x) {
                int height = heights[x];
                for (int y = 0; y < config_.height; ++y) {
                    int worldX = config_.baseX + x;
                    int worldY = config_.baseY + y;
                    int worldZ = config_.baseZ;
                    std::string block;
                    if (y < height) {
                        block = config_.blockType;
                    }
                    else {
                        block = config_.airType;
                    }
                    func << "setblock " << worldX << " " << worldY << " " << worldZ
                        << " " << block << " replace\n";
                }
            }
            std::cout << "\r已生成帧数：" << tick;
            std::cout.flush(); // 确保立即输出
            func.close();
        }

        // 生成启动函数，使用schedule依次调用各个tick函数
        std::ofstream start(functionsDir / "start.mcfunction");
        if (!start) {
            std::cerr << "无法创建start.mcfunction" << std::endl;
            return false;
        }

        for (int tick = 0; tick < tickCount; ++tick) {
            // 每个tick函数执行后，调度下一个tick（最后一个除外）
            start << "schedule function music_vis:tick_" << tick << " " << tick << "t\n";
        }
        start.close();

        // 生成stop函数，使用schedule依次调用各个tick函数
        std::ofstream stop(functionsDir / "stop.mcfunction");
        if (!stop) {
            std::cerr << "无法创建stop.mcfunction" << std::endl;
            return false;
        }

        for (int tick = 0; tick < tickCount; ++tick) {
            // 每个tick函数执行后，调度下一个tick（最后一个除外）
            stop << "schedule clear music_vis:tick_" << tick << "\n";
        }
        start.close();


        // 也可以生成一个停止函数（可选）
        std::cout << "数据包生成成功，包含 " << tickCount << " 个tick。\n";
        return true;
    }

private:
    VisualizationConfig config_;
};

// ----------------------------------------------
// 主程序
// ----------------------------------------------
int main(int argc, char* argv[]) {
    
    
    std::cout << "程序启动" << std::endl;
    

    if (argc < 6) {
        std::cerr << "用法: " << argv[0] << " <input.mp3> <output_datapack_dir> <start_x> <start_y> <start_z>\n";
        return 1;
    }
    std::cout << "程序开始运行" << std::endl;
    std::string inputFile = argv[1];
    std::string outputDir = argv[2];
    
    std::cout << "输入文件: " << inputFile << std::endl;
    std::cout << "输出目录: " << outputDir << std::endl;

    // 配置（可通过命令行参数修改，这里硬编码示例）

    VisualizationConfig config;
    config.baseX = (int)argv[3];
    config.baseY = (int)argv[4];
    config.baseZ = (int)argv[5];
    std::cout << "配置设置完毕" << std::endl;
    
    
    // 1. 解码MP3
    AudioDecoder decoder;
    std::vector<float> pcm;
    int sampleRate;
    if (!decoder.load(inputFile, pcm, sampleRate)) {
        return 1;
    }
    std::cout << "采样率: " << sampleRate << " Hz, 总样本数: " << pcm.size() << std::endl;
    
    // 2. 分帧
    int frameSize = config.frameSamples(sampleRate);
    int totalFrames = pcm.size() / frameSize;
    if (totalFrames == 0) {
        std::cerr << "音频太短，无法分帧" << std::endl;
        return 1;
    }
    std::cout << "帧大小: " << frameSize << " 样本, 总帧数: " << totalFrames << std::endl;

    // 3. 频率分析
    FrequencyAnalyzer analyzer(sampleRate, config);
    std::vector<std::vector<double>> allEnergies(totalFrames);
    for (int f = 0; f < totalFrames; ++f) {
        int start = f * frameSize;
        int end = (std::min)(start + frameSize, static_cast<int>(pcm.size()));
        std::vector<float> frame(pcm.begin() + start, pcm.begin() + end);
        // 如果帧长不足，补零（最后一帧可能不足，简单补零）
        frame.resize(frameSize, 0.0f);
        allEnergies[f] = analyzer.computeFrameEnergies(frame);
    }

    // 4. 归一化
    Normalizer normalizer(config.width);
    normalizer.computeGlobalMax(allEnergies);
    std::vector<std::vector<int>> frameHeights(totalFrames);
    for (int f = 0; f < totalFrames; ++f) {
        frameHeights[f] = normalizer.normalizeFrame(allEnergies[f]);
    }

    // 5. 生成数据包
    MinecraftDataPackGenerator generator(config);
    if (!generator.generate(outputDir, frameHeights)) {
        return 1;
    }

    std::cin.get();

}
