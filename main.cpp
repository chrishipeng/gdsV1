#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <opencv2/opencv.hpp>

#include "GdsProcessor.h"

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char* argv[]) {
    try {
#ifdef _WIN32
        // 统一控制台为 UTF-8，避免中文输出乱码。
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
#endif
        // Step1: 解析并校验运行参数。
        RuntimeConfig cfg;
        parseRuntimeArgs(argc, argv, cfg);
        if (cfg.cameraImagePath.empty()) {
            throw std::runtime_error("Missing camera image path, pass --camera_image=<path>");
        }
        if (cfg.gdsPath.empty()) {
            throw std::runtime_error("Missing gds path, pass --gds=<path>");
        }
        std::filesystem::create_directories(cfg.resultDir);

        // Step2: 读取 GDS，并按需要生成镜像 GDS 供后续验证。
        GdsProcessor processor(cfg.gdsPath);
        const std::string mirroredGdsPath =
            (std::filesystem::path(cfg.resultDir) / "gds_mirrored_for_validation.gds").string();
        if (cfg.exportMirroredGds) {
            processor.createMirroredTopCell(cfg.mirrorLeftRight, "MIRRORED_TOP");
            processor.saveLibraryAsGds(mirroredGdsPath);
        }

        // Step3: 将 GDS 渲染为图像并交互式选择 ROI。
        const std::string gdsPngPath =
            (std::filesystem::path(cfg.resultDir) / "gds_scale1.png").string();
        cv::Mat gdsPng = processor.renderToImageByPixelSize(cfg.scale1, gdsPngPath, cfg.renderSupersample);

        cv::Rect roiGdsPng =
            GdsProcessor::selectRoiWithZoomPan(gdsPng, "Select ROI on GDS PNG", cfg.initialZoom);
        if (roiGdsPng.width <= 0 || roiGdsPng.height <= 0) {
            throw std::runtime_error("ROI is empty, please reselect");
        }

        const std::string roiPath =
            (std::filesystem::path(cfg.resultDir) / "roi_selected.png").string();
        cv::Mat gdsTemplate = GdsProcessor::saveRoiImage(gdsPng, roiGdsPng, roiPath);

        // Step4: 读取相机图并执行配准。
        cv::Mat cameraGray = cv::imread(cfg.cameraImagePath, cv::IMREAD_GRAYSCALE);
        if (cameraGray.empty()) {
            throw std::runtime_error("Failed to read camera image: " + cfg.cameraImagePath);
        }
        cv::threshold(cameraGray, cameraGray, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        const GdsProcessor::RegistrationResult reg = processor.registerRoiToCamera(
            gdsTemplate, cameraGray, roiGdsPng, cfg.scale1, cfg.scaleSearchLow, cfg.scaleSearchHigh,
            cfg.scaleSearchStep);
        if (!reg.ok) {
            throw std::runtime_error("Template matching failed");
        }

        // Step5: 保存配准过程产物，便于质量评估。
        std::ostringstream artifactLog;
        GdsProcessor::saveRegistrationArtifacts(cameraGray, reg, cfg.scale1, cfg.resultDir, &artifactLog);

        // Step6: 输出关键指标和结果路径。
        std::cout << "粗匹配 scale1: " << cfg.scale1 << std::endl;
        std::cout << "配准 scale2: " << reg.scale2 << std::endl;
        std::cout << "最终比例(scale1*scale2): " << reg.finalScale << std::endl;
        std::cout << "peak_ratio(best/second): " << reg.peakRatio << std::endl;
        std::cout << "匹配面积占比: " << reg.matchAreaRatio << std::endl;
        std::cout << "匹配中心(像素): (" << reg.matchedCenterPx.x << ", " << reg.matchedCenterPx.y << ")"
                  << std::endl;
        std::cout << "映射 GDS 中心: (" << reg.centerGdsX << ", " << reg.centerGdsY << ")" << std::endl;
        if (cfg.exportMirroredGds) {
            std::cout << "镜像GDS: " << mirroredGdsPath << std::endl;
        }
        std::cout << "gds渲染图: " << gdsPngPath << std::endl;
        std::cout << "roi图: " << roiPath << std::endl;
        std::cout << artifactLog.str();
        const bool lowConfidence = (reg.peakRatio < 1.05);
        const std::string warningText =
            "警告: 当前匹配置信度较低(peak_ratio 过于接近 1.0)，建议重新选择 ROI。";
        const std::string consoleJsonPath =
            (std::filesystem::path(cfg.resultDir) / "09_console_log.json").string();
        auto escapeJson = [](const std::string& s) {
            std::string out;
            out.reserve(s.size() + 16);
            for (char c : s) {
                switch (c) {
                    case '\\':
                        out += "\\\\";
                        break;
                    case '\"':
                        out += "\\\"";
                        break;
                    case '\n':
                        out += "\\n";
                        break;
                    case '\r':
                        out += "\\r";
                        break;
                    case '\t':
                        out += "\\t";
                        break;
                    default:
                        out.push_back(c);
                        break;
                }
            }
            return out;
        };
        std::ofstream consoleJson(consoleJsonPath);
        consoleJson << std::fixed << std::setprecision(6);
        consoleJson << "{\n";
        consoleJson << "  \"scale1\": " << cfg.scale1 << ",\n";
        consoleJson << "  \"scale2\": " << reg.scale2 << ",\n";
        consoleJson << "  \"final_scale\": " << reg.finalScale << ",\n";
        consoleJson << "  \"peak_ratio\": " << reg.peakRatio << ",\n";
        consoleJson << "  \"match_area_ratio\": " << reg.matchAreaRatio << ",\n";
        consoleJson << "  \"matched_center_px\": {\"x\": " << reg.matchedCenterPx.x << ", \"y\": "
                    << reg.matchedCenterPx.y << "},\n";
        consoleJson << "  \"mapped_gds_center\": {\"x\": " << reg.centerGdsX << ", \"y\": " << reg.centerGdsY
                    << "},\n";
        consoleJson << "  \"paths\": {\n";
        if (cfg.exportMirroredGds) {
            consoleJson << "    \"mirrored_gds\": \"" << escapeJson(mirroredGdsPath) << "\",\n";
        }
        consoleJson << "    \"gds_png\": \"" << escapeJson(gdsPngPath) << "\",\n";
        consoleJson << "    \"roi_png\": \"" << escapeJson(roiPath) << "\"\n";
        consoleJson << "  },\n";
        consoleJson << "  \"artifact_log\": \"" << escapeJson(artifactLog.str()) << "\",\n";
        consoleJson << "  \"low_confidence\": " << (lowConfidence ? "true" : "false") << ",\n";
        consoleJson << "  \"warning\": \"" << (lowConfidence ? escapeJson(warningText) : "") << "\"\n";
        consoleJson << "}\n";

        std::cout << "控制台日志JSON: " << consoleJsonPath << std::endl;
        if (lowConfidence) {
            std::cout << warningText << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
