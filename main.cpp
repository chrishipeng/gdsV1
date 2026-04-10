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

namespace {

std::string escapeJson(const std::string& s) {
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
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
#ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
#endif
        RuntimeConfig cfg;
        parseRuntimeArgs(argc, argv, cfg);
        if (cfg.cameraImagePath.empty()) {
            throw std::runtime_error("Missing camera image path, pass --camera_image=<path>");
        }
        if (cfg.gdsPath.empty()) {
            throw std::runtime_error("Missing gds path, pass --gds=<path>");
        }
        std::filesystem::create_directories(cfg.resultDir);

        GdsProcessor processor(cfg.gdsPath);
        const std::string mirroredGdsPath =
            (std::filesystem::path(cfg.resultDir) / "gds_mirrored_for_validation.gds").string();
        if (cfg.exportMirroredGds) {
            processor.createMirroredTopCell(cfg.mirrorLeftRight, "MIRRORED_TOP");
            processor.saveLibraryAsGds(mirroredGdsPath);
        }

        const std::string gdsPngPath =
            (std::filesystem::path(cfg.resultDir) / "gds_scale1.png").string();
        cv::Mat gdsPng = processor.renderToImageByPixelSize(cfg.scale1, gdsPngPath, cfg.renderSupersample);

        cv::Rect roiGdsPng =
            GdsProcessor::selectRoiWithZoomPan(gdsPng, "Select ROI on GDS PNG", cfg.initialZoom);
        if (roiGdsPng.width <= 0 || roiGdsPng.height <= 0) {
            throw std::runtime_error("ROI is empty, please reselect");
        }

        // 模板中心 GDS 物理坐标(µm): 见 GdsProcessor::roiCenterToGds（与 gds_scale1 栅格化互逆）。
        const cv::Point2d roiCenterGdsUm = processor.roiCenterToGds(roiGdsPng, cfg.scale1);
        const double roiCenterPxX = roiGdsPng.x + 0.5 * roiGdsPng.width;
        const double roiCenterPxY = roiGdsPng.y + 0.5 * roiGdsPng.height;
        const std::string roiGdsJsonPath =
            (std::filesystem::path(cfg.resultDir) / "roi_center_gds_um.json").string();
        {
            std::ofstream j(roiGdsJsonPath);
            j << std::fixed << std::setprecision(6);
            j << "{\n";
            j << "  \"scale1_um_per_px\": " << cfg.scale1 << ",\n";
            j << "  \"roi_rect_px\": {\"x\": " << roiGdsPng.x << ", \"y\": " << roiGdsPng.y
              << ", \"w\": " << roiGdsPng.width << ", \"h\": " << roiGdsPng.height << "},\n";
            j << "  \"roi_center_px_on_gds_png\": {\"x\": " << roiCenterPxX << ", \"y\": " << roiCenterPxY
              << "},\n";
            j << "  \"gds_um\": {\"x\": " << roiCenterGdsUm.x << ", \"y\": " << roiCenterGdsUm.y << "}\n";
            j << "}\n";
        }
        std::cout << "ROI→GDS(µm): (" << roiCenterGdsUm.x << "," << roiCenterGdsUm.y << ") → " << roiGdsJsonPath
                  << std::endl;

        const std::string roiPath =
            (std::filesystem::path(cfg.resultDir) / "roi_selected.png").string();
        cv::Mat gdsTemplate = GdsProcessor::saveRoiImage(gdsPng, roiGdsPng, roiPath);

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

        std::ostringstream artifactLog;
        GdsProcessor::saveRegistrationArtifacts(cameraGray, reg, cfg.scale1, cfg.resultDir, &artifactLog);

        std::cout << "scale1=" << cfg.scale1 << " scale2=" << reg.scale2 << " final=" << reg.finalScale
                  << " peak_ratio=" << reg.peakRatio << " match_px=(" << reg.matchedCenterPx.x << ","
                  << reg.matchedCenterPx.y << ")\n";
        if (cfg.exportMirroredGds) {
            std::cout << mirroredGdsPath << std::endl;
        }

        bool hasMatchedStage = false;
        double matchedStageXmm = 0.0;
        double matchedStageYmm = 0.0;
        if (cfg.hasStageCenter) {
            constexpr double kImageCenterX = 640.0;
            constexpr double kImageCenterY = 512.0;
            const double finalScaleMmPerPx = reg.finalScale * 1e-3;
            const double dxPx = reg.matchedCenterPx.x - kImageCenterX;
            const double dyPx = reg.matchedCenterPx.y - kImageCenterY;
            matchedStageXmm = cfg.stageCenterXmm + dxPx * finalScaleMmPerPx;
            matchedStageYmm = cfg.stageCenterYmm - dyPx * finalScaleMmPerPx;
            hasMatchedStage = true;
            std::cout << "stage(mm): (" << matchedStageXmm << "," << matchedStageYmm << ")\n";
        }

        const std::string sampleCsvPath =
            (std::filesystem::path(cfg.resultDir) / "10_stage_mapping_samples.csv").string();
        if (hasMatchedStage) {
            const bool csvExists = std::filesystem::exists(sampleCsvPath);
            std::ofstream sampleCsv(sampleCsvPath, std::ios::app);
            if (!sampleCsv.is_open()) {
                std::cout << "CSV open failed: " << sampleCsvPath << std::endl;
            } else {
                if (!csvExists) {
                    sampleCsv << "matched_px_x,matched_px_y,matched_stage_x_mm,matched_stage_y_mm,"
                                 "gds_template_x_um,gds_template_y_um,scale1,scale2,final_scale,best_score,"
                                 "peak_ratio\n";
                }
                sampleCsv << std::fixed << std::setprecision(6) << reg.matchedCenterPx.x << ","
                          << reg.matchedCenterPx.y << "," << matchedStageXmm << "," << matchedStageYmm << ","
                          << roiCenterGdsUm.x << "," << roiCenterGdsUm.y << "," << cfg.scale1 << "," << reg.scale2
                          << "," << reg.finalScale << "," << reg.score << "," << reg.peakRatio << "\n";
                std::cout << sampleCsvPath << std::endl;
            }
        }

        const bool lowConfidence = (reg.peakRatio < 1.05);
        const std::string warningText =
            "警告: 当前匹配置信度较低(peak_ratio 过于接近 1.0)，建议重新选择 ROI。";
        const std::string consoleJsonPath =
            (std::filesystem::path(cfg.resultDir) / "09_console_log.json").string();
        std::ofstream consoleJson(consoleJsonPath);
        consoleJson << std::fixed << std::setprecision(6);
        consoleJson << "{\n";
        consoleJson << "  \"scale1\": " << cfg.scale1 << ",\n";
        consoleJson << "  \"scale2\": " << reg.scale2 << ",\n";
        consoleJson << "  \"final_scale\": " << reg.finalScale << ",\n";
        consoleJson << "  \"peak_ratio\": " << reg.peakRatio << ",\n";
        consoleJson << "  \"match_area_ratio\": " << reg.matchAreaRatio << ",\n";
        consoleJson << "  \"gds_template_um\": {\n";
        consoleJson << "    \"roi_center_px_on_gds_png\": {\"x\": " << roiCenterPxX << ", \"y\": "
                    << roiCenterPxY << "},\n";
        consoleJson << "    \"gds_um\": {\"x\": " << roiCenterGdsUm.x << ", \"y\": " << roiCenterGdsUm.y << "}\n";
        consoleJson << "  },\n";
        consoleJson << "  \"matched_center_px\": {\"x\": " << reg.matchedCenterPx.x << ", \"y\": "
                    << reg.matchedCenterPx.y << "},\n";
        if (cfg.hasStageCenter) {
            consoleJson << "  \"image_center_stage_mm\": {\"x\": " << cfg.stageCenterXmm << ", \"y\": "
                        << cfg.stageCenterYmm << "},\n";
        }
        if (hasMatchedStage) {
            consoleJson << "  \"matched_center_stage_mm\": {\"x\": " << matchedStageXmm << ", \"y\": "
                        << matchedStageYmm << "},\n";
        }
        consoleJson << "  \"paths\": {\n";
        if (cfg.exportMirroredGds) {
            consoleJson << "    \"mirrored_gds\": \"" << escapeJson(mirroredGdsPath) << "\",\n";
        }
        consoleJson << "    \"gds_png\": \"" << escapeJson(gdsPngPath) << "\",\n";
        consoleJson << "    \"roi_png\": \"" << escapeJson(roiPath) << "\",\n";
        consoleJson << "    \"roi_gds_json\": \"" << escapeJson(roiGdsJsonPath) << "\"\n";
        consoleJson << "  },\n";
        consoleJson << "  \"low_confidence\": " << (lowConfidence ? "true" : "false") << ",\n";
        consoleJson << "  \"warning\": \"" << (lowConfidence ? escapeJson(warningText) : "") << "\"\n";
        consoleJson << "}\n";

        std::cout << consoleJsonPath << "\n" << artifactLog.str();
        if (lowConfidence) {
            std::cout << warningText << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
