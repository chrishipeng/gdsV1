#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace stage_mapping_fitter {

struct PixelStageSample {
    double px = 0.0;
    double py = 0.0;
    double sx_mm = 0.0;
    double sy_mm = 0.0;
};

struct Affine2D {
    // sx = a*px + b*py + tx
    // sy = c*px + d*py + ty
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double d = 0.0;
    double tx = 0.0;
    double ty = 0.0;
};

struct FitReport {
    Affine2D affine;
    size_t sampleCount = 0;
    double rmsErrorMm = 0.0;
    double maxErrorMm = 0.0;
};

static bool readSamplesFromCsv(const std::string& csvPath, std::vector<PixelStageSample>& samples,
                               std::string* errMsg = nullptr) {
    std::ifstream in(csvPath);
    if (!in.is_open()) {
        if (errMsg) *errMsg = "无法打开CSV: " + csvPath;
        return false;
    }

    std::string line;
    bool firstLine = true;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (firstLine) {
            firstLine = false;
            if (line.find("matched_px_x") != std::string::npos) continue;
        }

        std::stringstream ss(line);
        std::string c0, c1, c2, c3;
        if (!std::getline(ss, c0, ',')) continue;
        if (!std::getline(ss, c1, ',')) continue;
        if (!std::getline(ss, c2, ',')) continue;
        if (!std::getline(ss, c3, ',')) continue;

        PixelStageSample s;
        s.px = std::stod(c0);
        s.py = std::stod(c1);
        s.sx_mm = std::stod(c2);
        s.sy_mm = std::stod(c3);
        samples.push_back(s);
    }

    if (samples.size() < 3) {
        if (errMsg) *errMsg = "样本数不足，至少需要3组点";
        return false;
    }
    return true;
}

static cv::Vec3d solveAxisByLeastSquares(const cv::Mat& A, const cv::Mat& b) {
    cv::Mat x;
    if (!cv::solve(A, b, x, cv::DECOMP_SVD)) {
        throw std::runtime_error("最小二乘求解失败");
    }
    return cv::Vec3d(x.at<double>(0, 0), x.at<double>(1, 0), x.at<double>(2, 0));
}

static FitReport fitAffine(const std::vector<PixelStageSample>& samples, const std::string& residualCsvPath) {
    const int n = static_cast<int>(samples.size());
    cv::Mat A(n, 3, CV_64F);
    cv::Mat bx(n, 1, CV_64F);
    cv::Mat by(n, 1, CV_64F);

    for (int i = 0; i < n; ++i) {
        A.at<double>(i, 0) = samples[i].px;
        A.at<double>(i, 1) = samples[i].py;
        A.at<double>(i, 2) = 1.0;
        bx.at<double>(i, 0) = samples[i].sx_mm;
        by.at<double>(i, 0) = samples[i].sy_mm;
    }

    const cv::Vec3d cx = solveAxisByLeastSquares(A, bx);
    const cv::Vec3d cy = solveAxisByLeastSquares(A, by);

    FitReport report;
    report.sampleCount = samples.size();
    report.affine.a = cx[0];
    report.affine.b = cx[1];
    report.affine.tx = cx[2];
    report.affine.c = cy[0];
    report.affine.d = cy[1];
    report.affine.ty = cy[2];

    double sumSq = 0.0;
    double maxErr = 0.0;

    std::ofstream residualOut(residualCsvPath);
    residualOut << "px,py,stage_x_gt_mm,stage_y_gt_mm,stage_x_pred_mm,stage_y_pred_mm,err_mm\n";
    for (const auto& s : samples) {
        const double predX = report.affine.a * s.px + report.affine.b * s.py + report.affine.tx;
        const double predY = report.affine.c * s.px + report.affine.d * s.py + report.affine.ty;
        const double ex = predX - s.sx_mm;
        const double ey = predY - s.sy_mm;
        const double err = std::sqrt(ex * ex + ey * ey);
        sumSq += err * err;
        maxErr = std::max(maxErr, err);
        residualOut << std::fixed << std::setprecision(6) << s.px << "," << s.py << "," << s.sx_mm << ","
                    << s.sy_mm << "," << predX << "," << predY << "," << err << "\n";
    }

    report.rmsErrorMm = std::sqrt(sumSq / samples.size());
    report.maxErrorMm = maxErr;
    return report;
}

static bool writeReportJson(const std::string& jsonPath, const std::string& sourceCsvPath,
                            const std::string& residualCsvPath, const FitReport& report,
                            std::string* errMsg = nullptr) {
    std::ofstream out(jsonPath);
    if (!out.is_open()) {
        if (errMsg) *errMsg = "无法写入JSON: " + jsonPath;
        return false;
    }

    out << std::fixed << std::setprecision(9);
    out << "{\n";
    out << "  \"model\": \"pixel_to_stage_affine\",\n";
    out << "  \"sample_count\": " << report.sampleCount << ",\n";
    out << "  \"matrix\": {\n";
    out << "    \"a\": " << report.affine.a << ",\n";
    out << "    \"b\": " << report.affine.b << ",\n";
    out << "    \"tx\": " << report.affine.tx << ",\n";
    out << "    \"c\": " << report.affine.c << ",\n";
    out << "    \"d\": " << report.affine.d << ",\n";
    out << "    \"ty\": " << report.affine.ty << "\n";
    out << "  },\n";
    out << "  \"rms_error_mm\": " << report.rmsErrorMm << ",\n";
    out << "  \"max_error_mm\": " << report.maxErrorMm << ",\n";
    out << "  \"sample_csv\": \"" << sourceCsvPath << "\",\n";
    out << "  \"residual_csv\": \"" << residualCsvPath << "\"\n";
    out << "}\n";
    return true;
}

// 对外入口：读取CSV -> 拟合 -> 输出JSON和残差CSV
bool fitPixelToStageAffineFromCsv(const std::string& sampleCsvPath, const std::string& outputJsonPath,
                                  const std::string& residualCsvPath, FitReport* outReport = nullptr,
                                  std::string* errMsg = nullptr) {
    try {
        std::vector<PixelStageSample> samples;
        if (!readSamplesFromCsv(sampleCsvPath, samples, errMsg)) return false;
        FitReport report = fitAffine(samples, residualCsvPath);
        if (!writeReportJson(outputJsonPath, sampleCsvPath, residualCsvPath, report, errMsg)) return false;
        if (outReport) *outReport = report;
        return true;
    } catch (const std::exception& e) {
        if (errMsg) *errMsg = std::string("拟合失败: ") + e.what();
        return false;
    }
}

// 可选工具函数：用已拟合矩阵预测任意像素点对应stage坐标
cv::Point2d predictStageFromPixel(const Affine2D& affine, const cv::Point2d& pixel) {
    return cv::Point2d(affine.a * pixel.x + affine.b * pixel.y + affine.tx,
                       affine.c * pixel.x + affine.d * pixel.y + affine.ty);
}

}  // namespace stage_mapping_fitter

