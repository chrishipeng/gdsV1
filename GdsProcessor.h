#ifndef GDS_PROCESSOR_H
#define GDS_PROCESSOR_H

#include <filesystem>
#include <ostream>
#include <string>
#include <vector>

#include <gdstk/gdstk.hpp>
#include <opencv2/opencv.hpp>

struct RuntimeConfig {
    std::string gdsPath;
    std::string cameraImagePath;
    double initialZoom = 1.0;
    double scale1 = 1.48;
    double scaleSearchLow = 0.80;
    double scaleSearchHigh = 6.00;
    double scaleSearchStep = 0.1;
    int renderSupersample = 1;
    std::string resultDir = "registration_results";
    bool exportMirroredGds = true;
    bool mirrorLeftRight = true;
    bool hasStageCenter = false;
    double stageCenterXmm = 0.0;
    double stageCenterYmm = 0.0;
};

bool readCommandLineArg(int argc, char* argv[], const std::string& key, std::string& out);
void parseRuntimeArgs(int argc, char* argv[], RuntimeConfig& cfg);

class ZoomPanRoiSelector {
public:
    explicit ZoomPanRoiSelector(const cv::Mat& srcGray);
    cv::Rect run(const std::string& windowName, double initialZoom);

private:
    static constexpr double kMinZoom = 0.01;
    static constexpr double kMaxZoom = 30.0;
    static constexpr int kMaxInteractiveDim = 2800;
    static void mouseCallback(int event, int x, int y, int flags, void* userdata);
    cv::Point toImagePoint(const cv::Point& viewPt) const;
    void onMouse(int event, int x, int y, int flags);
    void render();

    cv::Mat src;
    std::string winName;
    double zoom = 1.0;
    cv::Point2d pan{0.0, 0.0};
    bool selecting = false;
    bool panning = false;
    bool confirmed = false;
    bool cancel = false;
    bool mirrorX = false;
    bool mirrorY = false;
    cv::Point selectStartImg{0, 0};
    cv::Point selectNowImg{0, 0};
    cv::Point panStartMouse{0, 0};
    cv::Point2d panStartOffset{0.0, 0.0};
    cv::Rect selectedRoi;
    cv::Size displaySize{0, 0};
    cv::Rect contentRect{0, 0, 0, 0};
    cv::Size originalSize{0, 0};
    double coordScaleToOriginal = 1.0;
    cv::Mat cachedVis;
    bool cacheValid = false;
    bool cachedMirrorX = false;
    bool cachedMirrorY = false;
    bool dirty = true;
};

class GdsProcessor {
public:
    struct GdsBBox {
        double min_x = 0.0;
        double min_y = 0.0;
        double max_x = 0.0;
        double max_y = 0.0;
    };

    struct RegistrationResult {
        bool ok = false;
        double score = 0.0;
        double scale2 = 1.0;
        double refinedScale2 = 1.0;
        double finalScale = 1.0;
        double peakRatio = 0.0;
        double matchAreaRatio = 0.0;
        cv::Rect matchedRect;
        cv::Point2d matchedCenterPx;
        std::vector<double> testedScales;
        std::vector<double> testedScores;
        cv::Mat bestTemplateGray;
        cv::Mat bestResponseMap;
    };

    explicit GdsProcessor(const std::string& filePath);
    ~GdsProcessor();

    GdsBBox getBoundingBox() const;
    cv::Mat renderToImageByPixelSize(double pixelSizeUm, const std::string& outputPath,
                                     int supersample = 1) const;
    void createMirroredTopCell(bool mirrorLeftRight, const std::string& mirroredCellName);
    void saveLibraryAsGds(const std::string& outputPath) const;
    static cv::Rect selectRoiWithZoomPan(const cv::Mat& srcGray, const std::string& windowName,
                                         double initialZoom);
    static cv::Mat saveRoiImage(const cv::Mat& srcGray, const cv::Rect& roi, const std::string& outputPath);
    RegistrationResult registerRoiToCamera(const cv::Mat& roiTemplateGray, const cv::Mat& cameraGray,
                                           const cv::Rect& roiOnGdsImage, double scale1,
                                           double scaleSearchLow, double scaleSearchHigh,
                                           double scaleSearchStep) const;
    /// PNG 上 ROI 矩形中心 → GDS 库物理坐标(µm)；实现见 .cpp 中与 renderToImageByPixelSize 互逆。
    cv::Point2d roiCenterToGds(const cv::Rect& roiOnGdsImage, double pixelSizeUm) const;
    static void saveRegistrationArtifacts(const cv::Mat& cameraGray, const RegistrationResult& reg,
                                          double scale1, const std::string& resultDir,
                                          std::ostream* log = nullptr);

private:
    gdstk::Library m_library = {};
    gdstk::Cell* m_activeCell = nullptr;
};

#endif
